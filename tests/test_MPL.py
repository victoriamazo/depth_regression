import itertools
import numbers
import os
import shutil
import time
from collections import OrderedDict
import pandas as pd
import numpy as np

import dataloaders.img_transforms as transforms
from dataloaders.dataloader_builder import DataLoader
from metrics.metric_builder import Metric
from models.model_builder import Model
from tests.test_builder import Test

import torch
import torch.nn.functional as F
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()



class test_MPL(Test):
    def __init__(self, FLAGS):
        super(test_MPL, self).__init__(FLAGS)
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.metric = FLAGS.metric

        # seed
        if hasattr(FLAGS, 'seed'):
            self.seed = FLAGS.seed
        torch.manual_seed(self.seed)
        if use_cuda:
            torch.cuda.manual_seed(self.seed)


    def build(self):
        self._check_args()

        model = Model.model_builder(self.model, self.FLAGS)

        model_path = os.path.join(self.ckpts_dir, 'latest.pth')
        if self.worker_num != None:
            model_path = os.path.join(self.ckpts_dir, 'latest_{}.pth'.format(self.worker_num))
        find_best = True
        # load ckpt at a certain iteration
        if self.load_ckpt_iter != '' and isinstance(int(self.load_ckpt_iter), numbers.Number):
            for ckpt in os.listdir(self.ckpts_dir):
                if ckpt.startswith('best_{}'.format(self.load_ckpt_iter)):
                    model_path = os.path.join(self.ckpts_dir, ckpt)
                    find_best = False
        # load model
        if os.path.isfile(model_path):
            ckpt_dict = torch.load(model_path)
            epoch = ckpt_dict['epoch']
            train_iter = ckpt_dict['iteration']
            train_acc = ckpt_dict['train_acc']
            train_loss = ckpt_dict['train_loss']
            assert isinstance(ckpt_dict['state_dict'], (dict, OrderedDict)), type(ckpt_dict['state_dict'])
            model.load_state_dict(ckpt_dict['state_dict'])
            print('loaded checkpoint {} (iteration {}, epoch {}, train_acc {}%)'.format(model_path, train_iter, epoch,
                                                                                       train_acc))

            model.eval()
            if use_cuda:
                model.cuda()

            # run test
            t_begin = time.time()
            try:
                # load test dataset
                test_loader, num_samples = DataLoader.dataloader_builder(self.data_loader, self.FLAGS, 'test')

                test_loss = 0
                num_iters_per_epoch = int(num_samples / self.batch_size)
                x, y = [], []
                # data is a np array of shape (batch_size, height, width)
                # target is np array of shape (batch_size)
                for iteration, (data, target, _) in enumerate(test_loader):
                    if iteration == num_iters_per_epoch:
                        break

                    # transform to pytorch tensor
                    totensor = transforms.Compose([transforms.ToTensor(), ])
                    data_t = totensor(data)                                   # (batch_size, num_channels, height, width)
                    target_t = torch.from_numpy(target)
                    target_t = target_t.type(torch.LongTensor)
                    target_t = Variable(target_t)
                    data_t = Variable(data_t)
                    if use_cuda:
                        data_t, target_t = data_t.cuda(), target_t.cuda()

                    # run test and get predictions
                    output = model(data_t)
                    test_loss += F.cross_entropy(output, target_t).data[0]
                    x.append(output.data.cpu().numpy()[:])
                    y.append(target[:])

                # calculate accuracy
                test_loss = test_loss / num_iters_per_epoch                    # average over number of mini-batch
                logits = np.array(list(itertools.chain(*x)))
                y = np.array(list(itertools.chain(*y)))
                acc = Metric.metric_builder(self.metric, logits, y, self.FLAGS)
                print('\t\t\tTest: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, acc))

                # write accuracy to table
                results_table_path = os.path.join(self.train_dir, 'results.csv')
                if self.worker_num != None:
                    results_table_path = os.path.join(self.train_dir, 'results_{}.csv'.format(self.worker_num))

                row_idx = 0
                if not os.path.isfile(results_table_path):
                    columns = ['iter', 'test_acc', 'train_acc', 'train_loss']
                    index = np.arange(1)
                    results_table = pd.DataFrame(columns=columns, index=index)
                else:
                    results_table = pd.read_csv(results_table_path, index_col=0)
                    if len(results_table.index) > 0:
                        row_idx = len(results_table.index)
                results_table.ix[row_idx, 'iter'] = train_iter
                results_table.ix[row_idx, 'test_acc'] = acc
                results_table.ix[row_idx, 'train_acc'] = train_acc
                results_table.ix[row_idx, 'train_loss'] = train_loss
                results_table.to_csv(results_table_path, index=True)

                # save model, if it is the best
                if find_best:
                    best_acc = 0
                    ckpt_best = ''
                    for ckpt in os.listdir(self.ckpts_dir):
                        prefix = ckpt[:-4]
                        if ckpt.split('_') != None and len(ckpt.split('_')) > 2:
                            best_acc_tmp = float(prefix.split('_')[2])
                            if best_acc_tmp > best_acc:
                                best_acc = best_acc_tmp
                                ckpt_best = ckpt
                    if acc > best_acc:
                        shutil.copyfile(model_path, os.path.join(self.ckpts_dir, 'best_{}_{}.pth'.format(train_iter, acc)))
                        if ckpt_best != '':
                            os.remove(os.path.join(self.ckpts_dir, ckpt_best))
            except Exception as e:
                import traceback
                traceback.print_exc()
            finally:
                print("Total Elapse (test): {:.2f} s".format(time.time() - t_begin))
        else:
            print("no checkpoint found at {}".format(model_path))
































