import time
import os
from collections import OrderedDict
from tensorboardX import SummaryWriter                                 #  $ pip install tensorboardX
import pandas as pd
from shutil import copyfile
import numpy as np

from dataloaders.dataloader_builder import DataLoader
from models.model_builder import Model
from trains.train_builder import Train
from metrics.metric_builder import Metric
import dataloaders.img_transforms as transforms

import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
# print('PyTorch version: ', torch.__version__)



class train_MPL(Train):
    def __init__(self, FLAGS):
        super(train_MPL, self).__init__(FLAGS)
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.model = FLAGS.model
        self.loss = FLAGS.loss
        if len(FLAGS.decreasing_lr_epochs) > 0:
            self.decreasing_lr_epochs = list(map(int, FLAGS.decreasing_lr_epochs.split(',')))
        else:
            self.decreasing_lr_epochs = None
        self.weight_decay = FLAGS.weight_decay
        self.metric = FLAGS.metric

        self.load_ckpt = ''
        self.rm_train_dir = True
        if hasattr(FLAGS, 'load_ckpt') and FLAGS.load_ckpt != '':
            self.load_ckpt = FLAGS.load_ckpt
            self.rm_train_dir = False
        if self.worker_num != None:
            self.rm_train_dir = False
            self.writer = None
        else:
            self.writer = SummaryWriter(self.save_path)

        # seed
        if hasattr(FLAGS, 'seed'):
            self.seed = FLAGS.seed
        torch.manual_seed(self.seed)
        if use_cuda:
            torch.cuda.manual_seed(self.seed)


    def build(self):
        self._check_args(self.rm_train_dir)

        test_iters = {}
        results_table_path = os.path.join(self.train_dir, 'results.csv')
        results_table_path_tmp = os.path.join(self.train_dir, 'results_tmp.csv')
        if self.worker_num != None:
            results_table_path = os.path.join(self.train_dir, 'results_{}.csv'.format(self.worker_num))
            results_table_path_tmp = os.path.join(self.train_dir, 'results_tmp_{}.csv'.format(self.worker_num))

        line = ''
        args_dict = {arg: getattr(self.FLAGS, arg) for arg in vars(self.FLAGS)}
        for key, value in sorted(args_dict.items()):
            line += '{}={}, '.format(key, value)
        if self.worker_num == None:
            self.writer.add_text('Text', line, 0)

        # load model
        model = Model.model_builder(self.model, self.FLAGS)

        # optimizer
        optimizer = optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)

        # resume training
        epoch, train_iter = 0, 0
        if self.load_ckpt != '':
            model_path = os.path.join(self.ckpts_dir, '{}.pth'.format(self.load_ckpt))
            if os.path.isfile(model_path):
                ckpt_dict = torch.load(model_path)
                epoch = ckpt_dict['epoch']
                train_iter = ckpt_dict['iteration']
                train_acc = ckpt_dict['train_acc']
                assert isinstance(ckpt_dict['state_dict'], (dict, OrderedDict)), type(ckpt_dict['state_dict'])
                assert isinstance(ckpt_dict['optimizer'], (dict, OrderedDict)), type(ckpt_dict['optimizer'])
                model.load_state_dict(ckpt_dict['state_dict'])
                optimizer.load_state_dict(ckpt_dict['optimizer'])
                print('Training resumed from the ckpt {} (iteration {}, epoch {}, train_acc {}%)'.format(
                    model_path, train_iter, epoch, train_acc))
                train_iter += 1
        model.train()
        if use_cuda:
            model.cuda()      

        # train
        # print('decreasing lr epochs: ' + str(self.decreasing_lr_epochs))
        best_acc, old_file = 0, None
        t_begin = time.time()
        try:
            # load train data_tset
            train_loader, num_samples = DataLoader.dataloader_builder(self.data_loader, self.FLAGS, 'train')  #.build()

            batch_idx = 0
            num_iters_per_epoch = int(num_samples/self.batch_size)
            if self.num_iters_for_ckpt == 0:
                self.num_iters_for_ckpt = train_iter + num_iters_per_epoch * self.num_epochs - 1
            # data is a np array of shape (batch_size, height, width); target is np array of shape (batch_size)
            for iteration, (data, target, _) in enumerate(train_loader):
                iteration += train_iter
                print('iteration = ', iteration)  #, ' train_iter = ', train_iter
                if iteration == train_iter + num_iters_per_epoch * self.num_epochs:
                    break
                if iteration % (num_iters_per_epoch-1) == 0 and iteration != 0:
                    epoch += 1
                    batch_idx = 0

                # transform to pytorch tensor
                totensor = transforms.Compose([transforms.ToTensor(), ])
                data_t = totensor(data)                                 # (batch_size, num_channels, height, width)
                target_t = torch.from_numpy(target)
                target_t = target_t.type(torch.LongTensor)
                target_t = Variable(target_t)
                data_t = Variable(data_t)
                if use_cuda:
                    data_t, target_t = data_t.cuda(), target_t.cuda()
                if self.worker_num == None:
                    img = ((data[0]-np.min(data[0]))/np.max(data[0]-np.min(data[0]))).astype('float32')  #.reshape(self.height, self.width, 3)
                    img = np.stack((img, img, img), axis=2)
                    self.writer.add_image('train_images', img, iteration)

                # run model, get prediction and backprop error
                optimizer.zero_grad()
                output = model(data_t)
                loss = F.cross_entropy(output, target_t)
                loss.backward()
                optimizer.step()

                # calculate metric and save ckpt
                if iteration % self.num_iters_for_ckpt == 0 and iteration > 0:
                    # calculate metric
                    logits = output.data.cpu().numpy()
                    acc = Metric.metric_builder(self.metric, logits, target, self.FLAGS)
                    train_loss = loss.data[0]/self.batch_size
                    print('Train Epoch: {} (iter {}, {}/{}) Loss: {:.4f} Acc: {:.2f}%'.format(
                        epoch, iteration, batch_idx, num_iters_per_epoch, train_loss, acc))

                    # add to tensorboard
                    if self.worker_num == None:
                        self.writer.add_scalar('loss', train_loss, iteration)
                        self.writer.add_scalar('train_accuracy', acc, iteration)

                    # add test accuracy to tensorboard and results_table
                    if os.path.isfile(results_table_path):
                        copyfile(results_table_path, results_table_path_tmp)
                        results_table = pd.read_csv(results_table_path_tmp, index_col=0)
                        if len(results_table.index) > 0:
                            for row_idx in range(len(results_table.index)):
                                test_iter = int(results_table.iloc[row_idx]['iter'])
                                if not test_iter in test_iters:
                                    test_acc = results_table.iloc[row_idx]['test_acc']
                                    if self.worker_num == None:
                                        self.writer.add_scalar('test_accuracy', test_acc, test_iter)
                                    test_iters[test_iter] = test_acc

                    # save latest ckpt
                    model_path = os.path.join(self.ckpts_dir, 'latest.pth')
                    if self.worker_num != None:
                        model_path = os.path.join(self.ckpts_dir, 'latest_{}.pth'.format(self.worker_num))
                    if model_path != None:
                        ckpt_dict = {'epoch': epoch,
                                      'iteration': iteration,
                                      'state_dict': model.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'train_acc': acc,
                                      'train_loss': train_loss,
                                      }
                        torch.save(ckpt_dict, model_path)
                        print('ckpt {} saved'.format(model_path))
                batch_idx += 1

        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))

        if os.path.isfile(results_table_path_tmp):
            os.remove(results_table_path_tmp)








































            # # # loss
        # # total_loss = loss_builder(self.FLAGS.loss, logits, y, self.FLAGS).build()
        # #


            # # select gpu
            # args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
            # args.ngpu = len(args.gpu)
            #
            # # logger
            # print = misc.logger.info
            # misc.ensure_dir(args.logdir)


            # optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


            # model = torch.nn.DataParallel(model, device_ids=self.gpus)


            # print 'lr: {:.2e}'.format(optimizer.param_groups[0]['lr'])


            # indx_target_t = target_t.clone()
            # pred = output.data_t.max(1)[1]                # get the index of the max log-probability
            # correct = pred.cpu().eq(indx_target_t).sum()
            # acc = correct * 100.0 / len(data_t)