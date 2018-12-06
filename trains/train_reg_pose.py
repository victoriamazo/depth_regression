import time
import os
from collections import OrderedDict
from tensorboardX import SummaryWriter                                 # pip install tensorboardX
import csv

from dataloaders.dataloader_builder import DataLoader
from models.model_builder import Model
from trains.train_builder import Train
from utils.auxiliary import AverageMeter
from trains.losses import loss_regression
from utils.auxiliary import tensor2array, save_checkpoint, save_test_losses_to_tensorboard, write_summary_to_csv

import torch
from torch.autograd import Variable
import torch.utils.data as data
import torch.backends.cudnn as cudnn
use_cuda = torch.cuda.is_available()
print('PyTorch version: ', torch.__version__)



class train_reg_pose(Train):
    def __init__(self, FLAGS):
        super(train_reg_pose, self).__init__(FLAGS)
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.beta = FLAGS.beta
        self.model = FLAGS.model
        if len(FLAGS.decreasing_lr_epochs) > 0:
            self.decreasing_lr_epochs = list(map(int, FLAGS.decreasing_lr_epochs.split(',')))
        else:
            self.decreasing_lr_epochs = None
        self.weight_decay = FLAGS.weight_decay
        self.euler_angles = FLAGS.euler_angles
        self.rotation_mode = 'euler'
        if not self.euler_angles:
            self.rotation_mode = 'quat'
        self.num_iters_for_print = FLAGS.num_iters_for_print
        if hasattr(FLAGS, 'seed'):
            self.seed = FLAGS.seed
        self.debug = FLAGS.debug
        self.rm_train_dir = True
        if self.load_ckpt != '':
            self.rm_train_dir = False
        if self.worker_num != None:
            self.rm_train_dir = False
            self.debug = True
        if not self.debug:
            save_path = os.path.join('tensorboard', self.train_dir.split('/')[-1])
            self.writer = SummaryWriter(save_path)
        else:
            self.writer = None
        self.results_table_path = os.path.join(self.train_dir, 'results.csv')
        self.results_table_path_tmp = os.path.join(self.train_dir, 'results_tmp.csv')
        self.loss_summary_path = os.path.join(self.train_dir, 'loss_summary.csv')
        self.loss_full_path = os.path.join(self.train_dir, 'loss_full.csv')
        if self.worker_num != None:
            self.results_table_path = os.path.join(self.train_dir, 'results_{}.csv'.format(self.worker_num))
            self.results_table_path_tmp = os.path.join(self.train_dir, 'results_tmp_{}.csv'.format(self.worker_num))
        self.n_iter = 0
        self.n_epoch = 0
        self.test_iters_dict = {}
        self.test_total_loss_best = 0
        torch.manual_seed(self.seed)
        if use_cuda:
            torch.cuda.manual_seed(self.seed)

        # dataloader
        assert FLAGS.with_gt == 1
        self.dataloader, self.num_samples = DataLoader.dataloader_builder(FLAGS.data_loader, FLAGS, 'train',
                                                                     parent_class=data.Dataset)
        self.epoch_size = self.num_samples/self.batch_size
        print('Number of iterations per epoch: ', self.epoch_size)



    def _train_one_epoch(self, pose_exp_net, optimizer):
        '''
            train for one epoch
        '''
        one_iter_time = AverageMeter()                         # time to execute one iteration
        batch_load_time = AverageMeter()                          # time to load a data batch
        losses = AverageMeter(precision=4)
        test_total_loss = -1

        # switch to train mode
        pose_exp_net.train()

        end = time.time()
        for i, (tgt_img_l, ref_img_l, _, _, gt_trg_pose, gt_ref_pose, filename_tgt, filename_ref) in enumerate(self.dataloader):
            batch_load_time.update(time.time() - end)
            tgt_img_l_cpu = tgt_img_l
            ref_img_l_cpu = ref_img_l
            if use_cuda:
                tgt_img_l = tgt_img_l.cuda()
                ref_img_l = ref_img_l.cuda()
            tgt_img_l_var = Variable(tgt_img_l)
            ref_img_l_var = [Variable(ref_img_l)]

            # compute output
            _, pred_pose_delta = pose_exp_net(tgt_img_l_var, ref_img_l_var)             #(tx, ty, tz, rx, ry, rz)

            # compute loss
            gt_pose_delta = Variable(gt_ref_pose.view(-1, 6).type(torch.FloatTensor) - gt_trg_pose.view(-1, 6).type(torch.FloatTensor))                             #(tx, ty, tz, rx, ry, rz)
            if use_cuda:
                gt_pose_delta = gt_pose_delta.cuda()
            loss = loss_regression(pred_pose_delta.view(-1, 6), gt_pose_delta, beta=self.beta)


            if i > 0 and self.n_iter % self.num_iters_for_print == 0:
                if not self.debug:
                    self.writer.add_scalar('total_loss_train', loss.data[0], self.n_iter)
                    if self.n_iter % (self.num_iters_for_print*10) == 0:
                        self.writer.add_image('trg_img_train', tensor2array(tgt_img_l_cpu[0]), self.n_iter)
                        self.writer.add_image('ref_img_train', tensor2array(ref_img_l_cpu[0]), self.n_iter)

            if i > 0 and self.n_iter % self.num_iters_for_ckpt == 0:
                # get test losses from the results_table and add them to tensorboard
                is_best, self.test_iters_dict, self.test_total_loss_best, test_total_loss = save_test_losses_to_tensorboard(
                                self.results_table_path, self.results_table_path_tmp, self.test_total_loss_best,
                                self.debug, test_iters_dict=self.test_iters_dict, writer=self.writer)
                # save checkpoint
                state_names = ['exp_pose']
                states = [{'iteration': self.n_iter, 'epoch': self.n_epoch, 'state_dict': pose_exp_net.module.state_dict()}]
                save_checkpoint(self.ckpts_dir, states, state_names)

            # record loss, compute gradient and do Adam step
            losses.update(loss.data[0], self.batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            one_iter_time.update(time.time() - end)
            end = time.time()
            if self.n_iter % self.num_iters_for_print == 0:
                with open(self.loss_full_path, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow([self.n_iter, loss.data[0]])
                print('Train:  epoch {} (iter {}),  time for iter {},  mean loss {}'.format(self.n_epoch, self.n_iter,
                                                                                            one_iter_time, losses))

            if i >= self.epoch_size - 1:
                break
            self.n_iter += 1

        return losses.avg[0], test_total_loss


    def build(self):
        self._check_args(self.rm_train_dir)
        write_summary_to_csv(self.FLAGS, self.writer, self.loss_summary_path, self.loss_full_path, loss_names=[])

        # load models
        pose_exp_net = Model.model_builder(self.model, self.FLAGS)
        if self.debug:
            print('\nModel "{}": \n{}'.format(self.model, pose_exp_net))
        if use_cuda:
            pose_exp_net = pose_exp_net.cuda()

        # load or init weights
        train_iter_poseexp = 0
        if self.load_ckpt != '':
            model_path = self.load_ckpt
            if os.path.isfile(model_path):
                ckpt_dict = torch.load(model_path)
                train_iter_poseexp = ckpt_dict['iteration']
                assert isinstance(ckpt_dict['state_dict'], (dict, OrderedDict)), type(ckpt_dict['state_dict'])
                pose_exp_net.load_state_dict(ckpt_dict['state_dict'], strict=False)
                print('PoseExpNet training resumed from the ckpt {} (iter {})'.format(model_path, train_iter_poseexp))
        else:
            pose_exp_net.init_weights()
        if train_iter_poseexp > 0:
            self.n_iter += train_iter_poseexp + 1
            self.n_epoch = self.n_iter/self.epoch_size

        # run in parallel on several GPUs
        cudnn.benchmark = True
        pose_exp_net = torch.nn.DataParallel(pose_exp_net)

        # optimizer
        print('=> setting adam solver')
        parameters = pose_exp_net.parameters()
        optimizer = torch.optim.Adam(parameters, self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)

        t_begin = time.time()
        for epoch in range(self.n_epoch, self.num_epochs, 1):
            self.n_epoch = epoch
            if self.n_epoch in self.decreasing_lr_epochs:
                self.lr *= 0.5
                print('learning rate decreases 1/2 at epoch {}'.format(self.n_epoch))

            # run training for one epoch
            train_loss, test_total_loss = self._train_one_epoch(pose_exp_net, optimizer)

            # print times and training loss after each epoch
            elapse_time = time.time() - t_begin
            epoch_time = elapse_time / (epoch + 1)
            batch_time = epoch_time / self.epoch_size
            eta = epoch_time * self.num_epochs - elapse_time
            print("\n\nTrain: elapsed time {:.2f}s, time per epoch {:.2f}s, time per batch {:.2f}s, eta {:.2f}s, "
                  "train loss {:.3f}, test loss {:.3f}\n\n".format(elapse_time, epoch_time, batch_time, eta,
                  train_loss, test_total_loss))

            # write summary
            test_loss = 0
            with open(self.loss_summary_path, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow([self.n_iter, train_loss, test_loss])































