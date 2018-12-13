import time
import os
from tensorboardX import SummaryWriter                                 # pip install tensorboardX
from itertools import chain

from dataloaders.dataloader_builder import DataLoader
from trains.train_builder import Train
from trains.losses import compute_loss
from utils.auxiliary import AverageMeter, save_checkpoint, save_test_losses_to_tensorboard, load_model_and_weights, \
    save_train_losses_and_imgs_to_tensorboard_and_csv, convert_to_tensors, make_loss_dict, write_summary_to_csv

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
use_cuda = torch.cuda.is_available()



class train_unsup(Train):
    '''
        Unsupervised training of pose and depth with photometric loss for a
        stereo sequence of video in train and monosequence in test).
        It can be either divided into snippets of n frames (as in
        Zhou et al. "Unsupervised Learning of Depth and Ego-Motion from Video"),
        which are then merged into a global pose sequence, or trained and tested as
        concatenated target and reference images, then the estimated frame-to-frame
        camera poses are simply integrated over the entire sequence (Zhan el al.
        "Unsupervised Learning of Monocular Depth Estimation and Visual Odometry
        with Deep Feature Reconstruction")
    '''
    def __init__(self, FLAGS):
        super(train_unsup, self).__init__(FLAGS)
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.stereo = FLAGS.stereo
        self.with_gt_pose, self.with_gt_depth = False, False
        if hasattr(FLAGS, 'with_gt'):
            self.with_gt_pose = FLAGS.with_gt
        if hasattr(FLAGS, 'with_gt_pose'):
            self.with_gt_pose = FLAGS.with_gt_pose
        if hasattr(FLAGS, 'with_gt_pose'):
            self.with_gt_depth = FLAGS.with_gt_depth
        self.max_depth = 80
        if hasattr(FLAGS, 'max_depth'):
            self.max_depth = FLAGS.max_depth
        assert FLAGS.seq_length > 1
        if len(FLAGS.decreasing_lr_epochs) > 0:
            self.decreasing_lr_epochs = list(map(int, FLAGS.decreasing_lr_epochs.split(',')))
        else:
            self.decreasing_lr_epochs = []
        self.weight_decay = FLAGS.weight_decay
        self.euler_angles = FLAGS.euler_angles
        self.rotation_mode = 'euler'
        if not self.euler_angles:
            self.rotation_mode = 'quat'
        self.num_iters_for_print = FLAGS.num_iters_for_print
        if hasattr(FLAGS, 'seed'):
            self.seed = FLAGS.seed
        self.debug = FLAGS.debug
        self.load_ckpt_disp = ''
        self.rm_train_dir = True
        if self.load_ckpt != '' or (hasattr(FLAGS, 'load_ckpt_disp') and FLAGS.load_ckpt_disp != ''):
            self.load_ckpt_disp = FLAGS.load_ckpt_disp
            self.rm_train_dir = False
        if self.worker_num != None:
            self.rm_train_dir = False
            self.debug = True
        if not self.debug:
            save_path = os.path.join('tensorboard', self.train_dir.split('/')[-1])
            self.writer = SummaryWriter(save_path)
            line = ''
            args_dict = {arg: getattr(FLAGS, arg) for arg in vars(FLAGS)}
            for key, value in sorted(args_dict.items()):
                line += '{}={}, '.format(key, value)
                self.writer.add_text('Text', line, 0)
        else:
            self.writer = None
        self.test_iters_dict = {}
        self.train_iters_dict = {}
        self.results_table_path = os.path.join(self.train_dir, 'results.csv')
        self.loss_summary_path = os.path.join(self.train_dir, 'loss_summary.csv')
        self.loss_full_path = os.path.join(self.train_dir, 'loss_full.csv')
        if self.worker_num != None:
            self.results_table_path = os.path.join(self.train_dir, 'results_{}.csv'.format(self.worker_num))
        self.n_iter = 0
        self.n_epoch = 0
        torch.manual_seed(self.seed)
        if use_cuda:
            torch.cuda.manual_seed(self.seed)
        self.loss_weights_dict, self.loss_dict = make_loss_dict(FLAGS.loss_weights)

        self.concat_LR, self.disp_norm, self.upscaling, self.edge_aware = False, False, False, False
        if hasattr(FLAGS, 'concat_LR') and FLAGS.concat_LR:
            self.concat_LR = True
        if hasattr(FLAGS, 'disp_norm') and FLAGS.disp_norm:
            self.disp_norm = True
        if hasattr(FLAGS, 'upscaling') and FLAGS.upscaling:
            self.upscaling = True
        if hasattr(FLAGS, 'edge_aware') and FLAGS.edge_aware:
            self.edge_aware = True
        self.loss_params_dict = {'stereo': self.stereo, 'with_gt_pose': self.with_gt_pose,
                                 'with_gt_depth': self.with_gt_depth, 'rotation_mode': self.rotation_mode,
                                 'disp_norm': self.disp_norm, 'upscaling': self.upscaling,
                                 'edge_aware': self.edge_aware, 'concat_LR': self.concat_LR,
                                 'max_depth': self.max_depth, 'mode': 'train'}

        # dataloader
        self.dataloader, self.num_samples = DataLoader.dataloader_builder(FLAGS.data_loader, FLAGS, 'train',
                                                                     parent_class=data.Dataset)
        self.epoch_size = self.num_samples//self.batch_size
        print('Number of iterations per epoch: ', self.epoch_size)


    def _train_one_epoch(self, disp_net, pose_exp_net, optimizer):
        one_iter_time = AverageMeter()                         # time to execute one iteration
        losses = AverageMeter(precision=4)
        filenames_tgt, filenames_ref = [], []

        # switch to train mode
        disp_net.train()
        pose_exp_net.train()

        end = time.time()
        for i, var_dict_np in enumerate(self.dataloader):
            # convert numpy input to pytorch tensors
            var_dict_t, filenames_tgt, filenames_ref = convert_to_tensors(var_dict_np, filenames_tgt, filenames_ref,
                                                        self.batch_size, self.loss_params_dict, use_cuda, test=False)

            # compute output
            disp_input = var_dict_t['tgt_img_l']
            if self.stereo and self.concat_LR:
                disp_input = torch.cat((var_dict_t['tgt_img_l'], var_dict_t['tgt_img_r']), 1)
            disp = disp_net(disp_input)
            disp_l = [d[:, :1, :, :] for d in disp]
            b, f = 1, 1
            # if self.stereo:
            #     b = var_dict_t['baseline']
            #     f = var_dict_t['intrinsics_l']  #intrinsics.view(-1)[0]
            depth_l = [b*f / d for d in disp_l]
            disp_r = None
            if self.stereo and self.concat_LR:
                disp_r = [d[:, 1:, :, :] for d in disp]
            elif self.stereo and (('w_RL' in self.loss_weights_dict and self.loss_weights_dict['w_RL'] > 0) or
                                        ('w_DC' in self.loss_weights_dict and self.loss_weights_dict['w_DC'] > 0)):
                disp = disp_net(var_dict_t['tgt_img_r'])
                disp_r = [d[:, :1, :, :] for d in disp]

            explainability_mask, pose = pose_exp_net(var_dict_t['tgt_img_l'], var_dict_t['ref_imgs_l'])     # pose [B, (tx, ty, tz, rx, ry, rz)]

            # compute loss
            losses_list, loss_names = compute_loss(var_dict_t, disp_l, depth_l, disp_r, explainability_mask, pose,
                                                   self.loss_weights_dict, self.loss_dict, self.loss_params_dict)
            loss = losses_list[0]
            losses.update(loss.data[0], self.batch_size)

            # save train losses to tensorboard and csv
            if i > 0 and self.n_iter % self.num_iters_for_print == 0:
                self.train_iters_dict = save_train_losses_and_imgs_to_tensorboard_and_csv(var_dict_t, self.writer,
                        losses_list, loss_names, disp_l, depth_l, pose, explainability_mask, self.n_iter,
                        self.num_iters_for_print, self.loss_full_path, self.train_iters_dict, self.n_epoch,
                        self.rotation_mode, self.debug)

            if i > 0 and self.n_iter % self.num_iters_for_ckpt == 0:
                # save test losses to tensorboard and results_table.csv
                self.test_iters_dict = save_test_losses_to_tensorboard(self.test_iters_dict, self.results_table_path,
                                                                       self.writer, self.debug)
                # save checkpoint
                state_names = ['dispnet', 'exp_pose']
                states = [{'iteration': self.n_iter, 'epoch': self.n_epoch, 'state_dict': disp_net.module.state_dict()},
                          {'iteration': self.n_iter, 'epoch': self.n_epoch, 'state_dict':
                              pose_exp_net.module.state_dict()}]
                save_checkpoint(self.ckpts_dir, states, state_names)

            # compute gradient and do Adam step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            one_iter_time.update(time.time() - end)
            end = time.time()
            if self.n_iter % self.num_iters_for_print == 0:
                print('\n\nTrain:  epoch {} (iter {}),  time for iter {},  total loss {}\n'.format(self.n_epoch,
                            self.n_iter, one_iter_time, losses))
            self.n_iter += 1
            if i >= self.epoch_size - 1:
                break

        return losses.avg[0]


    def build(self):
        self._check_args(self.rm_train_dir)

        # initialize or resume training
        _, models, _, self.n_iter, self.n_epoch = load_model_and_weights(self.load_ckpt, self.load_ckpt_disp,
                                                                         self.FLAGS, use_cuda)
        disp_net, pose_exp_net = models

        # run in parallel on several GPUs
        cudnn.benchmark = True
        disp_net = torch.nn.DataParallel(disp_net)
        pose_exp_net = torch.nn.DataParallel(pose_exp_net)

        # optimizer
        print('=> setting adam solver')
        parameters = chain(disp_net.parameters(), pose_exp_net.parameters())
        optimizer = torch.optim.Adam(parameters, self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)

        # run training for n epochs
        t_begin = time.time()
        for epoch in range(self.n_epoch, self.num_epochs, 1):
            self.n_epoch = epoch
            # TODO: to include decreasing lr schedule (in resume mode as well)
            if self.n_epoch in self.decreasing_lr_epochs:
                self.lr *= 0.5
                print('learning rate decreases 1/2 at epoch {}'.format(self.n_epoch))

            # run training for one epoch
            train_loss = self._train_one_epoch(disp_net, pose_exp_net, optimizer)

            # write train and test losses to 'loss_summary.csv
            test_loss = write_summary_to_csv(self.loss_summary_path, self.results_table_path,
                                 self.n_iter, self.n_epoch, train_loss)

            # print times and training loss after each epoch
            elapse_time = (time.time() - t_begin)/60
            epoch_time = elapse_time / (epoch + 1) / 60
            batch_time = epoch_time / self.epoch_size
            eta = epoch_time * self.num_epochs - elapse_time
            print("\n\nTrain: elapsed time {:.2f} min, time per epoch {:.2f} min, time per batch {:.2f} s, eta {:.2f} min, "
                  "train loss {:.3f}, test loss {:.3f}\n\n".format(elapse_time, epoch_time, batch_time, eta, train_loss,
                                                                   test_loss))

































