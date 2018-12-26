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
        self.with_gt_depth = True
        if hasattr(FLAGS, 'model'):
            self.model_names = list(FLAGS.model.split(','))
        else:
            self.model_names = ['DispNetS']
        self.max_depth = 80
        if hasattr(FLAGS, 'max_depth'):
            self.max_depth = FLAGS.max_depth
        self.decreasing_lr_epochs = []
        if len(FLAGS.decreasing_lr_epochs) > 0:
            self.decreasing_lr_epochs = list(map(int, FLAGS.decreasing_lr_epochs.split(',')))
        self.weight_decay = FLAGS.weight_decay
        self.num_iters_for_print = FLAGS.num_iters_for_print
        if hasattr(FLAGS, 'seed'):
            self.seed = FLAGS.seed
        self.debug = FLAGS.debug
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
        self.test_iters_dict, self.train_iters_dict = {}, {}
        self.n_iter, self.n_epoch = 0, 0
        self.results_table_path = os.path.join(self.train_dir, 'results.csv')
        self.loss_summary_path = os.path.join(self.train_dir, 'loss_summary.csv')
        self.loss_full_path = os.path.join(self.train_dir, 'loss_full.csv')
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
        self.loss_params_dict = {'stereo': self.stereo, 'with_gt_depth': self.with_gt_depth,
                                 'disp_norm': self.disp_norm, 'upscaling': self.upscaling,
                                 'edge_aware': self.edge_aware, 'concat_LR': self.concat_LR,
                                 'max_depth': self.max_depth, 'mode': 'train'}

        # dataloader
        self.dataloader, self.num_samples = DataLoader.dataloader_builder(FLAGS.data_loader, FLAGS, 'train',
                                                                     parent_class=data.Dataset)
        self.epoch_size = self.num_samples//self.batch_size
        print('Number of iterations per epoch: ', self.epoch_size)


    def _train_one_epoch(self, models, optimizer):
        one_iter_time = AverageMeter()                         # time to execute one iteration
        losses = AverageMeter(precision=4)
        filenames_tgt = []

        # switch to train mode
        disp_net = models[0]
        disp_net.train()

        end = time.time()
        for i, var_dict_np in enumerate(self.dataloader):
            # convert numpy input to pytorch tensors
            var_dict_t, filenames_tgt = convert_to_tensors(var_dict_np, filenames_tgt, self.loss_params_dict, use_cuda)
            # compute output
            disp_input = var_dict_t['tgt_img_l']
            if self.stereo and self.concat_LR:
                disp_input = torch.cat((var_dict_t['tgt_img_l'], var_dict_t['tgt_img_r']), 1)
            disp = disp_net(disp_input)
            disp_l = [d[:, :1, :, :] for d in disp]
            depth_l = [1 / d for d in disp_l]
            disp_r, depth_r = None, None
            if self.stereo and self.concat_LR:
                disp_r = [d[:, 1:, :, :] for d in disp]
                depth_r = [1 / d for d in disp_r]
            elif self.stereo and (('w_RL' in self.loss_weights_dict and self.loss_weights_dict['w_RL'] > 0) or
                                        ('w_DC' in self.loss_weights_dict and self.loss_weights_dict['w_DC'] > 0)):
                disp = disp_net(var_dict_t['tgt_img_r'])
                disp_r = [d[:, :1, :, :] for d in disp]
                depth_r = [1 / d for d in disp_r]

            # compute loss
            losses_list, loss_names = compute_loss(var_dict_t, disp_l, depth_l, disp_r, depth_r, self.loss_weights_dict,
                                                   self.loss_dict, self.loss_params_dict)
            loss = losses_list[0]
            losses.update(loss.data.item(), self.batch_size)

            # save train losses to tensorboard and csv
            if i > 0 and self.n_iter % self.num_iters_for_print == 0:
                self.train_iters_dict = save_train_losses_and_imgs_to_tensorboard_and_csv(var_dict_t, self.writer,
                        losses_list, loss_names, disp_l, depth_l, self.n_iter,
                        self.num_iters_for_print, self.loss_full_path, self.train_iters_dict, self.n_epoch, self.debug)

                # save test losses to tensorboard and results_table.csv
                self.test_iters_dict = save_test_losses_to_tensorboard(self.test_iters_dict, self.results_table_path,
                                                                       self.writer, self.debug)
                # save checkpoint
                states = [{'iteration': self.n_iter, 'epoch': self.n_epoch, 'state_dict': disp_net.module.state_dict()}]
                save_checkpoint(self.ckpts_dir, states, self.model_names)

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
        self._check_args()

        # initialize or resume training
        _, models, _, self.n_iter, self.n_epoch = load_model_and_weights(self.model_names, self.load_ckpt, self.FLAGS,
                                                                         self.ckpts_dir, use_cuda)
        # run in parallel on several GPUs
        cudnn.benchmark = True
        models = [torch.nn.DataParallel(models[i]) for i in range(len(models))]

        # optimizer
        print('=> setting adam solver')
        if len(models) == 1:
            parameters = chain(models[0].parameters())
        optimizer = torch.optim.Adam(parameters, self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)

        # run training for n epochs
        t_begin = time.time()
        for epoch in range(self.n_epoch, self.num_epochs, 1):
            self.n_epoch = epoch
            if len(self.decreasing_lr_epochs) > 0 and (self.n_epoch in self.decreasing_lr_epochs):
                idx = self.decreasing_lr_epochs.index(self.n_epoch) + 1
                self.lr /= 2 ** idx
                print('learning rate decreases by {} at epoch {}'.format(2 ** idx, self.n_epoch))

            # run training for one epoch
            train_loss = self._train_one_epoch(models, optimizer)

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

































