import itertools
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable

from dataloaders.dataloader_builder import DataLoader
from metrics.APE_RPE import APE_RPE_metrics
from metrics.terr_rerr_metrics import kittiEvalOdom
from models.model_builder import Model
from tests.test_builder import Test
from trains.losses import loss_regression
from utils.auxiliary import AverageMeter
from utils.auxiliary import save_loss_to_resultstable, ensure_dir, \
    save_test_losses_to_tensorboard, calculate_rmse, save_mat_pose_to_file
from utils.inverse_warp import pose_vec2mat
from utils.visualization import visualization_trajectories

use_cuda = torch.cuda.is_available()



class test_pose(Test):
    '''
        Returns pose regression loss (requires pose GT).
    '''
    def __init__(self, FLAGS):
        super(test_pose, self).__init__(FLAGS)
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.beta = FLAGS.beta
        self.euler_angles = FLAGS.euler_angles
        self.rotation_mode = 'euler'
        self.n_dof = 6
        self.seq_length = FLAGS.seq_length
        if not self.euler_angles:
            self.rotation_mode = 'quat'
            self.n_dof = 7
        self.with_gt = FLAGS.with_gt
        self.model = FLAGS.model
        assert self.with_gt == 1, 'no pose GT in test dataset'

        if hasattr(FLAGS, 'seed'):
            self.seed = FLAGS.seed
        self.debug = FLAGS.debug
        if self.worker_num != None:
            self.debug = True
        self.results_table_path = os.path.join(self.train_dir, 'results.csv')
        if self.worker_num != None:
            self.results_table_path = os.path.join(self.train_dir, 'results_{}.csv'.format(self.worker_num))
        self.n_iter = 0
        self.visualization_paths_test_dir = os.path.join(self.train_dir, 'visualization_paths_test')
        ensure_dir(self.visualization_paths_test_dir)
        self.results_table_path = os.path.join(self.train_dir, 'results.csv')
        self.results_table_path_tmp = os.path.join(self.train_dir, 'results_tmp.csv')
        torch.manual_seed(self.seed)
        if use_cuda:
            torch.cuda.manual_seed(self.seed)

        # dataloader
        self.dataloader, self.num_samples = DataLoader.dataloader_builder(FLAGS.data_loader, FLAGS, 'test',
                                                                     parent_class=data.Dataset)


    def _test(self, pose_exp_net):
        #TODO: to update test, because now ref_imgs_l, filenames_ref and gt_ref_poses are lists (of length seq_lenth-1 for batch=1
        '''
            run test with or without GT (without GT metric is actually losses, the same as in training)
        '''
        one_iter_time = AverageMeter()                         # time to execute one iteration
        losses = AverageMeter(i=4, precision=4)
        filenames_tgt, filenames_ref = [], []
        pred_ref_poses_rel = np.zeros((self.num_samples, self.n_dof))
        pred_ref_poses_abs = np.zeros((self.num_samples, self.n_dof))            # [N, (tx, ty, tz, rx, ry, rz)]
        pred_ref_poses_abs_np = 0
        gt_ref_pose_np_prev = np.zeros(self.batch_size, self.n_dof)
        gt_ref_pose_np_prev_var = torch.zeros(self.n_dof)
        gt_ref_poses_abs = np.zeros((self.num_samples,  self.n_dof))
        gt_ref_poses_rel = np.zeros((self.num_samples, self.n_dof))

        # switch to evaluate mode
        pose_exp_net.eval()

        end = time.time()
        for i, (tgt_img_l, ref_img_l, _, _, gt_trg_pose, gt_ref_pose, filename_tgt, filename_ref) \
                in enumerate(self.dataloader):
            filename_tgt_cut = [(filename_tgt[i].split("/")[-1]).split(".")[0] for i in range(len(filename_tgt))]
            filename_ref_cut = [(filename_ref[i].split("/")[-1]).split(".")[0] for i in range(len(filename_ref))]
            filenames_tgt.append(filename_tgt_cut[0:len(filename_tgt_cut)])
            filenames_ref.append(filename_ref_cut[0:len(filename_ref_cut)])
            if use_cuda:
                tgt_img_l = tgt_img_l.cuda()
                ref_img_l = ref_img_l.cuda()
            tgt_img_l_var = Variable(tgt_img_l, volatile=True)
            ref_img_l_var = [Variable(ref_img_l, volatile=True)]

            # compute output
            _, pose = pose_exp_net(tgt_img_l_var, ref_img_l_var)

            # compute loss
            # gt_pose_delta = Variable(gt_ref_pose.view(-1, 6).type(torch.FloatTensor) -
            #                          gt_trg_pose.view(-1, 6).type(torch.FloatTensor))         #(tx, ty, tz, rx, ry, rz)
            gt_pose_delta = Variable(gt_ref_pose.view(-1, 6).type(torch.FloatTensor) - gt_ref_pose_np_prev_var)
            gt_ref_pose_np_prev_var = gt_ref_pose.view(-1, 6).type(torch.FloatTensor)
            if use_cuda:
                gt_pose_delta = gt_pose_delta.cuda()
            loss = loss_regression(pose.view(-1, 6), gt_pose_delta, beta=self.beta)
            loss_names = ['tot_loss']
            losses.update(loss.data[0], self.batch_size)

            # relative and absolute poses
            assert self.batch_size == 1
            pred_ref_poses_rel[i] = pose.cpu().data[0].numpy()                      # [1, 6]
            pred_ref_poses_abs_np += pose.cpu().data[0].numpy()                      # [1, 6]
            pred_ref_poses_abs[i] = pred_ref_poses_abs_np
            gt_ref_pose_abs_np = gt_ref_pose.cpu().numpy()
            gt_ref_poses_abs[i] = gt_ref_pose_abs_np
            gt_ref_poses_rel[i] = gt_ref_pose.cpu().numpy() - gt_ref_pose_np_prev
            abs_error_pos = np.sum(np.abs(pred_ref_poses_abs_np[:, :3] - gt_ref_pose_abs_np[:, :3]))/3
            print('i = {}, abs_error_pos = {:10.4f}, pred_ref_poses_abs_z = {:10.4f}, gt_ref_pose_abs_z = {:10.4f} '.
                  format(i, abs_error_pos, pred_ref_poses_abs_np[:, 2][0], gt_ref_pose_abs_np[:, 2][0]))
            gt_ref_pose_np_prev = gt_ref_pose.cpu().numpy()

            # measure elapsed time
            one_iter_time.update(time.time() - end)
            end = time.time()
            if self.debug and i%int(self.num_samples/self.batch_size/10) == 0:
                print('Test: finished processing batch {}'.format(i))

        filenames_tgt = list(itertools.chain(*filenames_tgt))
        filenames_ref = list(itertools.chain(*filenames_ref))
        pred_ref_poses_rel = np.concatenate((np.zeros((1, 6)), pred_ref_poses_rel), axis=0)
        pred_ref_poses_abs = np.concatenate((np.zeros((1, 6)), pred_ref_poses_abs), axis=0)
        gt_ref_poses_rel = np.concatenate((np.zeros((1, 6)), gt_ref_poses_rel), axis=0)
        gt_ref_poses_abs = np.concatenate((np.zeros((1, 6)), gt_ref_poses_abs), axis=0)

        return losses.avg, loss_names, pred_ref_poses_rel, pred_ref_poses_abs, gt_ref_poses_rel, gt_ref_poses_abs, filenames_tgt, filenames_ref, one_iter_time


    def build(self):
        self._check_args()

        # load model
        pose_exp_net = Model.model_builder('PoseExpNet', self.FLAGS)
        if self.debug:
            print("\nPoseExpNet = ", pose_exp_net)
        if use_cuda:
            pose_exp_net = pose_exp_net.cuda()

        # load weights
        if self.load_ckpt != '':
            pose_model_path = self.load_ckpt
        else:
            pose_model_path = os.path.join(self.ckpts_dir, 'exp_pose_ckpt.pth.tar')
        if os.path.isfile(pose_model_path):
            ckpt_dict = torch.load(pose_model_path)
            train_iter_poseexp = ckpt_dict['iteration']
            assert isinstance(ckpt_dict['state_dict'], (dict, OrderedDict)), type(ckpt_dict['state_dict'])
            pose_exp_net.load_state_dict(ckpt_dict['state_dict'], strict=False)
            print('PoseExpNet ckpt loaded from {} (iter {})'.format(pose_model_path, train_iter_poseexp))

            if train_iter_poseexp > 0:
                self.n_iter = train_iter_poseexp

            # run test
            t_begin = time.time()
            test_total_loss, loss_names, pred_ref_poses_rel, pred_ref_poses_abs, gt_ref_poses_rel, gt_ref_poses_abs, \
            filenames_tgt, filenames_ref, one_iter_time = self._test(pose_exp_net)

            is_best, _, test_total_loss_best, _ = save_test_losses_to_tensorboard(self.results_table_path,
                                                                                  self.results_table_path_tmp,
                                                                                  debug=self.debug, iter=self.n_iter)

            # KITTY odometry eval
            t_err, r_err = -1, -1
            if self.data_loader == 'KITTY_odom' and (is_best or self.debug):
                results_dir = os.path.join(self.train_dir, 'results_{}'.format(self.n_iter))
                ensure_dir(results_dir)
                test_sequence_path = os.path.join(self.data_dir, 'test.txt')
                sequence = [folder[:-1] for folder in open(test_sequence_path)][0]
                pred_poses_mat = pose_vec2mat(torch.from_numpy(pred_ref_poses_abs.reshape(-1, 6)).float(), detach=False).numpy().reshape(-1, 3, 4)  # [N, 3, 4]
                save_mat_pose_to_file(pred_poses_mat, results_dir, sequence)
                gt_dir = os.path.join('/'.join(self.data_dir.split('/')[:-1]), 'poses')
                # EvalOdom
                odom_eval = kittiEvalOdom(gt_dir)
                odom_eval.eval_seqs = [int(sequence)]
                t_err, r_err = odom_eval.eval(results_dir)
                # APE, RPE
                gt_file = os.path.join(gt_dir, '{}.txt'.format(sequence))
                est_file = os.path.join(results_dir, '{}.txt'.format(sequence))
                APE_RPE_metrics(gt_file, est_file, use_aligned_trajectories=False)


            # calculate rmse
            rmse_pos, rmse_ang, abs_error_pos, abs_error_ang = calculate_rmse(gt_ref_poses_abs, pred_ref_poses_abs,
                                                self.results_table_path,  self.n_iter)

            # save test losses and metrics to results table
            values = test_total_loss + [abs_error_pos] + [abs_error_ang] + [t_err] + [r_err]
            col_names = loss_names + ['abs_error_pos'] + ['abs_error_ang'] + ['t_err'] + ['r_err']
            save_loss_to_resultstable(values, col_names, self.results_table_path, self.n_iter)

            if is_best or self.debug:
                # save predicted poses in format (tx, ty, tz, rx, ry, rz)
                # save_pose_to_file(filenames_tgt, filenames_ref, pred_ref_poses_rel, pred_ref_poses_abs, gt_ref_poses_rel, gt_ref_poses_abs,
                #                   self.train_dir, self.n_iter)
                # visualize predicted anf GT paths
                visualization_trajectories(pred_ref_poses_abs, gt_ref_poses_abs, self.visualization_paths_test_dir, self.n_iter)

            if self.debug:
                print('Test: iter {} is best: {}, current loss: {:.4f}, best loss: {:.4f}'.
                      format(self.n_iter, is_best, test_total_loss[0], test_total_loss_best))
                print('rmse_pos = {:10.4f}, rmse_ang = {:10.4f}, abs_error_pos = {:10.4f}, abs_error_ang = {:10.4f}'.
                      format(rmse_pos, rmse_ang, abs_error_pos, abs_error_ang))
            print("\nTest: iter {}, time per test {:.2f}s, time per one batch (of {}) {:.2f}s, total avg loss {:.4f}, rmse_pos {:.6f} m, abs_error_pos {:.6f} rad\n".
                format(self.n_iter, time.time() - t_begin, self.batch_size, one_iter_time.avg[0], test_total_loss[0],
                       rmse_pos, abs_error_pos))
        else:
            print('no ckpt found for running test')



















































# import time
# import os
# from collections import OrderedDict
# import numpy as np
# import itertools
#
# from dataloaders.dataloader_builder import DataLoader
# from models.model_builder import Model
# from tests.test_builder import Test
# from utils.auxiliary import AverageMeter
# from trains.losses import loss_regression
# from utils.auxiliary import tensor2array, save_pose_to_file, save_loss_to_resultstable, ensure_dir, \
#     save_test_losses_to_tensorboard, calculate_rmse
# from visualizations.visualization import save_concat_img_results, visualization_trajectories
#
# import torch
# from torch.autograd import Variable
# import torch.utils.data as data
# import torch.backends.cudnn as cudnn
# use_cuda = torch.cuda.is_available()
#
#
#
# class test_pose(Test):
#     '''
#         Returns pose regression loss (requires pose GT).
#     '''
#     def __init__(self, FLAGS):
#         super(test_pose, self).__init__(FLAGS)
#         self.height = FLAGS.height
#         self.width = FLAGS.width
#         self.beta = FLAGS.beta
#         self.euler_angles = FLAGS.euler_angles
#         self.rotation_mode = 'euler'
#         self.n_dof = 6
#         self.seq_length = FLAGS.seq_length
#         if not self.euler_angles:
#             self.rotation_mode = 'quat'
#             self.n_dof = 7
#         self.with_gt = FLAGS.with_gt
#         self.model = FLAGS.model
#         assert self.with_gt == 1, 'no pose GT in test dataset'
#
#         if hasattr(FLAGS, 'seed'):
#             self.seed = FLAGS.seed
#         self.debug = FLAGS.debug
#         if self.worker_num != None:
#             self.debug = True
#         self.results_table_path = os.path.join(self.train_dir, 'results.csv')
#         if self.worker_num != None:
#             self.results_table_path = os.path.join(self.train_dir, 'results_{}.csv'.format(self.worker_num))
#         self.n_iter = 0
#         self.visualization_paths_test_dir = os.path.join(self.train_dir, 'visualization_paths_test')
#         ensure_dir(self.visualization_paths_test_dir)
#         self.results_table_path = os.path.join(self.train_dir, 'results.csv')
#         self.results_table_path_tmp = os.path.join(self.train_dir, 'results_tmp.csv')
#         torch.manual_seed(self.seed)
#         if use_cuda:
#             torch.cuda.manual_seed(self.seed)
#
#         # dataloader
#         self.dataloader, self.num_samples = DataLoader.dataloader_builder(FLAGS.data_loader, FLAGS, 'test',
#                                                                      parent_class=data.Dataset)
#
#
#     def _test(self, pose_exp_net):
#         '''
#             run test without GT (metric is actually losses, the same as in training)
#         '''
#         one_iter_time = AverageMeter()                         # time to execute one iteration
#         losses = AverageMeter(precision=4)
#         pred_poses_delta, gt_poses_delta, gt_poses_tgt, filenames_tgt, filenames_ref = [], [], [], [], []
#         if self.seq_length > 1:
#             pred_poses_delta = np.zeros((self.num_samples, self.n_dof))    # [N, (tx, ty, tz, rx, ry, rz)]
#             if self.with_gt:
#                 gt_poses_delta = np.zeros((self.num_samples, self.n_dof))
#                 gt_poses_tgt = np.zeros((self.num_samples, self.n_dof))
#
#         # switch to evaluate mode
#         pose_exp_net.eval()
#
#         end = time.time()
#         for i, (tgt_img_l, ref_img_l, _, _, gt_trg_pose, gt_ref_pose, filename_tgt, filename_ref) \
#                 in enumerate(self.dataloader):
#             filename_tgt_cut = [(filename_tgt[i].split("/")[-1]).split(".")[0] for i in range(len(filename_tgt))]
#             filename_ref_cut = [(filename_ref[i].split("/")[-1]).split(".")[0] for i in range(len(filename_ref))]
#             filenames_tgt.append(filename_tgt_cut[0:len(filename_tgt_cut)])
#             filenames_ref.append(filename_ref_cut[0:len(filename_ref_cut)])
#             if use_cuda:
#                 tgt_img_l = tgt_img_l.cuda()
#                 ref_img_l = ref_img_l.cuda()
#             tgt_img_l_var = Variable(tgt_img_l, volatile=True)
#             ref_img_l_var = [Variable(ref_img_l, volatile=True)]
#
#             # compute output
#             _, pred_pose_delta = pose_exp_net(tgt_img_l_var, ref_img_l_var)         # (tx, ty, tz, rx, ry, rz)
#
#             # compute loss
#             gt_pose_delta = Variable(gt_ref_pose.view(-1, 6).type(torch.FloatTensor) - gt_trg_pose.view(-1, 6).type(torch.FloatTensor))                             #(tx, ty, tz, rx, ry, rz)
#             if use_cuda:
#                 gt_pose_delta = gt_pose_delta.cuda()
#             loss = loss_regression(pred_pose_delta.view(-1, 6), gt_pose_delta, beta=self.beta)
#             loss_names = ['tot_loss']
#             losses.update(loss.data[0], self.batch_size)
#
#             # save predicted pose values to table
#             if self.seq_length > 1:
#                 step = self.batch_size * (self.seq_length - 1)
#                 pred_poses_delta[i * step: (i + 1) * step] = pred_pose_delta.data.cpu().view(-1, 6).numpy()     # [N, (tx, ty, tz, rx, ry, rz)]
#                 if self.with_gt:
#                     # [N, (tx, ty, tz, rx, ry, rz)]
#                     gt_poses_delta[i * step: (i + 1) * step] = gt_ref_pose.view(-1, 6).numpy() - gt_trg_pose.view(-1, 6).numpy()
#                     gt_poses_tgt[i * step: (i + 1) * step] = gt_trg_pose.view(-1, 6).numpy()
#
#             # measure elapsed time
#             one_iter_time.update(time.time() - end)
#             end = time.time()
#             if self.debug and i%int(self.num_samples/self.batch_size/10) == 0:
#                 print('Test: finished processing batch {}'.format(i))
#
#         filenames_tgt = list(itertools.chain(*filenames_tgt))
#         filenames_ref = list(itertools.chain(*filenames_ref))
#         pred_poses_tgt = np.cumsum(pred_poses_delta, axis=0)
#         pred_poses_tgt[:, 3] = np.unwrap(pred_poses_tgt[:, 3])
#         pred_poses_tgt[:, 4] = np.unwrap(pred_poses_tgt[:, 4])
#         pred_poses_tgt[:, 5] = np.unwrap(pred_poses_tgt[:, 5])
#         gt_poses_tgt[:, 3] = np.unwrap(gt_poses_tgt[:, 3])
#         gt_poses_tgt[:, 4] = np.unwrap(gt_poses_tgt[:, 4])
#         gt_poses_tgt[:, 5] = np.unwrap(gt_poses_tgt[:, 5])
#         pred_poses_tgt = pred_poses_tgt[:, :] - pred_poses_tgt[0, :]
#         gt_poses_tgt = gt_poses_tgt[:, :] - gt_poses_tgt[0, :]
#         return losses.avg, loss_names, pred_poses_delta, pred_poses_tgt, gt_poses_delta, gt_poses_tgt, filenames_tgt, \
#                filenames_ref, one_iter_time
#
#
#     def build(self):
#         self._check_args()
#
#         # load model
#         pose_exp_net = Model.model_builder(self.model, self.FLAGS)
#         # if self.debug:
#         #     print('\nModel "{}": \n{}'.format(self.model, pose_exp_net))
#         if use_cuda:
#             pose_exp_net = pose_exp_net.cuda()
#
#         # load weights
#         if self.load_ckpt != '':
#             model_path = self.load_ckpt
#         else:
#             model_path = os.path.join(self.ckpts_dir, 'exp_pose_ckpt.pth.tar')
#         if os.path.isfile(model_path):
#             ckpt_dict = torch.load(model_path)
#             train_iter_poseexp = ckpt_dict['iteration']
#             assert isinstance(ckpt_dict['state_dict'], (dict, OrderedDict)), type(ckpt_dict['state_dict'])
#             pose_exp_net.load_state_dict(ckpt_dict['state_dict'], strict=False)
#             print('Ckpt loaded from {} (iter {})'.format(model_path, train_iter_poseexp))
#             if train_iter_poseexp > 0:
#                 self.n_iter = train_iter_poseexp
#
#             # run in parallel on several GPUs
#             cudnn.benchmark = True
#             pose_exp_net = torch.nn.DataParallel(pose_exp_net)
#
#             # run test
#             t_begin = time.time()
#             test_total_loss, loss_names, pred_poses_delta, pred_poses_tgt, gt_poses_delta, gt_poses_tgt, filenames_tgt, \
#             filenames_ref, one_iter_time = self._test(pose_exp_net)
#
#             # save test losses to results table
#             save_loss_to_resultstable(test_total_loss, loss_names, self.results_table_path, self.n_iter)
#
#             is_best, _, test_total_loss_best, _ = save_test_losses_to_tensorboard(self.results_table_path,
#                                                                                      self.results_table_path_tmp,
#                                                                                      debug=self.debug, iter=self.n_iter)
#
#             # calculate rmse
#             rmse_loc, rmse_ang = -1, -1
#             if self.with_gt:
#                 rmse_loc, rmse_ang = calculate_rmse(gt_poses_delta, pred_poses_delta, self.with_gt,
#                                                     self.results_table_path,
#                                                     self.n_iter)
#                 if self.debug:
#                     print(
#                         '\nTest: gt_pose_loc avg {:.6f}, pred_pose_loc avg {:.6f}, gt_pose_ang avg {:.6f}, pred_pose_loc avg {:.6f}'.
#                         format(gt_poses_delta[:, :3].mean(), pred_poses_delta[:, :3].mean(),
#                                gt_poses_delta[:, 3:].mean(),
#                                pred_poses_delta[:, 3:].mean()))
#                     print(
#                         'Test: gt_pose_loc std {:.6f}, pred_pose_loc std {:.6f}, gt_pose_ang std {:.6f}, pred_pose_loc std {:.6f}'.
#                         format(np.std(gt_poses_delta[:, :3]), np.std(pred_poses_delta[:, :3]),
#                                np.std(gt_poses_delta[:, 3:]),
#                                np.std(pred_poses_delta[:, 3:])))
#
#             if self.debug:
#                 print('Test: iter {} is best: {}, current loss: {:.4f}, best loss: {:.4f}'.format(self.n_iter, is_best,
#                                                                                                   test_total_loss[0],
#                                                                                                   test_total_loss_best))
#
#             # save predicted poses in format (tx, ty, tz, rx, ry, rz)
#             if is_best:
#                 save_pose_to_file(filenames_tgt, filenames_ref, pred_poses_delta, pred_poses_tgt, gt_poses_delta,
#                                   gt_poses_tgt,
#                                   self.train_dir, self.n_iter)
#
#             # # visualize predicted anf GT paths
#             visualization_trajectories(pred_poses_tgt, gt_poses_tgt, self.visualization_paths_test_dir, self.n_iter)
#
#             print(
#                 "\nTest: iter {}, time per test {:.2f}s, time per one batch (of {}) {:.2f}s, total avg loss {:.4f}, rsme_loc {:.6f} m, rsme_ang {:.6f} rad\n".
#                 format(self.n_iter, time.time() - t_begin, self.batch_size, one_iter_time.avg[0], test_total_loss[0],
#                        rmse_loc, rmse_ang))
#         else:
#             print('no ckpt found for running test')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#



