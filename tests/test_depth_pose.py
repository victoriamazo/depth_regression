import os
import shutil
import time

import numpy as np
import torch
import torch.utils.data as data

from dataloaders.dataloader_builder import DataLoader
from metrics.ATE_RE_metrics import compute_ATE_RE
from metrics.run_VO_metrics import run_VO_metrics
from metrics.depth_metrics import compute_depth_metrics_i, compute_depth_metrics
from tests.test_builder import Test
from trains.losses import compute_loss
from utils.auxiliary import AverageMeter, save_loss_to_resultstable, ensure_dir, check_if_best_model_and_save, \
    save_concat_img_results, load_model_and_weights, convert_to_tensors, make_loss_dict, flip_and_concat_imgs

use_cuda = torch.cuda.is_available()


class test_depth_pose(Test):
    '''
        Returns unsupervised photometric loss (and smooth and explanatory auxiliary losses)
        and, if pose GT is given, returns pose RMSE metric.
    '''
    def __init__(self, FLAGS):
        super(test_depth_pose, self).__init__(FLAGS)
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.batch_size = FLAGS.batch_size
        assert self.batch_size == 1, 'batch size is not 1 in test'
        self.euler_angles = FLAGS.euler_angles
        self.rotation_mode = 'euler'
        self.n_dof = 6
        self.seq_length = FLAGS.seq_length
        if not self.euler_angles:
            self.rotation_mode = 'quat'
            self.n_dof = 7
        self.with_gt_pose, self.with_gt_depth = False, False
        if hasattr(FLAGS, 'with_gt'):
            self.with_gt_pose = FLAGS.with_gt
        if hasattr(FLAGS, 'with_gt_pose'):
            self.with_gt_pose = FLAGS.with_gt_pose
        if hasattr(FLAGS, 'with_gt_pose'):
            self.with_gt_depth = FLAGS.with_gt_depth
        self.stereo_test = False
        if hasattr(FLAGS, 'stereo_test') and FLAGS.stereo_test:
            self.stereo_test = True
        self.stereo = FLAGS.stereo
        self.best_criteria_pose = "APE"
        if hasattr(FLAGS, 'best_criteria_pose'):
            self.best_criteria_pose = FLAGS.best_criteria_pose
        self.best_criteria_depth = "RSME"
        if hasattr(FLAGS, 'best_criteria_pose'):
            self.best_criteria_depth = FLAGS.best_criteria_depth
        self.max_depth = 80
        if hasattr(FLAGS, 'max_depth'):
            self.max_depth = FLAGS.max_depth
        if hasattr(FLAGS, 'seed'):
            self.seed = FLAGS.seed
        self.debug = FLAGS.debug
        self.load_ckpt_disp = ''
        if hasattr(FLAGS, 'load_ckpt_disp') and FLAGS.load_ckpt_disp != '':
            self.load_ckpt_disp = FLAGS.load_ckpt_disp
        if self.worker_num != None:
            self.debug = True
        self.results_table_path = os.path.join(self.train_dir, 'results.csv')
        if self.worker_num != None:
            self.results_table_path = os.path.join(self.train_dir, 'results_{}.csv'.format(self.worker_num))
        self.n_iter, self.epoch = 0, 0
        self.visualization_test_dir = os.path.join(self.train_dir, 'visualization_test')
        ensure_dir(self.visualization_test_dir)
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
        self.loss_params_dict = {'stereo': self.stereo_test, 'with_gt_pose': self.with_gt_pose,
                                 'with_gt_depth': self.with_gt_depth, 'rotation_mode': self.rotation_mode,
                                 'disp_norm': self.disp_norm, 'upscaling': self.upscaling,
                                 'edge_aware': self.edge_aware, 'concat_LR': self.concat_LR,
                                 'max_depth': self.max_depth, 'mode': 'test'}
        # dataloader
        self.dataloader, self.num_samples = DataLoader.dataloader_builder(FLAGS.data_loader, FLAGS, 'test',
                                                                     parent_class=data.Dataset)

    def _test(self, models):
        ''' Run test with or without GT (without GT metric is actually losses, the same as in training)
            (ref_imgs_l, filenames_ref and gt_ref_poses are lists (of length seq_lenth-1 for batch=1)).
            If seq_lenth<=2, predicted poses are an array of reference frame poses plus zero poses at the beginning,
            if seq_length=3,5,..., predicted poses are an array of stacked 'seq_length'-frame snippets.
        '''
        one_iter_time = AverageMeter()                         # time to execute one iteration
        losses = AverageMeter(i=len(self.loss_dict) + 1, precision=4)
        filenames_tgt, filenames_ref, ps_arr = [], [], []
        ATE_RE_errors = np.zeros((self.num_samples, 2), np.float32)
        depth_metrics = np.zeros((7, self.num_samples), np.float32)

        # switch to evaluate mode
        disp_net, pose_exp_net = models
        disp_net.eval()
        pose_exp_net.eval()

        end = time.time()
        for i, var_dict_np in enumerate(self.dataloader):
            # convert numpy input to pytorch tensors
            var_dict_t, filenames_tgt, filenames_ref = convert_to_tensors(var_dict_np, filenames_tgt, filenames_ref,
                                                        self.batch_size, self.loss_params_dict, use_cuda, test=True)
            # compute output
            disp_input = var_dict_t['tgt_img_l']
            if self.stereo and self.concat_LR:
                if self.stereo_test:
                    disp_input = torch.cat((var_dict_t['tgt_img_l'], var_dict_t['tgt_img_r']), 1)
                else:
                    disp_input = flip_and_concat_imgs(var_dict_t['tgt_img_l_cpu'], var_dict_t['tgt_img_l'])
            disp = disp_net(disp_input)
            disp_l = disp[:, :1, :, :]
            depth_l = 1 / disp_l
            disp_r = None
            if self.stereo_test and self.stereo and self.concat_LR:
                disp_r = disp[:, 1:, :, :]
            elif self.stereo_test and (('w_RL' in self.loss_weights_dict and self.loss_weights_dict['w_RL'] > 0) or
                                    ('w_DC' in self.loss_weights_dict and self.loss_weights_dict['w_DC'] > 0)):
                disp_r = disp_net(var_dict_t['tgt_img_r'])

            explainability_mask, pose = pose_exp_net(var_dict_t['tgt_img_l'], var_dict_t['ref_imgs_l'])     # pose [B, (tx, ty, tz, rx, ry, rz)]

            # compute loss
            losses_list, loss_names = compute_loss(var_dict_t, disp_l, depth_l, disp_r, explainability_mask, pose,
                                                   self.loss_weights_dict, self.loss_dict, self.loss_params_dict)
            losses_list_cpu = [losses.data[0] for losses in losses_list]
            if len(losses) != len(losses_list_cpu):
                losses.reset(len(losses_list_cpu))
            losses.update(losses_list_cpu)

            # make array of predicted poses
            poses_cpu = pose.cpu().data[0]  # [num_ref_imgs, 6]
            # for 5 - frames snippet, image '2' is the tgt image, images '0', '1', '3', '4' are ref images
            if self.seq_length == 2:
                poses_cpu = torch.cat([torch.zeros(1, 6).float(), poses_cpu])
                # gt_poses = torch.cat([var_dict_t['gt_trg_pose'], var_dict_t['gt_ref_poses'][0]])[:, :3]
            elif self.seq_length > 2:
                poses_cpu = torch.cat([poses_cpu[:self.seq_length // 2], torch.zeros(1, 6).float(),
                                       poses_cpu[self.seq_length // 2:]])  # [seq_length, 6]
            if i % (self.seq_length - 1) == 0:
                # for 4 ref images we take predicted poses every 4 steps
                pred_poses = poses_cpu.numpy().reshape(-1, 6)  # [num_ref_imgs, 6]
                ps_arr.append(pred_poses)  # ps_arr is a list of 'seq_lenth'-frame snippets

            # ATE, RE calculation
            if self.with_gt_pose:
                if self.seq_length == 2:
                    gt_poses = torch.cat([var_dict_t['gt_trg_pose'], var_dict_t['gt_ref_poses'][0]])[:,:3]
                elif self.seq_length > 2:
                    gt_poses = torch.cat([var_dict_t['gt_ref_poses'][0][:self.seq_length // 2], var_dict_t['gt_trg_pose'],
                                           var_dict_t['gt_ref_poses'][0][self.seq_length // 2:]])[:,:3]
                ATE, RE = compute_ATE_RE(poses_cpu, gt_poses)
                ATE_RE_errors[i] = ATE, RE

            # depth metric for frame i
            if self.with_gt_depth:
                depth_l_cpu = depth_l.cpu().data[0].numpy()[0]
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_metrics_i(depth_l_cpu,
                                            var_dict_t['gt_depth_l'].data[0].cpu().numpy(), max_depth=self.max_depth)
                depth_metrics[:, i] = abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

            # save every nth image concatenated with its outputs (depth visualization):
            if i%500 == 0:
                save_concat_img_results(var_dict_t, disp_l, depth_l, pose, explainability_mask, self.n_dof,
                                        self.rotation_mode, self.visualization_test_dir, self.n_iter, filenames_tgt)

            # measure elapsed time
            one_iter_time.update(time.time() - end)
            end = time.time()
            if self.debug and i%int(self.num_samples/self.batch_size/10) == 0:
                print('Test: finished processing batch {}'.format(i))

        return losses.avg, loss_names, filenames_tgt, filenames_ref, one_iter_time, ps_arr, ATE_RE_errors, depth_metrics


    def build(self):
        self._check_args()


        # load models and weights
        models_loaded, models, model_names, self.n_iter, self.epoch = load_model_and_weights(self.load_ckpt,
                self.load_ckpt_disp, self.FLAGS, use_cuda, ckpts_dir=self.ckpts_dir, train=False)

        if models_loaded:
            # run test
            t_begin = time.time()
            losses_avg, loss_names, filenames_tgt, filenames_ref, one_iter_time, ps_arr, ATE_RE_errors, depth_metrics = \
                self._test(models)

            # run pose metrics
            results_dir_lst, pose_metric_names, pose_metric_values = [], [], []
            if self.with_gt_pose and self.seq_length > 1:
                results_dir_lst, pose_metric_names, pose_metric_values = run_VO_metrics(self.data_dir, self.train_dir,
                                                    self.n_iter, self.data_loader, ps_arr, ATE_RE_errors, self.stereo)

            # compute depth metrics
            depth_metric_names, depth_metric_values = [], []
            if self.with_gt_depth:
                depth_metric_names, depth_metric_values = compute_depth_metrics(depth_metrics, self.train_dir,
                                                                                self.n_iter)

            # save test losses and metrics to results table
            metric_names = pose_metric_names + depth_metric_names
            metric_values = pose_metric_values + list(depth_metric_values)
            col_names = loss_names + metric_names
            values = losses_avg + metric_values
            save_loss_to_resultstable(values, col_names, self.results_table_path, self.n_iter, self.epoch, self.debug)

            # check if best model (saves best model not in debug mode)
            save_path = os.path.join(self.train_dir, 'ckpts')
            if self.with_gt_pose:
                is_best_pose = check_if_best_model_and_save(self.results_table_path, self.best_criteria_pose, models,
                                                       model_names, self.n_iter, self.epoch, save_path, self.debug,
                                                       suffix='pose')
                print('is_best_pose = ', is_best_pose)
                if not is_best_pose and not self.debug:
                    for results_dir in results_dir_lst:
                        shutil.rmtree(results_dir)
                    print("Removing {}".format(results_dir))
            if self.with_gt_depth:
                is_best_depth = check_if_best_model_and_save(self.results_table_path, self.best_criteria_pose, models,
                                                       model_names, self.n_iter, self.epoch, save_path, self.debug,
                                                       suffix='depth')
                print('is_best_depth = ', is_best_depth)
        # if self.seq_length == 2 and is_best_pose:
        #     # visualize predicted anf GT paths
        #     visualization_paths_test_dir = os.path.join(self.train_dir, 'visualization_paths_test')
        #     ensure_dir(visualization_paths_test_dir)
        #     visualization_trajectories(pred_ref_poses_abs, gt_ref_poses_abs, visualization_paths_test_dir, self.n_iter)
            metrics_str = ','.join([' {}: {:.4f}'.format(name, val) for name, val in zip(metric_names, metric_values)])
            print("\nTest: epoch {} (iter {}), time per test {:.2f}s, time per one batch (of {}) {:.2f}s, total avg "
                  "loss {:.4f}, {}\n".
                format(self.epoch, self.n_iter, time.time() - t_begin, self.batch_size, one_iter_time.avg[0],
                       losses_avg[0], metrics_str))

        else:
            print('no ckpt found for running test')
























