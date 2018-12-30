import os
import time

import numpy as np
import torch
import torch.utils.data as data

from dataloaders.dataloader_builder import DataLoader
from metrics.depth_metrics import compute_depth_metrics_i, compute_depth_metrics
from tests.test_builder import Test
from trains.losses import compute_loss
from utils.auxiliary import AverageMeter, save_loss_to_resultstable, ensure_dir, check_if_best_model_and_save, \
    save_concat_img_results, load_model_and_weights, convert_to_tensors, make_loss_dict, flip_and_concat_imgs

use_cuda = torch.cuda.is_available()


class test_depth(Test):
    def __init__(self, FLAGS):
        super(test_depth, self).__init__(FLAGS)
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.batch_size = FLAGS.batch_size
        assert self.batch_size == 1, 'batch size is not 1 in test'
        if hasattr(FLAGS, 'model'):
            self.model_names = list(FLAGS.model.split(','))
        else:
            self.model_names = ['DispNetS']
        self.stereo_test = False
        if hasattr(FLAGS, 'stereo_test') and FLAGS.stereo_test:
            self.stereo_test = True
        self.stereo = FLAGS.stereo
        self.best_criteria_depth = "RSME"
        if hasattr(FLAGS, 'best_criteria_depth'):
            self.best_criteria_depth = FLAGS.best_criteria_depth
        self.max_depth = 80
        if hasattr(FLAGS, 'max_depth'):
            self.max_depth = FLAGS.max_depth
        if hasattr(FLAGS, 'seed'):
            self.seed = FLAGS.seed
        self.debug = FLAGS.debug
        self.results_table_path = os.path.join(self.train_dir, 'results.csv')
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
        self.loss_params_dict = {'stereo': self.stereo_test, 'disp_norm': self.disp_norm, 'upscaling': self.upscaling,
                                 'edge_aware': self.edge_aware, 'concat_LR': self.concat_LR,
                                 'max_depth': self.max_depth, 'mode': 'test'}
        # dataloader
        self.dataloader, self.num_samples = DataLoader.dataloader_builder(FLAGS.data_loader, FLAGS, 'test',
                                                                     parent_class=data.Dataset)

    def _test(self, models):
        one_iter_time = AverageMeter()                         # time to execute one iteration
        losses = AverageMeter(i=len(self.loss_dict) + 1, precision=4)
        filenames_tgt = []
        depth_metrics = np.zeros((7, self.num_samples), np.float32)

        # switch to evaluate mode
        disp_net = models[0]
        disp_net.eval()

        end = time.time()
        for i, var_dict_np in enumerate(self.dataloader):
            # convert numpy input to pytorch tensors
            var_dict_t, filenames_tgt = convert_to_tensors(var_dict_np, filenames_tgt, self.loss_params_dict, use_cuda,
                                                           test=True)
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
            disp_r, depth_r = None, None
            if self.stereo_test and self.stereo and self.concat_LR:
                disp_r = disp[:, 1:, :, :]
                depth_r = 1 / disp_r
            elif self.stereo_test and (('w_RL' in self.loss_weights_dict and self.loss_weights_dict['w_RL'] > 0) or
                                    ('w_DC' in self.loss_weights_dict and self.loss_weights_dict['w_DC'] > 0)):
                disp_r = disp_net(var_dict_t['tgt_img_r'])
                depth_r = 1 / disp_r

            # compute loss
            losses_list, loss_names = compute_loss(var_dict_t, disp_l, depth_l, disp_r, depth_r, self.loss_weights_dict,
                                                   self.loss_dict, self.loss_params_dict)
            losses_list_cpu = [losses.data.item() for losses in losses_list]
            if len(losses) != len(losses_list_cpu):
                losses.reset(len(losses_list_cpu))
            losses.update(losses_list_cpu)

            # depth metric for frame i
            depth_l_cpu = depth_l.cpu().data[0].numpy()[0]
            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_metrics_i(depth_l_cpu,
                                        var_dict_t['gt_depth_l'].data[0].cpu().numpy(), max_depth=self.max_depth)
            depth_metrics[:, i] = abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

            # save every nth image concatenated with its outputs (depth visualization):
            if i%300 == 0:
                save_concat_img_results(var_dict_t, disp_l, self.visualization_test_dir, self.n_iter,
                                        filenames_tgt[i])

            # measure elapsed time
            one_iter_time.update(time.time() - end)
            end = time.time()
            if self.debug and i%int(self.num_samples/self.batch_size/10) == 0:
                print('Test: finished processing batch {}'.format(i))

        return losses.avg, loss_names, filenames_tgt, one_iter_time, depth_metrics


    def build(self):
        self._check_args()

        # load models and weights
        models_loaded, models, model_names, self.n_iter, self.epoch = load_model_and_weights(self.model_names,
                             self.load_ckpt, self.FLAGS, self.ckpts_dir, use_cuda, train=False)

        if models_loaded:
            # run test
            t_begin = time.time()
            losses_avg, loss_names, filenames_tgt, one_iter_time, depth_metrics = self._test(models)

            # compute depth metrics
            depth_metric_names, depth_metric_values = compute_depth_metrics(depth_metrics, self.train_dir, self.n_iter)

            # save test losses and metrics to results table
            col_names = loss_names + depth_metric_names
            values = losses_avg + list(depth_metric_values)
            save_loss_to_resultstable(values, col_names, self.results_table_path, self.n_iter, self.epoch, self.debug)

            # check if best model (saves best model if not in debug mode)
            min_value = True
            if self.best_criteria_depth in ["a1", "a2", "a3"]:
                min_value = False
            is_best_depth = check_if_best_model_and_save(self.results_table_path, self.best_criteria_depth, models,
                                model_names, self.n_iter, self.epoch, self.ckpts_dir, self.debug, min_value=min_value)
            print('is_best_depth = ', is_best_depth)
            metrics_str = ','.join([' {}: {:.4f}'.format(name, val) for name, val in zip(col_names, values)])
            print("\nTest: epoch {} (iter {}), time per test {:.2f}s, time per one batch (of {}) {:.2f}s, total avg "
                  "loss {:.4f}, {}\n".
                format(self.epoch, self.n_iter, time.time() - t_begin, self.batch_size, one_iter_time.avg[0],
                       losses_avg[0], metrics_str))

        else:
            print('no ckpt found for running test')























