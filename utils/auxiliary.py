import csv
import os
import shutil
from collections import OrderedDict
import matplotlib.pyplot as plt
from collections import Counter
import datetime
import numpy as np
import pandas as pd
from path import Path
from scipy.misc import imread
from wand.image import Image
from shutil import copyfile

from models.model_builder import Model
from utils.visualization import save_concat_imgs
from utils.bilinear_sampler import bilinear_sampler_1d

import torch
from torch.autograd import Variable


def make_loss_dict(loss_weights_str):
    loss_weights_str_split = loss_weights_str.split(',')

    loss_weights_dict = {}
    for item in loss_weights_str_split:
        k, v = item.split('-')
        loss_weights_dict['w_{}'.format(k)] = float(v)
    print('loss_weights_dict = ', loss_weights_dict)

    loss_dict = {}
    for weight_name, weight_value in loss_weights_dict.items():
        if weight_value > 0:
            loss_dict['loss_{}'.format(weight_name[2:])] = torch.zeros(1)

    return loss_weights_dict, loss_dict


def flip_and_concat_imgs(tgt_img_l_cpu, tgt_img_l_var):
    tgt_img_l_cpu_np = tgt_img_l_cpu.numpy().copy()
    tgt_img_l_cpu_np = ((tgt_img_l_cpu_np + 1) / 2 * 255).astype('int64')
    tgt_img_l_flipped = np.fliplr(tgt_img_l_cpu_np)
    tgt_img_l_flipped = 2 * (tgt_img_l_flipped / 255) - 1
    tgt_img_l_flipped_var = Variable(torch.from_numpy(tgt_img_l_flipped).type(torch.cuda.FloatTensor))
    disp_input = torch.cat((tgt_img_l_var, tgt_img_l_flipped_var), 1)
    return disp_input


def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)


def load_as_float(path):
    img_np = None
    if path is not None:
        img_np = imread(path).astype(np.float32)
        if len(img_np.shape) == 2:
            img_np = np.stack((img_np, img_np, img_np), axis=2)
    return img_np


def convert_to_tensors(var_dict_np, filenames_tgt, params_dict, use_cuda, test=False):
    assert 'tgt_img_l' in var_dict_np, '{} not given by dataloader'.format('tgt_img_l')
    var_dict_t = {}
    filename_tgt_cut = [str((var_dict_np['filename_tgt'][0].split("/")[-1]).split(".")[0]) for i in range(len(var_dict_np['filename_tgt']))]
    filenames_tgt.append(filename_tgt_cut[0:len(filename_tgt_cut)])
    var_dict_t['tgt_img_l_cpu'] = var_dict_np['tgt_img_l']
    if use_cuda:
        tgt_img_l = var_dict_np['tgt_img_l'].cuda()
        if test:
            with torch.no_grad():
                var_dict_t['tgt_img_l'] = Variable(tgt_img_l)
        else:
            var_dict_t['tgt_img_l'] = Variable(tgt_img_l)
    if test:
        with torch.no_grad():
            var_dict_t['gt_depth_l'] = Variable(var_dict_np['gt_depth_l'])
    else:
        var_dict_t['gt_depth_l'] = Variable(var_dict_np['gt_depth_l'])

    if params_dict['stereo']:
        assert 'tgt_img_r' in var_dict_np, '{} not given by dataloader'.format('tgt_img_r')
        if use_cuda:
            tgt_img_r = var_dict_np['tgt_img_r'].cuda()
        var_dict_t['tgt_img_r'] = Variable(tgt_img_r)
        if not test:
            var_dict_t['gt_depth_r'] = Variable(var_dict_np['gt_depth_r'])
    if test:
        with torch.no_grad():
            var_dict_t['gt_depth_r'] = Variable(var_dict_np['gt_depth_r'])

    return var_dict_t, filenames_tgt


def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1
    return points


def read_calib_file_kitty(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def get_intrinsics_baseline(calib_dir):
    cam2cam = read_calib_file_kitty(calib_dir + '/calib_cam_to_cam.txt')
    P2_rect = cam2cam['P_rect_02'].reshape(3, 4)
    P3_rect = cam2cam['P_rect_03'].reshape(3, 4)

    # cam 2 is left of camera 0  -6cm
    # cam 3 is to the right  +54cm
    b2 = P2_rect[0, 3] / -P2_rect[0, 0]
    b3 = P3_rect[0, 3] / -P3_rect[0, 0]
    baseline = b3 - b2

    intrinsics_l = P2_rect[:, :3].astype(np.float32)
    intrinsics_r = P3_rect[:, :3].astype(np.float32)

    return intrinsics_l, intrinsics_r, baseline


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def generate_depth_map(calib_dir, velo_file_name, im_shape, cam=2, odom=False):
    # load calibration files
    if odom:
        calib_data = read_calib_file_kitty(calib_dir / 'calib.txt')
        R_cam2rect = np.eye(4)
        # in KITTY_odom R_rect_00 is absent in the calib file, therefor took it from KITTI_raw/2011_09_26/calib_cam_to_cam.txt
        R_cam2rect[:3, :3] = np.array([9.999239e-01, 9.837760e-03, -7.445048e-03, -9.869795e-03, 9.999421e-01,
                                       -4.278459e-03, 7.402527e-03, 4.351614e-03, 9.999631e-01]).reshape(3, 3)
        P_rect_20 = np.reshape(calib_data['P2'], (3, 4))
        P_rect_30 = np.reshape(calib_data['P3'], (3, 4))
        velo2cam = np.reshape(calib_data['Tr'], (3, 4))
        velo2cam = np.vstack([velo2cam, [0, 0, 0, 1]])
        if cam == 2:
            P_velo2im = np.dot(np.dot(P_rect_20, R_cam2rect), velo2cam)
        elif cam == 3:
            P_velo2im = np.dot(np.dot(P_rect_30, R_cam2rect), velo2cam)
    else:
        cam2cam = read_calib_file_kitty(calib_dir / 'calib_cam_to_cam.txt')
        velo2cam = read_calib_file_kitty(calib_dir / 'calib_velo_to_cam.txt')
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

        # compute projection matrix velodyne->image plane
        R_cam2rect = np.eye(4)
        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
        P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)
        P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, -1:]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0
    return depth


def generate_mask(gt_depth, min_depth=1e-3, max_depth=100):
    '''for single frame i'''
    mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
    # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
    gt_height, gt_width = gt_depth.shape
    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)

    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    return mask


def generate_mask_tensor(gt_depth, min_depth=1e-3, max_depth=100):
    '''for a batch of frames
        - gt_depth (B,h,w) (np array) '''
    mask = ((gt_depth > min_depth) * (gt_depth < max_depth)).type(torch.cuda.ByteTensor)
    _, h, w = gt_depth.size()
    crop_mask = torch.zeros(mask.size())
    crop0 = int(0.40810811 * h)
    crop1 = int(0.99189189 * h)
    crop2 = int(0.03594771 * w)
    crop3 = int(0.96405229 * w)
    crop_mask[:, crop0:crop1, crop2:crop3] = 1
    crop_mask = crop_mask.type(torch.cuda.ByteTensor)
    mask = (mask * Variable(crop_mask, volatile=False)).type(torch.cuda.ByteTensor)
    return mask


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0]*i
        self.avg = [0]*i
        self.sum = [0]*i
        self.count = 0
        self.meters = i

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert(len(val) == self.meters)
        self.count += n
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)

    def __len__(self):
        return self.meters


def tensor2array(tensor, max_value=255, colormap='rainbow'):
    if max_value is None:
        max_value = tensor.max()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if cv2.__version__.startswith('3'):
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = ((255*tensor.squeeze())/max_value).numpy().clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy().transpose(1, 2, 0)*0.5
    return array


def _gray2rgb(im, cmap='plasma'):
  cmap = plt.get_cmap(cmap)
  rgba_img = cmap(im.astype(np.float32))
  rgb_img = np.delete(rgba_img, 3, 2)
  return rgb_img


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='plasma'):
  """
  (From Mahjourian Vid2Depth)
  Converts a depth map to an RGB image."""
  if normalizer is not None:
        depth /= normalizer
  else:
      depth /= (np.percentile(depth, pc) + 1e-6)
  depth = np.clip(depth, 0, 1.2)
  depth = _gray2rgb(np.max(depth) - depth, cmap=cmap)
  keep_h = int(depth.shape[0] * (1 - crop_percent))
  depth = depth[:keep_h]
  return depth


def check_if_best_model_and_save(results_table_path, best_criteria, models, model_names, iter, epoch, save_path, debug,
                                 min_value=True):
    ''' Get test losses and metrics from the results_table,
        decide whether the current loss/metric is the best.
        The decision is made based according to 'best_criteria', which should be a
        name of column in the results table.
        If best, save the ckpt as best (not in debug mode).
        If min_value=True, best criteria is the smalles item, else the greatest.'''
    filename = (results_table_path.split('/')[-1]).split('.')[0]
    results_table_path_tmp = Path(results_table_path).dirname() / '{}_tmp.csv'.format(filename)

    # check whether 'iter' is the best iteration according to the best criteria
    is_best = False
    if not debug:
        if os.path.isfile(results_table_path):
            copyfile(results_table_path, results_table_path_tmp)
            results_table = pd.read_csv(results_table_path_tmp, index_col=0)
            if len(results_table.index) > 0:
                col_names = results_table.columns.tolist()[1:]
                assert best_criteria in col_names, 'criteria for best model is not in the results table'
                best_criteria_col_np = np.array(results_table[best_criteria])
                if min_value:
                    best_criteria_value = np.min(best_criteria_col_np)
                else:
                    best_criteria_value = np.max(best_criteria_col_np)
                iter_col_np = np.array(results_table['iter'])
                assert iter in iter_col_np, 'current iteration is not in the results table'
                iter_idx = np.where(np.array(results_table['iter'])==iter)[0][0]
                iter_criteria_value = best_criteria_col_np[iter_idx]
                if ((min_value and iter_criteria_value <= best_criteria_value) or
                        (not min_value and iter_criteria_value >= best_criteria_value)):
                    is_best = True
            os.remove(results_table_path_tmp)

        if is_best:
            states = []
            for model in models:
                states.append({'iteration': iter, 'epoch': epoch, 'state_dict': model.state_dict()})
            save_checkpoint(save_path, states, model_names, is_best=True)

    return is_best


def save_checkpoint(save_path, states, state_names, is_best=False, filename='ckpt.pth.tar'):
    if is_best:
        for (prefix, state) in zip(state_names, states):
            best_ckpt_path = os.path.join(save_path, '{}_ckpt_best.pth.tar'.format(prefix))
            torch.save(state, best_ckpt_path)
            print('saved best {} model (iter {}) to {}'.format(prefix, state['iteration'], save_path))
    else:
        for (prefix, state) in zip(state_names, states):
            ckpt_path = os.path.join(save_path, '{}_{}'.format(prefix, filename))
            torch.save(state, ckpt_path)
            # print('saved {} model (iter {}) to {}'.format(prefix, state['iteration'], save_path))


def load_model_and_weights(model_names, load_ckpt, FLAGS, ckpts_dir, use_cuda, train=True):
    '''model_lst includes names of one or two models, separated by comma'''

    # load models
    num_models = len(model_names)
    assert num_models > 0 and num_models <= 2
    models = [Model.model_builder(model_name, FLAGS) for model_name in model_names]
    if use_cuda:
        models = [model.cuda() for model in models]
    models_loaded = False

    # load weights
    model_paths = []
    is_file = False
    ckpts_dir = Path(ckpts_dir)
    for i, model in enumerate(models):
        if load_ckpt != '':
            if i == 0:
                model_paths.append(Path(load_ckpt))
        elif load_ckpt == '' and not train:
            if i == 0 and num_models == 1:
                model_paths.append(ckpts_dir / '{}_ckpt.pth.tar'.format(model_names[i]))
        if len(model_paths) > 0:
            is_file = os.path.isfile(model_paths[0])
        # print('{} model: {}'.format(model_names[i], model))

    n_iter, n_epoch, epoch, train_iter = 0, 0, -1, 0
    if is_file:
        for i, model_path in enumerate(model_paths):
            model_path_tmp = ckpts_dir / '{}_ckpt_tmp.pth.tar'.format(model_names[i])
            copyfile(model_path, model_path_tmp)
            ckpt_dict = torch.load(model_path_tmp)
            if 'iteration' in ckpt_dict:
                train_iter = ckpt_dict['iteration']
            if 'epoch' in ckpt_dict:
                epoch = ckpt_dict['epoch']
            assert isinstance(ckpt_dict['state_dict'], (dict, OrderedDict)), type(ckpt_dict['state_dict'])
            models[i].load_state_dict(ckpt_dict['state_dict'], strict=False)
            if train:
                print('{} training resumed from the ckpt {} (epoch {}, iter {})'.format(model_names[i], model_paths[i],
                                                                                        epoch, train_iter))
            else:
                print('{} ckpt loaded from {} (epoch {}, iter {})'.format(model_names[i], model_paths[i], epoch,
                                                                          train_iter))
            if i == num_models-1:
                models_loaded = True
                os.remove(model_path_tmp)

        if train_iter > 0:
            if train:
                n_iter = train_iter + 1
            else:
                n_iter = train_iter
            n_epoch = epoch
    else:
        if train:
            for i, model in enumerate(models):
                model.init_weights()

    return models_loaded, models, model_names, n_iter, n_epoch


def save_train_losses_and_imgs_to_tensorboard_and_csv(var_dict_t, writer, losses_list, loss_names, disp, depth, n_iter,
                                                num_iters_for_print, loss_full_path, train_iters_dict, n_epoch, debug):
    if not debug:
        assert len(losses_list) == len(loss_names)

        # add losses to 'loss_full_path.csv'
        if not os.path.isfile(loss_full_path):
            with open(loss_full_path, 'w') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                loss_names = ['epoch', 'iter'] + loss_names
                csv_writer.writerow(loss_names[0:len(loss_names)])
        else:
            with open(loss_full_path, 'a') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                loss_val_list = [n_epoch, n_iter]
                loss_val_list += [l.data.item() for l in losses_list]
                csv_writer.writerow(loss_val_list[0:len(loss_val_list)])

        # add losses to tensorboard
        if os.path.isfile(loss_full_path):
            loss_table = pd.read_csv(loss_full_path, index_col=0)
            if len(loss_table.index) > 0:
                col_names = loss_table.columns.tolist()[1:]
                for row_idx in range(len(loss_table.index)):
                    test_iter = int(loss_table.iloc[row_idx]['iter'])
                    if test_iter not in train_iters_dict:
                        train_iters_dict[test_iter] = 1
                        for col_name in col_names:
                            writer.add_scalar(col_name + '_train', loss_table.iloc[row_idx][col_name], test_iter)

        # # add images to tensorboard
        # if n_iter % (num_iters_for_print * 10) == 0:
        #     writer.add_image('trg_img_train', tensor2array(var_dict_t['tgt_img_l_cpu'][0]), n_iter)
        #
        #     for k, scaled_depth in enumerate(depth):
        #         if not debug and k == 0 and n_iter % (num_iters_for_print * 10) == 0:
        #             writer.add_image('disp_s{}_train'.format(k),
        #                                   tensor2array(disp[k].data.item().cpu(), max_value=None, colormap='bone'),
        #                                   n_iter)
        #             writer.add_image('depth_s{}_train'.format(k),
        #                                   tensor2array(1 / disp[k].data.item().cpu(), max_value=10),
        #                                   n_iter)
    return train_iters_dict


def save_test_losses_to_tensorboard(test_iters_dict, results_table_path, writer, debug=False):
    ''' Get test losses from the results_table and add them to tensorboard'''
    filename = (results_table_path.split('/')[-1]).split('.')[0]
    results_table_path_tmp = Path(results_table_path).dirname() / '{}_tmp.csv'.format(filename)
    if os.path.isfile(results_table_path):
        copyfile(results_table_path, results_table_path_tmp)
        results_table = pd.read_csv(results_table_path_tmp, index_col=0)
        if len(results_table.index) > 0:
            col_names = results_table.columns.tolist()[2:]
            for row_idx in range(len(results_table.index)):
                test_iter = int(results_table.iloc[row_idx]['iter'])
                if test_iter not in test_iters_dict:
                    test_iters_dict[test_iter] = 1
                    if not debug and writer is not None:
                        for col_name in col_names:
                            writer.add_scalar(col_name+'_test',  results_table.iloc[row_idx][col_name], test_iter)

        if os.path.isfile(results_table_path_tmp):
            os.remove(results_table_path_tmp)

    return test_iters_dict


def save_loss_to_resultstable(values_list, col_names, results_table_path, n_iter, epoch, debug=False):
    '''Saves test losses and metrics to results table (not in debug mode)'''
    assert len(values_list) == len(col_names)

    if not debug:
        # init results table
        row_idx = 0
        if not os.path.isfile(results_table_path):
            columns = ['epoch'] + ['iter'] + col_names
            index = np.arange(1)
            results_table = pd.DataFrame(columns=columns, index=index)
        else:
            results_table = pd.read_csv(results_table_path, index_col=0)
            if len(results_table.index) > 0:
                iters = list(results_table['iter'])
                if float(n_iter) in iters:
                    row_idx = iters.index(n_iter)
                else:
                    row_idx = len(results_table.index)

        # write all values to the results table
        results_table.ix[row_idx, 'epoch'] = epoch
        results_table.ix[row_idx, 'iter'] = n_iter
        for i, value in enumerate(values_list):
            results_table.ix[row_idx, col_names[i]] = value
        results_table.to_csv(results_table_path, index=True)


def write_summary_to_csv(loss_summary_path, results_table_path, n_iter, epoch, train_loss):
    ''' define results and loss writers '''

    if not os.path.isfile(loss_summary_path):
        with open(loss_summary_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['epoch', 'iter', 'train_loss', 'test_loss'])

    # get test loss for current iteration
    test_loss = -1
    filename = (results_table_path.split('/')[-1]).split('.')[0]
    results_table_path_tmp = Path(results_table_path).dirname() / '{}_tmp.csv'.format(filename)
    if os.path.isfile(results_table_path):
        copyfile(results_table_path, results_table_path_tmp)
        results_table = pd.read_csv(results_table_path_tmp, index_col=0)
        if len(results_table.index) > 0:
            iter_col_np = np.array(results_table['iter'])
            if n_iter in iter_col_np:
                iter_idx = np.where(np.array(results_table['iter']) == n_iter)[0][0]
                test_loss = results_table.iloc[iter_idx]['tot_loss']
        os.remove(results_table_path_tmp)

    # write train and test losses to 'loss_summary.csv
    with open(loss_summary_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([epoch, n_iter, train_loss, test_loss])

    return test_loss


def save_concat_img_results(var_dict_t, disp, visualization_test_dir, n_iter, filename_tgt_cut):
    images = [tensor2array(var_dict_t['tgt_img_l_cpu'][0])]
    images.append(normalize_depth_for_display(var_dict_t['gt_depth_r'].data[0].cpu().numpy()))
    # images.append(tensor2array(disp.data[0].cpu(), max_value=None, colormap='bone'))
    depth = 1.0 / (disp.data[0].cpu().squeeze().numpy() + 1e-6)
    images.append(normalize_depth_for_display(depth))
    img_names = ['trg', 'disp', 'depth', 'gt_depth']
    save_path = os.path.join(visualization_test_dir, 'img_comb_{}_{}.jpg'.format(n_iter, filename_tgt_cut[0]))
    save_concat_imgs(images, img_names, save_path)


def generate_image_left(img_r, disp_l):
    return bilinear_sampler_1d(img_r, -disp_l)


def generate_image_right(img_l, disp_r):
    return bilinear_sampler_1d(img_l, disp_r)























