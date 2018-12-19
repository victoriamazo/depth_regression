import csv
import os
import shutil
from collections import OrderedDict
import matplotlib.pyplot as plt
from collections import Counter
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from path import Path
from scipy.misc import imread
from wand.image import Image
from shutil import copyfile

from models.model_builder import Model
from utils.inverse_warp import pose_vec2mat, quat2euler_arr, inverse_warp
from utils.visualization import save_concat_imgs

from torch.autograd import Variable


def convert_pdf2jpg(root_dir, n_iter, suffix='no', sequence='09'):
    '''Converts and saves pdf paths to jmg for all paths in results
    to quickly see the non-scaled paths.
    Suffix can be:
      - 'no' for non-scaled paths
      - 'scaling' for scaled paths
    '''
    if suffix == 'no':
        save_dir = os.path.join(root_dir, 'non-scaled_paths')
    else:
        save_dir = os.path.join(root_dir, 'scaled_paths')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for subdir in os.listdir(root_dir):
        if subdir.startswith('results_{}'.format(n_iter)):
            subdir_path = os.path.join(root_dir, subdir)
            n_iter = subdir.split('_')[1]
            subdir_suffix = ''
            if len(subdir.split('_')) > 2:
                subdir_suffix = subdir.split('_')[2]
            if subdir_suffix == suffix:
                pdf_path = os.path.join(subdir_path, 'plot_path/sequence_{}.pdf'.format(sequence))
                jpg_path = os.path.join(save_dir, '{}.jpg'.format(n_iter))
                with Image(filename=pdf_path, resolution=200) as img:
                    # keep good quality
                    img.compression_quality = 80
                    # save it to tmp name
                    img.save(filename=jpg_path)


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


def idx2label(data_dir):
    dataset = data_dir.split('/')[-1]
    label_dict = {}
    if dataset == 'mnist':
        label_dict = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '0': 0}
    return label_dict


def convert_to_onehot(label, label_dict, num_classes):
    i = int(label_dict[label])
    onehot = np.zeros(num_classes)
    onehot[i] = 1
    return onehot


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


def read_calib_file(path, is_left):
    f = open(path, 'r')
    s = f.readlines()
    f.close()
    assert len(s) == 4
    intrinsics_np = np.zeros(4)
    for i, line in enumerate(s):
        line_split = [i for i in line.split(" ")][1:]
        if (i == 2 and is_left) or (i == 3 and not is_left):
            intrinsics_np[0] = float(line_split[0])
            intrinsics_np[1] = float(line_split[2])
            intrinsics_np[2] = float(line_split[5])
            intrinsics_np[3] = float(line_split[6])
            b = -float(line_split[3])/float(line_split[0])          # baseline (in m)
    return intrinsics_np.astype(np.float32), b


def intrinsics_matrix(intrinsics_np, is_kitty=False):
    assert intrinsics_np.shape[0] == 4
    intrinsics_matrix = np.zeros((3, 3)).astype(np.float32)
    intrinsics_matrix[0, 0] = intrinsics_np[0]
    intrinsics_matrix[1, 2] = intrinsics_np[3]
    intrinsics_matrix[2, 2] = 1.0
    if is_kitty:
        intrinsics_matrix[1, 1] = intrinsics_np[2]
        intrinsics_matrix[0, 2] = intrinsics_np[1]
    else:
        intrinsics_matrix[1, 1] = intrinsics_np[1]
        intrinsics_matrix[0, 2] = intrinsics_np[2]
    return intrinsics_matrix


def calculate_rmse(gt_ref_poses_abs, pred_ref_poses_abs, results_table_path, n_iter):
    '''gt_poses and pred_poses are numpy arrays of shape [N, (tx, ty, tz, rx, ry, rz)]'''
    abs_error_pos = np.abs(pred_ref_poses_abs[:, :3] - gt_ref_poses_abs[:, :3]).mean()
    abs_error_ang = np.abs(pred_ref_poses_abs[:, 3:] - gt_ref_poses_abs[:, 3:]).mean()
    # rmse_error_pos_mean = np.sqrt(((pred_ref_poses_abs[:, :3] - gt_ref_poses_abs[:, :3])**2).mean())
    rmse_pos = np.linalg.norm(pred_ref_poses_abs[:, :3] - gt_ref_poses_abs[:, :3]) / np.sqrt(
        len(pred_ref_poses_abs) * 3)
    rmse_ang = np.linalg.norm(pred_ref_poses_abs[:, 3:] - gt_ref_poses_abs[:, 3:]) / np.sqrt(
        len(pred_ref_poses_abs) * 3)

    # save rmse to results table
    if os.path.isfile(results_table_path):
        results_table = pd.read_csv(results_table_path, index_col=0)
        iters = list(results_table['iter'])
        if float(n_iter) in iters:
            n_iter_idx = iters.index(n_iter)
            if 'rmse_pos' not in results_table:
                results_table['rmse_pos'] = None
            if 'rmse_ang' not in results_table:
                results_table['rmse_ang'] = None
            results_table.ix[n_iter_idx, 'rmse_pos'] = rmse_pos
            results_table.ix[n_iter_idx, 'rmse_ang'] = rmse_ang
            results_table.to_csv(results_table_path, index=True)
    return rmse_pos, rmse_ang, abs_error_pos, abs_error_ang


def getKT(height, width, calib_file_path):
    '''Get intrinsic (K) and extrinsic (T) camera matrices for raw KITTY dataset'''
    # ----------------------------------------------------------------------
    # Get K (camera intrinsic) and T (camera extrinsic)
    # ----------------------------------------------------------------------
    new_image_size = [float(height), float(width)]

    # ----------------------------------------------------------------------
    # Get original K
    # ----------------------------------------------------------------------
    f = open(calib_file_path, 'r')
    camTxt = f.readlines()
    f.close()
    K_dict = {}
    for line in camTxt:
        line_split = line.split(":")
        K_dict[line_split[0]] = line_split[1]

    # ----------------------------------------------------------------------
    # original K02
    # ----------------------------------------------------------------------
    P_split = K_dict["P_rect_02"].split(" ")
    S_split = K_dict["S_rect_02"].split(" ")
    ref_img_size = [float(S_split[2]), float(S_split[1])] # height, width

    # ----------------------------------------------------------------------
    # Get new K & position
    # ----------------------------------------------------------------------
    W_ratio = new_image_size[1] / ref_img_size[1]
    H_ratio = new_image_size[0] / ref_img_size[0]
    fx = float(P_split[1]) * W_ratio
    fy = float(P_split[6]) * H_ratio
    cx = float(P_split[3]) * W_ratio
    cy = float(P_split[7]) * H_ratio

    tx_L = float(P_split[4]) / float(P_split[1])
    # ty_L = float(P_split[8]) / float(P_split[6])

    # ----------------------------------------------------------------------
    # original K03
    # ----------------------------------------------------------------------
    P_split = K_dict["P_rect_03"].split(" ")
    S_split = K_dict["S_rect_03"].split(" ")

    tx_R = float(P_split[4]) / float(P_split[1])
    # ty_R = float(P_split[8]) / float(P_split[6])

    # ----------------------------------------------------------------------
    # Get position of Right camera w.r.t Left
    # ----------------------------------------------------------------------
    # Tx is a baseline between the cameras
    Tx = np.abs(tx_R - tx_L)
    # Ty = np.abs(tx_R - tx_L)

    se3 = [0,0,0,Tx,0,0]                    #[rx, ry, rz, tx, ty, tz]

    return [fx,fy,cx,cy,se3]


def convert_to_tensors(var_dict_np, filenames_tgt, filenames_ref, batch_size, params_dict, use_cuda, test=False):
    assert 'tgt_img_l' in var_dict_np, '{} not given by dataloader'.format('tgt_img_l')
    assert 'ref_imgs_l' in var_dict_np, '{} not given by dataloader'.format('ref_imgs_l')
    assert 'intrinsics_l' in var_dict_np, '{} not given by dataloader'.format('intrinsics_l')
    assert 'intrinsics_inv_l' in var_dict_np, '{} not given by dataloader'.format('intrinsics_inv_l')
    var_dict_t = {}
    filename_tgt_cut = [(var_dict_np['filename_tgt'][i].split("/")[-1]).split(".")[0] for i in range(len(var_dict_np['filename_tgt']))]
    filenames_tgt.append(filename_tgt_cut[0:len(filename_tgt_cut)])
    filename_ref_cut = [(var_dict_np['filename_ref'][i][0].split("/")[-1]).split(".")[0] for i in range(len(var_dict_np['filename_ref']))]
    filenames_ref.append(filename_ref_cut[0:len(filename_ref_cut)])
    var_dict_t['tgt_img_l_cpu'] = var_dict_np['tgt_img_l']
    var_dict_t['ref_imgs_l_cpu'] = var_dict_np['ref_imgs_l'][0]
    if use_cuda:
        tgt_img_l = var_dict_np['tgt_img_l'].cuda()
        ref_imgs_l = [ref_img_l.cuda() for ref_img_l in var_dict_np['ref_imgs_l']]
        intrinsics_l = var_dict_np['intrinsics_l'].cuda()
        intrinsics_inv_l = var_dict_np['intrinsics_inv_l'].cuda()
        var_dict_t['tgt_img_l'] = Variable(tgt_img_l, volatile=test)
    var_dict_t['ref_imgs_l'] = [Variable(ref_img_l, volatile=test) for ref_img_l in ref_imgs_l]
    var_dict_t['intrinsics_l'] = Variable(intrinsics_l, volatile=test)
    var_dict_t['intrinsics_l_inv'] = Variable(intrinsics_inv_l, volatile=test)
    if params_dict['with_gt_pose']:
        var_dict_t['gt_trg_pose'] = Variable(var_dict_np['gt_trg_pose'], volatile=test)
        var_dict_t['gt_ref_poses'] = Variable(var_dict_np['gt_ref_poses'], volatile=test)
    if params_dict['with_gt_depth']:
        var_dict_t['gt_depth_l'] = Variable(var_dict_np['gt_depth_l'], volatile=test)

    if params_dict['stereo']:
        assert 'tgt_img_r' in var_dict_np, '{} not given by dataloader'.format('tgt_img_r')
        assert 'ref_imgs_r' in var_dict_np, '{} not given by dataloader'.format('ref_imgs_r')
        assert 'intrinsics_r' in var_dict_np, '{} not given by dataloader'.format('intrinsics_r')
        assert 'intrinsics_inv_r' in var_dict_np, '{} not given by dataloader'.format('intrinsics_inv_r')
        var_dict_t['ref_imgs_r_cpu'] = var_dict_np['ref_imgs_r'][0]
        T_LR = np.array(batch_size * [var_dict_np['baseline'][0], 0, 0, 0, 0, 0]).reshape(batch_size, 1, -1)  # (B, (tx, ty, tz, rx, ry, rz))
        T_LR = torch.from_numpy(T_LR).float()
        if use_cuda:
            tgt_img_r = var_dict_np['tgt_img_r'].cuda()
            ref_imgs_r = [ref_img_r.cuda() for ref_img_r in var_dict_np['ref_imgs_r']]
            T_LR = T_LR.cuda()
            intrinsics_r = var_dict_np['intrinsics_r'].cuda()
            intrinsics_inv_r = var_dict_np['intrinsics_inv_r'].cuda()
        var_dict_t['tgt_img_r'] = Variable(tgt_img_r)
        var_dict_t['ref_imgs_r'] = [Variable(ref_img_r, volatile=test) for ref_img_r in ref_imgs_r]
        var_dict_t['intrinsics_r'] = Variable(intrinsics_r, volatile=test)
        var_dict_t['intrinsics_r_inv'] = Variable(intrinsics_inv_r, volatile=test)
        var_dict_t['T_LR'] = Variable(T_LR)
        if params_dict['with_gt_depth']:
            var_dict_t['gt_depth_r'] = Variable(var_dict_np['gt_depth_r'], volatile=test)

    return var_dict_t, filenames_tgt, filenames_ref
# def convert_to_tensors(var_list, filenames_tgt, filenames_ref, batch_size, stereo, use_cuda, test=True):
#     if stereo:
#         tgt_img_l, ref_imgs_l, tgt_img_r, ref_imgs_r, intrinsics_l, intrinsics_r, intrinsics_inv_l, \
#         intrinsics_inv_r, gt_trg_pose, gt_ref_poses, filename_tgt_b, filenames_ref_b, baseline = var_list
#     else:
#         tgt_img_l, ref_imgs_l, intrinsics_l, intrinsics_inv_l, gt_trg_pose, gt_ref_poses, filename_tgt_b, \
#         filenames_ref_b = var_list
#
#     filename_tgt_cut = [(filename_tgt_b[i].split("/")[-1]).split(".")[0] for i in range(len(filename_tgt_b))]
#     filenames_tgt.append(filename_tgt_cut[0:len(filename_tgt_cut)])
#     filename_ref_cut = [(filenames_ref_b[i][0].split("/")[-1]).split(".")[0] for i in range(len(filenames_ref_b))]
#     filenames_ref.append(filename_ref_cut[0:len(filename_ref_cut)])
#     tgt_img_l_cpu = tgt_img_l
#     ref_imgs_l_cpu = ref_imgs_l[0]
#     if use_cuda:
#         tgt_img_l = tgt_img_l.cuda()
#         ref_imgs_l = [ref_img_l.cuda() for ref_img_l in ref_imgs_l]
#         intrinsics_l = intrinsics_l.cuda()
#         intrinsics_inv_l = intrinsics_inv_l.cuda()
#     tgt_img_l_var = Variable(tgt_img_l, volatile=test)
#     ref_imgs_l_var = [Variable(ref_img_l, volatile=test) for ref_img_l in ref_imgs_l]
#     intrinsics_l_var = Variable(intrinsics_l, volatile=test)
#     intrinsics_l_inv_var = Variable(intrinsics_inv_l, volatile=test)
#
#     if stereo:
#         ref_imgs_r_cpu = ref_imgs_r[0]
#         T_LR = np.array(batch_size * [baseline[0], 0, 0, 0, 0, 0]).reshape(batch_size, 1,
#                                                                            -1)  # (B, (tx, ty, tz, rx, ry, rz))
#         T_LR = torch.from_numpy(T_LR).float()
#         if use_cuda:
#             ref_imgs_r = [ref_img_r.cuda() for ref_img_r in ref_imgs_r]
#             tgt_img_r = tgt_img_r.cuda()
#             T_LR = T_LR.cuda()
#             intrinsics_r = intrinsics_r.cuda()
#             intrinsics_inv_r = intrinsics_inv_r.cuda()
#         ref_imgs_r_var = [Variable(ref_img_r, volatile=test) for ref_img_r in ref_imgs_r]
#         tgt_img_r_var = Variable(tgt_img_r)
#         intrinsics_r_var = Variable(intrinsics_r, volatile=test)
#         intrinsics_r_inv_var = Variable(intrinsics_inv_r, volatile=test)
#         T_LR_var = Variable(T_LR)
#
#         return tgt_img_l_cpu, tgt_img_l_var, ref_imgs_l_cpu, ref_imgs_l_var, tgt_img_r_var, ref_imgs_r_cpu, \
#                ref_imgs_r_var, intrinsics_l_var, intrinsics_r_var, intrinsics_l_inv_var, intrinsics_r_inv_var, gt_trg_pose, \
#                gt_ref_poses, filenames_tgt, filenames_ref, T_LR_var
#     else:
#         return tgt_img_l_cpu, tgt_img_l_var, ref_imgs_l_cpu, ref_imgs_l_var, intrinsics_l_var, \
#                intrinsics_l_inv_var, gt_trg_pose, gt_ref_poses, filenames_tgt, filenames_ref


def read_pose_csv(pose_gt_csv_path, filename2time_csv_path, euler_angles=True):
    # read filename-to-time table
    filename2time_table = pd.read_csv(filename2time_csv_path, sep=',', header=None)
    timestamp_file_str = np.array(filename2time_table[0][1:]).astype('str')
    filename = np.array(filename2time_table[1][1:]).astype('str')

    # read pose table
    pose_table = pd.read_csv(pose_gt_csv_path, sep=',', header=None)
    timestamp_pose_str = np.array(pose_table[0][1:]).astype('str')
    tx_raw = np.array(pose_table[1][1:]).astype('float32')
    ty_raw = np.array(pose_table[2][1:]).astype('float32')
    tz_raw = np.array(pose_table[3][1:]).astype('float32')
    qw_raw = np.array(pose_table[4][1:]).astype('float32')
    qx_raw = np.array(pose_table[5][1:]).astype('float32')
    qy_raw = np.array(pose_table[6][1:]).astype('float32')
    qz_raw = np.array(pose_table[7][1:]).astype('float32')


    # since timestamp is in format 'datetime', find the first digit of relevant time scale
    ## and cut timestamps correspondingly
    timestamp_pose_first = timestamp_pose_str[0]
    timestamp_pose_last = timestamp_pose_str[-1]
    same_digit_bool = [timestamp_pose_first[i]==timestamp_pose_last[i] for i in range(len(timestamp_pose_first))]
    for first_diff_digit in range(len(same_digit_bool)):
        if same_digit_bool[first_diff_digit] == False:
            break
    timestamp_pose_int = np.array([int(timestamp_pose_str[i][first_diff_digit:]) for i in range(len(timestamp_pose_str))])
    timestamp_file_int = np.array([int(timestamp_file_str[i][first_diff_digit:]) for i in range(len(timestamp_file_str))])


    # find corresponding timestamp_file to timestamp_pose
    first_timestamp_file_idx = np.where((timestamp_file_int-timestamp_pose_int[0]) >= 0)[0][0]
    file_idxs = [first_timestamp_file_idx]
    pose_idxs = [0]
    for timestamp_file_idx in range(first_timestamp_file_idx+1, len(timestamp_file_int), 1):
        timestamp_pose_idx = np.where((timestamp_pose_int - timestamp_file_int[timestamp_file_idx]) >= 0)[0]
        if len(timestamp_pose_idx) != 0:
            file_idxs.append(timestamp_file_idx)
            pose_idxs.append(timestamp_pose_idx[0])
    file_idxs = np.array(file_idxs)
    pose_idxs = np.array(pose_idxs)
    assert len(file_idxs) == len(pose_idxs)

    # corresponding pose
    filenames_with_gt = list(filename[file_idxs])
    timestamp = timestamp_file_int[file_idxs]
    timestamp -= timestamp[0]
    tx = tx_raw[pose_idxs]
    ty = ty_raw[pose_idxs]
    tz = tz_raw[pose_idxs]
    qw = qw_raw[pose_idxs]
    qx = qx_raw[pose_idxs]
    qy = qy_raw[pose_idxs]
    qz = qz_raw[pose_idxs]

    # convert quaternion to angles
    if euler_angles:
        q = np.stack((qw, qz, qy, qx), axis=1)
        ry, rz, rx = quat2euler_arr(q)
        angles = (rx, ry, rz)
    else:
        angles = (qx, qy, qz, qw)

    return filenames_with_gt, angles, tx, ty, tz


def read_KITTY_poses(file_name):
    # ----------------------------------------------------------------------
    # Each line in the file should follow one of the following structures
    # (1) idx pose(3x4 matrix in terms of 12 numbers)
    # (2) pose(3x4 matrix in terms of 12 numbers)
    # ----------------------------------------------------------------------
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    poses = {}
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ")]
        withIdx = int(len(line_split) == 13)
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row * 4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        poses[frame_idx] = P
    return poses


def read_scene_data_KITTY(data_root, sequence_set, seq_length=3, step=1):
    data_root = Path(data_root)
    # im_sequences = []
    poses_sequences = []
    indices_sequences = []
    demi_length = (seq_length - 1) // 2
    shift_range = np.array([step*i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1)

    sequences = set()
    for seq in sequence_set:
        corresponding_dirs = set((data_root).dirs(seq))
        sequences = sequences | corresponding_dirs

    # print('getting test metadata for theses sequences : {}'.format(sequences))
    data_dir = Path('/'.join(data_root.split('/')[:-1]))
    for sequence in sequences:
        poses = np.genfromtxt(data_dir/'poses'/'{}.txt'.format(sequence.name)).astype(np.float64).reshape(-1, 3, 4)
        imgs = sorted((sequence/'image_2').files('*.png'))
        # construct 5-snippet sequences
        tgt_indices = np.arange(demi_length, len(imgs) - demi_length).reshape(-1, 1)
        snippet_indices = shift_range + tgt_indices
        # im_sequences.append(imgs)
        poses_sequences.append(poses)
        indices_sequences.append(snippet_indices)
    return poses_sequences, indices_sequences


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
        calib_dir = '/media/victoria/d/data/KITTI_raw/2011_09_26'
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


def get_displacements(oxts_root, index, shifts):
    with open(oxts_root / 'timestamps.txt') as f:
        timestamps = [datetime.datetime.strptime(ts[:-3], "%Y-%m-%d %H:%M:%S.%f").timestamp() for ts in
                      f.read().splitlines()]
    oxts_data = np.genfromtxt(oxts_root / 'data' / '{:010d}.txt'.format(index))
    speed = np.linalg.norm(oxts_data[8:11])
    assert (all(index + shift < len(timestamps) and index + shift >= 0 for shift in shifts)), str(
        [index + shift for shift in shifts])
    return [speed * abs(timestamps[index] - timestamps[index + shift]) for shift in shifts]


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
            array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
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


def normalize_depth_for_display(disp, pc=95, crop_percent=0, normalizer=None, cmap='plasma'):
  """
  (From Mahjourian Vid2Depth)
  Converts a depth map to an RGB image."""
  disp = disp.squeeze().numpy()
  depth = 1.0 / (disp + 1e-6)
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
                                 suffix='pose', metric=True):
    ''' Get test losses and metrics from the results_table,
        decide whether the current loss/metric is the best.
        The decision is made based according to 'best_criteria', which should be a
        name of column in the results table.
        If best, save the ckpt as best (not in debug mode).
        If metric=True, best criteria is the largest item, else the smallest.'''
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
                if metric:
                    best_criteria_value = np.max(best_criteria_col_np)
                else:
                    best_criteria_value = np.min(best_criteria_col_np)
                iter_col_np = np.array(results_table['iter'])
                assert iter in iter_col_np, 'current iteration is not in the results table'
                iter_idx = np.where(np.array(results_table['iter'])==iter)[0][0]
                iter_criteria_value = best_criteria_col_np[iter_idx]
                if ((metric and iter_criteria_value >= best_criteria_value) or
                        (not metric and iter_criteria_value <= best_criteria_value)):
                    is_best = True
            os.remove(results_table_path_tmp)

        if is_best:
            states = []
            for model in models:
                states.append({'iteration': iter, 'epoch': epoch, 'state_dict': model.state_dict()})
            save_checkpoint(save_path, states, model_names, is_best=True, suffix=suffix)

    return is_best


def save_checkpoint(save_path, states, state_names, is_best=False, filename='ckpt.pth.tar', suffix='pose'):
    if is_best:
        for (prefix, state) in zip(state_names, states):
            best_ckpt_path = os.path.join(save_path, '{}_ckpt_best_{}.pth.tar'.format(prefix, suffix))
            torch.save(state, best_ckpt_path)
            print('saved best {} model (iter {}) to {}'.format(prefix, state['iteration'], save_path))
    else:
        for (prefix, state) in zip(state_names, states):
            ckpt_path = os.path.join(save_path, '{}_{}'.format(prefix, filename))
            torch.save(state, ckpt_path)
            # print('saved {} model (iter {}) to {}'.format(prefix, state['iteration'], save_path))


def save_pose_to_file(filenames_tgt, filenames_ref, pred_poses_delta, pred_poses_tgt, gt_poses_delta, gt_poses_tgt,
                      train_dir, n_iter):
    '''
        Input:
            - filenames_tgt of the tgt images, filenames_ref of the corresponding reference image
            - pred_poses_delta and gt_poses_delta - np arrays of shape [N*B, 6]
        Save predicted pose to file in formats:
         - pose_mat_np_flat: matrix form
         - pose_mat_np_cum_flat: matrix form of cummulative changes in position
         - pose_np_flat: in format (tx, ty, tz, rx, ry, rz)
         - pose_6dof_np_cum_flat: cummulative changes in position in format (tx, ty, tz, rx, ry, rz)
        '''
    pose_file_ang = os.path.join(train_dir, 'pose_6dof_best_{}.csv'.format(n_iter))
    pose_file_ang_iter = [file for file in os.listdir(train_dir) if file.startswith('pose_6dof_best_')]
    if len(pose_file_ang_iter) > 0:
        pose_file_ang_iter = os.path.join(train_dir, pose_file_ang_iter[0])
        if os.path.isfile(pose_file_ang_iter):
            os.remove(pose_file_ang_iter)
    with open(pose_file_ang, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['filename_tgt', 'filename_ref', 'delta_tx_pred', 'delta_ty_pred', 'delta_tz_pred',
                         'delta_rx_pred', 'delta_ry_pred', 'delta_rz_pred', 'delta_tx_gt', 'delta_ty_gt', 'delta_tz_gt',
                         'delta_rx_gt', 'delta_ry_gt', 'delta_rz_gt'])

    pose_file_ang_cum = os.path.join(train_dir, 'pose_6dof_cum_best_{}.csv'.format(n_iter))
    pose_file_ang_cum_iter = [file for file in os.listdir(train_dir) if file.startswith('pose_6dof_cum_best_')]
    if len(pose_file_ang_cum_iter) > 0:
        pose_file_ang_cum_iter = os.path.join(train_dir, pose_file_ang_cum_iter[0])
        if os.path.isfile(pose_file_ang_cum_iter):
            os.remove(pose_file_ang_cum_iter)
    with open(pose_file_ang_cum, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['filename_tgt', 'tx_pred', 'ty_pred', 'tz_pred', 'rx_pred', 'ry_pred', 'rz_pred', 'tx_gt',
                         'ty_gt', 'tz_gt', 'rx_gt', 'ry_gt', 'rz_gt'])

    pose_file_mat = os.path.join(train_dir, 'pose_mat_best_{}.csv'.format(n_iter))
    pose_file_mat_iter = [file for file in os.listdir(train_dir) if file.startswith('pose_mat_best_')]
    if len(pose_file_mat_iter) > 0:
        pose_file_mat_iter = os.path.join(train_dir, pose_file_mat_iter[0])
        if os.path.isfile(pose_file_mat_iter):
            os.remove(pose_file_mat_iter)
    with open(pose_file_mat, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['filename_tgt', 'filename_ref', 'r00_pred', 'r01_pred', 'r02_pred', 'tx_pred', 'r10_pred',
                         'r11_pred', 'r12_pred', 'ty_pred', 'r20_pred', 'r21_pred', 'r22_pred', 'tz_pred'])

    if len(pred_poses_delta) > 0:
        pose_mat_np_cum = np.eye(4)
        assert len(filenames_tgt) == len(filenames_ref)
        assert len(gt_poses_delta) == len(gt_poses_tgt)
        for i in range(len(filenames_tgt)):
            pred_pose_np = pred_poses_delta[i, :]                    # np array [6]
            pose_np_flat = pred_pose_np.reshape(1, -1)               # np array [1, 6]
            pose_tensor = torch.from_numpy(pose_np_flat).float()
            pos_mat = pose_vec2mat(pose_tensor, detach=False)
            pose_mat_np = pos_mat[0, :, :].cpu().numpy()
            # pose_mat_np_ext = np.vstack((pose_mat_np, np.array([0, 0, 0, 1]))).reshape(4, 4)
            # pose_mat_np_cum = np.dot(pose_mat_np_cum, pose_mat_np_ext)  #np.dot(pose_mat_np_ext, pose_mat_np_cum)

            with open(pose_file_ang, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                pred_pos_ang = np.around(pred_pose_np, decimals=10)
                if len(gt_poses_delta) > 0:
                    gt_pose_np = gt_poses_delta[i, :]  # np array [6]
                    gt_pos_ang = np.around(gt_pose_np, decimals=10)
                    writer.writerow([filenames_tgt[i]+'.png', filenames_ref[i]+'.png', pred_pos_ang[0], pred_pos_ang[1], pred_pos_ang[2],
                                     pred_pos_ang[3], pred_pos_ang[4], pred_pos_ang[5], gt_pos_ang[0], gt_pos_ang[1], gt_pos_ang[2],
                                     gt_pos_ang[3], gt_pos_ang[4], gt_pos_ang[5]])
                else:
                    writer.writerow([filenames_tgt[i]+'.png', filenames_ref[i]+'.png', pred_pos_ang[0], pred_pos_ang[1], pred_pos_ang[2],
                                     pred_pos_ang[3], pred_pos_ang[4], pred_pos_ang[5]])

            with open(pose_file_ang_cum, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                pred_pose_6dof_cum = np.around(pred_poses_tgt[i, :], decimals=10)
                if len(gt_poses_tgt) > 0:
                    gt_pose_6dof_cum = np.around(gt_poses_tgt[i, :], decimals=10)
                    writer.writerow([filenames_tgt[i]+'.png', pred_pose_6dof_cum[0], pred_pose_6dof_cum[1],
                                     pred_pose_6dof_cum[2], pred_pose_6dof_cum[3], pred_pose_6dof_cum[4], pred_pose_6dof_cum[5],
                                     gt_pose_6dof_cum[0], gt_pose_6dof_cum[1], gt_pose_6dof_cum[2], gt_pose_6dof_cum[3],
                                     gt_pose_6dof_cum[4], gt_pose_6dof_cum[5]])
                else:
                    writer.writerow([filenames_tgt[i] + '.png', pred_pose_6dof_cum[0], pred_pose_6dof_cum[1],
                                     pred_pose_6dof_cum[2], pred_pose_6dof_cum[3], pred_pose_6dof_cum[4], pred_pose_6dof_cum[5]])

            with open(pose_file_mat, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                pose_mat = np.around(pose_mat_np, decimals=10).reshape(-1)
                writer.writerow([filenames_tgt[i]+'.png', filenames_ref[i]+'.png', pose_mat[0], pose_mat[1], pose_mat[2], pose_mat[3],
                                 pose_mat[4], pose_mat[5], pose_mat[6], pose_mat[7],
                                 pose_mat[8], pose_mat[9], pose_mat[10], pose_mat[11]])


def save_mat_pose_to_file(gt_poses, results_dir, sequence):
    '''
        Input:
            - gt_poses - np array of GT poses in the matrix form of shape (N, 3, 4)
        '''
    pose_file = os.path.join(results_dir, '{}.txt'.format(sequence))
    gt_poses = np.around(gt_poses.reshape(gt_poses.shape[0], -1), decimals=10)
    np.savetxt(pose_file, gt_poses, delimiter=' ')


def load_model_and_weights(load_ckpt, load_ckpt_disp, FLAGS, use_cuda, ckpts_dir=None, dispnet='DispNetS',
                           posenet='PoseExpNet', train=True):
    # load models
    disp_net = Model.model_builder(dispnet, FLAGS)
    pose_exp_net = Model.model_builder(posenet, FLAGS)
    # if debug:
    #     print("\nDispNetS = ", disp_net)
    #     print("\nPoseExpNet = ", pose_exp_net)
    if use_cuda:
        disp_net = disp_net.cuda()
        pose_exp_net = pose_exp_net.cuda()
    models = [disp_net, pose_exp_net]
    model_names = ['dispnet', 'exp_pose']
    models_loaded = False

    # load weights
    pose_model_path, disp_model_path = '', ''
    if load_ckpt_disp != '':
        disp_model_path = load_ckpt_disp
    elif load_ckpt_disp == '' and not train:
        disp_model_path = os.path.join(ckpts_dir, 'dispnet_ckpt.pth.tar')
    if load_ckpt != '':
        pose_model_path = load_ckpt
    elif load_ckpt == '' and not train:
        pose_model_path = os.path.join(ckpts_dir, 'exp_pose_ckpt.pth.tar')
    n_iter, n_epoch, train_iter_disp, train_iter_poseexp = 0, 0, 0, 0
    if os.path.isfile(pose_model_path) and os.path.isfile(disp_model_path):
        pose_model_path = Path(pose_model_path)
        ckpt_dir = pose_model_path.dirname()
        pose_model_path_tmp = ckpt_dir / 'exp_pose_ckpt_tmp.pth.tar'
        disp_model_path_tmp = ckpt_dir / 'dispnet_ckpt_tmp.pth.tar'
        copyfile(pose_model_path, pose_model_path_tmp)
        copyfile(disp_model_path, disp_model_path_tmp)
        pose_ckpt_dict = torch.load(pose_model_path_tmp)
        if 'iteration' in pose_ckpt_dict:
            train_iter_poseexp = pose_ckpt_dict['iteration']
        if 'epoch' in pose_ckpt_dict:
            epoch = pose_ckpt_dict['epoch']
        assert isinstance(pose_ckpt_dict['state_dict'], (dict, OrderedDict)), type(pose_ckpt_dict['state_dict'])
        pose_exp_net.load_state_dict(pose_ckpt_dict['state_dict'], strict=False)
        # get seq_length from the loaded model
        seq_length = int(pose_ckpt_dict['state_dict']['conv1.0.weight'].size(1) / 3)
        assert seq_length == FLAGS.seq_length, 'seq_length in config file is different from the loaded model'

        disp_ckpt_dict = torch.load(disp_model_path_tmp)
        if 'iteration' in disp_ckpt_dict:
            train_iter_disp = disp_ckpt_dict['iteration']
        assert isinstance(disp_ckpt_dict['state_dict'], (dict, OrderedDict)), type(disp_ckpt_dict['state_dict'])
        disp_net.load_state_dict(disp_ckpt_dict['state_dict'], strict=False)
        # assert train_iter_poseexp == train_iter_disp, 'iterations in posenet and dispnet are different'
        models_loaded = True
        os.remove(pose_model_path_tmp)
        os.remove(disp_model_path_tmp)

        if train:
            print('PoseExpNet training resumed from the ckpt {} (epoch {}, iter {})'.format(pose_model_path, epoch, train_iter_poseexp))
            print('DispNet training resumed from the ckpt {} (epoch {}, iter {})'.format(disp_model_path, epoch, train_iter_disp))
        else:
            print('PoseExpNet ckpt loaded from {} (epoch {}, iter {})'.format(pose_model_path, epoch, train_iter_poseexp))
            print('DispNet ckpt loaded from {} (epoch {}, iter {})'.format(disp_model_path, epoch, train_iter_disp))

        if train_iter_poseexp > 0:
            if train:
                n_iter = train_iter_poseexp + 1
            else:
                n_iter = train_iter_poseexp
            n_epoch = epoch
    else:
        if train:
            pose_exp_net.init_weights()
            disp_net.init_weights()

    return models_loaded, models, model_names, n_iter, n_epoch


def save_train_losses_and_imgs_to_tensorboard_and_csv(var_dict_t, writer, losses_list, loss_names, disp, depth, pose,
            explainability_mask, n_iter, num_iters_for_print, loss_full_path, train_iters_dict, n_epoch, rotation_mode,
            debug):
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
                loss_val_list += [l.data[0] for l in losses_list]
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

        # add images to tensorboard
        if n_iter % (num_iters_for_print * 10) == 0:
            writer.add_image('trg_img_train', tensor2array(var_dict_t['tgt_img_l_cpu'][0]), n_iter)
            writer.add_image('ref_img_train', tensor2array(var_dict_t['ref_imgs_l_cpu'][0]), n_iter)

            for k, scaled_depth in enumerate(depth):
                if not debug and k == 0 and n_iter % (num_iters_for_print * 10) == 0:
                    writer.add_image('disp_s{}_train'.format(k),
                                          tensor2array(disp[k].data[0].cpu(), max_value=None, colormap='bone'),
                                          n_iter)
                    writer.add_image('depth_s{}_train'.format(k),
                                          tensor2array(1 / disp[k].data[0].cpu(), max_value=10),
                                          n_iter)
                b, _, h, w = scaled_depth.size()
                downscale = var_dict_t['tgt_img_l'].size(2) / h

                tgt_img_scaled = nn.functional.adaptive_avg_pool2d(var_dict_t['tgt_img_l'], (h, w))
                ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in var_dict_t['ref_imgs_l']]

                intrinsics_scaled = torch.cat((var_dict_t['intrinsics_l'][:, 0:2] / downscale, var_dict_t['intrinsics_l'][:, 2:]), dim=1)
                intrinsics_scaled_inv = torch.cat(
                    (var_dict_t['intrinsics_l_inv'][:, :, 0:2] * downscale, var_dict_t['intrinsics_l_inv'][:, :, 2:]), dim=2)

                # log warped images along with explainability mask
                for j, ref in enumerate(ref_imgs_scaled):
                    ref_warped = inverse_warp(ref, scaled_depth[:, 0], pose[:, j], intrinsics_scaled,
                                              intrinsics_scaled_inv, rotation_mode=rotation_mode)[0]
                    if not debug and k == 0 and j == 0 and n_iter % (num_iters_for_print * 10) == 0:
                        writer.add_image('ref_warped_s{}_train'.format(k),
                                              tensor2array(ref_warped.data.cpu()), n_iter)
                        writer.add_image('trg-ref_warped_s{}_train'.format(k), tensor2array(
                            0.5 * (tgt_img_scaled[0] - ref_warped).abs().data.cpu()), n_iter)
                        if explainability_mask[k] is not None:
                            writer.add_image('explain_mask_s{}_train'.format(k),
                                                  tensor2array(explainability_mask[k][0, j].data.cpu(),
                                                               max_value=1, colormap='bone'), n_iter)
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


def save_concat_img_results(var_dict_t, disp, depth, pose, explainability_mask, n_dof, rotation_mode,
                            visualization_test_dir, n_iter, filename_tgt_cut):
    images = [tensor2array(var_dict_t['tgt_img_l_cpu'][0])]
    images.append(tensor2array(var_dict_t['ref_imgs_l_cpu'][0]))
    images.append(tensor2array(disp.data[0].cpu(), max_value=None, colormap='bone'))
    images.append(normalize_depth_for_display(disp.data[0].cpu()))
    # images.append(tensor2array(1. / (disp.data[0].cpu()+1e-4), max_value=100)) #, colormap='plasma'))
    ref_warped = inverse_warp(var_dict_t['ref_imgs_l'][0][:1], depth[:1, 0], Variable(pose.data[0][0].view(-1, n_dof)),
                              var_dict_t['intrinsics_l'][:1], var_dict_t['intrinsics_l_inv'][:1],
                              rotation_mode=rotation_mode, detach=False)[0]
    images.append(tensor2array(ref_warped.data.cpu()))
    images.append(tensor2array(0.5 * (var_dict_t['tgt_img_l'][0] - ref_warped).abs().data.cpu()))
    if explainability_mask is not None:
        images.append(tensor2array(explainability_mask[0, 0].data.cpu(), max_value=1, colormap='bone'))
    img_names = ['trg', 'ref', 'disp', 'depth', 'ref_inv_warp', 'ref_warped', 'trg-ref_warped']
    save_path = os.path.join(visualization_test_dir, 'img_comb_{}_{}.jpg'.format(n_iter, filename_tgt_cut[0][0]))
    save_concat_imgs(images, img_names, save_path)


def auto_select_gpu(mem_bound=500, utility_bound=0, gpus=(0, 1, 2, 3, 4, 5, 6, 7), num_gpu=1, selected_gpus=None):
    import sys
    import os
    import subprocess
    import re
    import time
    import numpy as np
    if 'CUDA_VISIBLE_DEVCIES' in os.environ:
        sys.exit(0)
    if selected_gpus is None:
        mem_trace = []
        utility_trace = []
        for i in range(5): # sample 5 times
            info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
            mem = [int(s[:-5]) for s in re.compile('\d+MiB\s/').findall(info)]
            utility = [int(re.compile('\d+').findall(s)[0]) for s in re.compile('\d+%\s+Default').findall(info)]
            mem_trace.append(mem)
            utility_trace.append(utility)
            time.sleep(0.1)
        mem = np.mean(mem_trace, axis=0)
        utility = np.mean(utility_trace, axis=0)
        assert(len(mem) == len(utility))
        nGPU = len(utility)
        ideal_gpus = [i for i in range(nGPU) if mem[i] <= mem_bound and utility[i] <= utility_bound and i in gpus]

        if len(ideal_gpus) < num_gpu:
            print("No sufficient resource, available: {}, require {} gpu".format(ideal_gpus, num_gpu))
            sys.exit(0)
        else:
            selected_gpus = list(map(str, ideal_gpus[:num_gpu]))
    else:
        selected_gpus = selected_gpus.split(',')

    print("Setting GPU: {}".format(selected_gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected_gpus)
    return selected_gpus




if __name__ == '__main__':
    # folder_path = Path('/media/victoria/d/data/EuRoC_MAV/V1_01_easy')
    # filename2time_csv_path = folder_path/'cam0'/'data.csv'
    # pose_gt_csv_path = folder_path / 'state_groundtruth_estimate0/data.csv'
    # read_pose_csv(pose_gt_csv_path, filename2time_csv_path)

    # KITTY calibration file read
    calib_file_path = '/media/victoria/d/data/KITTI_raw/2011_09_26/calib_cam_to_cam.txt'
    height, width = 128, 416
    getKT(height, width, calib_file_path)


















