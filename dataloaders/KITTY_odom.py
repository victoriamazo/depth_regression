import torch.utils.data as data
import numpy as np
from path import Path
import random
import os
from PIL import Image

import dataloaders.img_transforms as transforms
from utils.auxiliary import load_as_float, read_calib_file, intrinsics_matrix, generate_depth_map, \
    read_KITTY_poses

import torch


class KITTY_odom(data.Dataset):
    """
        Returns KITTY odometry images depending on the input parameters.
        KITTY_odom is a dataset that includes serieses of stereo images with GT pose and GT depth
        (incl. camera and velodyne calibrations).

        Input:
         - FLAGS: class that includes the following attributes:
                 - data_dir: root directory of KITTY_odom
                 - stereo: True or False
                 - sequence length:
                        - seq_length=1 returns only target image i
                        - seq_length=2 returns target image i and reference image (i-1)*skip (list of one))
                        - seq_length>=3 (odd) returns target image and list of reference images, e.g. for
                            seq_length=3, tgt_img i, ref_imgs [(i-1)*skip, (i+1)*skip])
                        - seq_length>=4 (even) returns target image i and list of reference images, e.g. for
                            seq_length=4, tgt_img i, ref_imgs [(i-2)*skip, (i-1)*skip, (i+1)*skip])
                 - height: resize height (if smaller than input height, image height stays the same)
                 - width: resize width (if smaller than input width, image width stays the same)
                 - with_gt_pose: returns only samples with pose GT
                 - with_gt_depth: returnsalso depth GT
                 - hflip: if True, images are randomly flipped (if stereo, left and right images are interchanged)
                 - rand_crop: if True, image will be randomly cropped (up to 15%) (and resized)
                 - seed: random seed
        - mode: 'train', 'test' and 'eval'.

        Output:
            if stereo is False and seq_length = 1 every call returns a dictionary of variables:
                {tgt_img_l, intrinsics, intrinsics_inv, gt_trg_pose, filename_tgt}
            if stereo is False and seq_length > 1 every call returns a dictionary of variables:
                {tgt_img_l, ref_imgs_l, intrinsics, intrinsics_inv, gt_trg_pose, gt_ref_poses, filename_tgt, filenames_ref}
            if stereo is True and seq_length = 1 every call returns a dictionary of variables:
                {tgt_img_l, tgt_img_r, intrinsics_l, intrinsics_r, intrinsics_inv_l, intrinsics_inv_r, gt_trg_pose,
                filename_tgt, baseline}
            if stereo is True and seq_length > 1 every call returns a dictionary of variables:
                {tgt_img_l, ref_imgs_l, tgt_img_r, ref_imgs_r, intrinsics_l, intrinsics_r, intrinsics_inv_l,
                intrinsics_inv_r, gt_trg_pose, gt_ref_poses, filename_tgt, filenames_ref, baseline}

            - image is a PyTorch tensor in format (B, C, H, W)
            - ref_imgs_l, ref_imgs_r, filenames_ref and ref_poses are lists (of length seq_lenth-1)
            - intrinsics is a PyTorch tensor (B, 3, 3)
            - pose is a PyTorch tensor of shape (B, 4, 4) (in the matrix form)
            - filename is of shape (B, 1), it is a full path to an image (in format Path)

    """
    def __init__(self, FLAGS, mode):
        super(KITTY_odom, self).__init__()
        seed = FLAGS.seed
        if mode == 'train':
            self.stereo = FLAGS.stereo
        else:
            self.stereo = FLAGS.stereo_test
        self.seq_length = FLAGS.seq_length
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.with_gt_pose, self.with_gt_depth = False, False
        if hasattr(FLAGS, 'with_gt'):
            self.with_gt_pose = FLAGS.with_gt
        if hasattr(FLAGS, 'with_gt_pose'):
            self.with_gt_pose = FLAGS.with_gt_pose
        if hasattr(FLAGS, 'with_gt_pose'):
            self.with_gt_depth = FLAGS.with_gt_depth
        if hasattr(FLAGS, 'skip'):
            self.skip = FLAGS.skip
        else:
            self.skip = 1
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(FLAGS.data_dir)
        self.euler_angles = FLAGS.euler_angles
        train, eval = False, False
        if mode == 'train':
            train = True
        if mode == 'eval':
            eval = True
        scene_list_path = self.root/'train.txt' if train else self.root/'test.txt'
        if train and self.with_gt_pose:
            scene_list_path = self.root / 'train_with_gt.txt'
        if eval:
            scene_list_path = self.root/'eval.txt'
        self.sequences = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.vel_root = self.root.dirname().dirname()/'data_odometry_velodyne/sequences'
        self.depth_w, self.depth_h = 1240, 376

        # transforms
        normalize = transforms.NormalizeStereo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = []
        if FLAGS.hflip:
            transform.append(transforms.RandomHorizontalFlipStereo())
        if FLAGS.rand_crop and not (self.with_gt_depth and mode == 'test'):
            transform.append(transforms.RandomScaleCropResizeStereo())
        else:
            if FLAGS.rand_crop:
                print('no random crop will be performed, since depth GT is used in test mode')
            transform.append(transforms.ResizeStereo())
        transform.append(transforms.ArrayToTensorStereo())
        transform.append(normalize)
        self.transform = transforms.Compose(transform)
        if self.with_gt_depth:
            depth_transform = []
            if mode == 'train':
                depth_transform.append(transforms.DepthRandomScaleCropResizeStereo())
            depth_transform.append(transforms.DepthFlipToTensorStereo())
            self.depth_transform = transforms.Compose(depth_transform)

        self.samples = self._crawl_folders()


    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img_l = load_as_float(sample['tgt_img_l'])
        ref_imgs_l = [load_as_float(img) for img in sample['ref_imgs_l']]
        tgt_img_r = load_as_float(sample['tgt_img_r'])
        ref_imgs_r = [load_as_float(img) for img in sample['ref_imgs_r']]
        intrinsics_l = np.copy(sample['intrinsics_l'])
        intrinsics_r = np.copy(sample['intrinsics_r'])
        filename_tgt = sample['tgt_img_l']
        # loading dense depth
        if self.with_gt_depth:
            gt_depth_l_im = Image.open(sample['gt_trg_depth'])
            gt_depth_l = np.array(gt_depth_l_im)/255.
            # Image.fromarray(gt_depth_l).show()
            if self.stereo:
                gt_depth_r = np.zeros_like(gt_depth_l)
            ## loading sparse depth
            # # filling with zeros absent pixels in depth map (left corner)
            # gt_depth_l_tmp = generate_depth_map(sample['calib_dir'], sample['gt_trg_depth'], tgt_img_l.shape[:2], cam=2,
            #                                 odom=True)[:self.depth_h, :self.depth_w]
            # gt_depth_l = np.zeros((self.depth_h, self.depth_w))
            # h_min = self.depth_h - gt_depth_l_tmp.shape[0]
            # w_min = self.depth_w - gt_depth_l_tmp.shape[1]
            # gt_depth_l[h_min:, w_min:] = gt_depth_l_tmp
            # if self.stereo:
            #     gt_depth_r_tmp = generate_depth_map(sample['calib_dir'], sample['gt_trg_depth'], tgt_img_l.shape[:2], cam=3,
            #                                     odom=True)[:self.depth_h, :self.depth_w]
            #     gt_depth_r = np.zeros((self.depth_h, self.depth_w))
            #     h_min = self.depth_h - gt_depth_r_tmp.shape[0]
            #     w_min = self.depth_w - gt_depth_r_tmp.shape[1]
            #     gt_depth_r[h_min:, w_min:] = gt_depth_r_tmp
        baseline = sample['baseline']
        images = [tgt_img_l]
        intrinsics = [intrinsics_l]
        num_ref_img = 0
        if self.seq_length > 1 and len(ref_imgs_l) > 0:
            num_ref_img = len(ref_imgs_l)
            for img in ref_imgs_l:
                images.append(img)
        if self.stereo and tgt_img_r is not None:
            images.append(tgt_img_r)
            intrinsics.append(intrinsics_r)
        if self.seq_length > 1 and len(ref_imgs_r) > 0:
            for img in ref_imgs_r:
                images.append(img)

        # transform
        args = [intrinsics, self.stereo, self.height, self.width, [0]*9]
        images_out, args_out = self.transform(images, args)
        intrinsics_l = args_out[0][0]
        if self.stereo and tgt_img_r is not None:
            intrinsics_r = args_out[0][1]
        tgt_img_l = images_out[0]

        # depth transforms (returns depth in the original size, since resize reduces quality)
        if self.with_gt_depth:
            depth_list = [gt_depth_l]
            if self.stereo:
                depth_list.append(gt_depth_r)
            depth_list_out, args_out = self.depth_transform(depth_list, args_out)
            gt_depth_l = depth_list_out[0] #[:370, :1224]
            if self.stereo:
                gt_depth_r = depth_list_out[1]  #[:370, :1224]

        var_dict_np = {'tgt_img_l': tgt_img_l, 'intrinsics_l': intrinsics_l,
                       'intrinsics_inv_l': np.linalg.inv(intrinsics_l), 'filename_tgt': filename_tgt}
        if self.with_gt_pose:
            var_dict_np['gt_trg_pose'] = sample['gt_trg_pose']
        if self.with_gt_depth:
            var_dict_np['gt_depth_l'] = gt_depth_l
        if self.seq_length > 1:
            var_dict_np['ref_imgs_l'] = images_out[1:1 + num_ref_img]
            var_dict_np['filename_ref'] = sample['ref_imgs_l']
            if self.with_gt_pose:
                var_dict_np['gt_ref_poses'] = sample['gt_ref_poses']
        if self.stereo:
            var_dict_np['intrinsics_r'] = intrinsics_r
            var_dict_np['intrinsics_inv_r'] = np.linalg.inv(intrinsics_r)
            var_dict_np['baseline'] = baseline
            if self.with_gt_depth:
                var_dict_np['gt_depth_r'] = gt_depth_r
            if self.seq_length == 1:
                var_dict_np['tgt_img_r'] = images_out[1]
            else:
                var_dict_np['tgt_img_r'] = images_out[1 + num_ref_img]
                var_dict_np['ref_imgs_r'] = images_out[2 + num_ref_img:2 + 2 * num_ref_img]

        return var_dict_np


    def __len__(self):
        return len(self.samples)


    def _crawl_folders(self):
        sequence_set = []
        demi_length = (self.seq_length - 1) // 2
        demi_length_min, demi_length_max = demi_length, demi_length
        if self.seq_length % 2 == 0:
            demi_length_min = demi_length + 1
        shift_range = [self.skip * i for i in (list(range(-demi_length_min, 0)) + list(range(1, demi_length_max + 1)))]

        for sequence_path in self.sequences:
            intrinsics_path = sequence_path / 'calib.txt'
            intrinsics_np, baseline_l = read_calib_file(intrinsics_path, is_left=True)
            intrinsics_l = intrinsics_matrix(intrinsics_np, is_kitty=True)
            intrinsics_np, baseline_r = read_calib_file(intrinsics_path, is_left=False)
            intrinsics_r = intrinsics_matrix(intrinsics_np, is_kitty=True)
            baseline = baseline_r - baseline_l
            sequence = sequence_path.split('/')[-1]
            if self.with_gt_pose:
                pose_gt_path = self.root.dirname()/'poses/{}.txt'.format(sequence)
                gt_poses = read_KITTY_poses(pose_gt_path)   # absolute GT poses (matrix)
            img_dir_path = sequence_path / 'image_2'
            file_list = os.listdir(img_dir_path)
            file_list_sort = sorted(file_list)
            for file in file_list_sort:
                index, ext = file.split('.')
                index = int(index)
                tgt_img_l_path = img_dir_path / file
                if index >= self.skip*demi_length_min and index < (len(file_list_sort) - self.skip*demi_length_max):
                    ref_imgs_l_path = [
                        tgt_img_l_path.dirname() / '{:06d}.png'.format(index + shift) for shift
                        in shift_range]
                    tgt_img_r_path = sequence_path / 'image_3/{:06d}.png'.format(index)
                    ref_imgs_r_path = [
                        tgt_img_r_path.dirname() / '{:06d}.png'.format(index + shift) for
                        shift in shift_range]
                    # vel_path = self.vel_root / sequence / 'velodyne' / '{:06d}.bin'.format(index)
                    # vel_path = self.vel_root / sequence / 'velodyne_png' / '{:06d}.png'.format(index)
                    vel_path = self.vel_root / sequence / 'velodyne_png_sparseconv' / '{:06d}.png'.format(index)
                    gt_trg_pose, gt_ref_poses = [], []
                    if self.with_gt_pose:
                        gt_trg_pose = gt_poses[index]
                        gt_ref_poses = np.array([gt_poses[index + shift] for shift in shift_range])

                    if tgt_img_l_path.isfile():
                        sample = {'tgt_img_l': tgt_img_l_path, 'ref_imgs_l': ref_imgs_l_path,
                                  'gt_trg_depth': vel_path, 'calib_dir': self.vel_root/sequence,
                                  'intrinsics_l': intrinsics_l, 'intrinsics_r': intrinsics_r,
                                  'tgt_img_r': tgt_img_r_path, 'ref_imgs_r': ref_imgs_r_path, 'baseline': baseline,
                                  'gt_trg_pose': gt_trg_pose, 'gt_ref_poses': gt_ref_poses}
                        sequence_set.append(sample)
        return sequence_set







if __name__ == '__main__':
    import multiprocessing
    class FLAGS():
        def __init__(self):
            self.batch_size = 4
            self.data_dir = '/media/victoria/d/data/KITTI_odom/data_odometry_color/sequences'
            self.height = 128
            self.width = 416
            self.euler_angles = True
            self.seed = 127
            self.stereo = 1
            self.stereo_test = 1
            self.seq_length = 3
            self.with_gt_pose = 0
            self.with_gt_depth = 1
            self.hflip = 1
            self.rand_crop = 1
            self.shuffle = 1
    FLAGS = FLAGS()
    mode = 'train'
    train_set = KITTY_odom(FLAGS, mode)
    print('{} samples found in {} {} sequences'.format(len(train_set), mode, len(train_set.sequences)))
    print('stereo: ', bool(FLAGS.stereo), ", seq_length = ", FLAGS.seq_length, ', batch_size = ', FLAGS.batch_size,
          ', with_gt_pose:', bool(FLAGS.with_gt_pose), ', with_gt_depth:', bool(FLAGS.with_gt_depth)
          , ', hflip:', bool(FLAGS.hflip), ', rand_crop:', bool(FLAGS.rand_crop)
          , ', shuffle:', bool(FLAGS.shuffle))

    train_loader = data.DataLoader(train_set, batch_size=FLAGS.batch_size, shuffle=FLAGS.shuffle,
                                   num_workers=int(multiprocessing.cpu_count() / 4), pin_memory=True, drop_last=True)
    epoch_size = len(train_loader)
    print('epoch_size = ', epoch_size)

    if not FLAGS.stereo and FLAGS.seq_length == 1:
        for i, var_dict_np in enumerate(train_loader):
            print('\n', i, ' filename_tgt = ', var_dict_np['filename_tgt'])
            print('tgt_img_l = ', var_dict_np['tgt_img_l'].shape)  # tgt_img is a PyTorch tensor
            print('intrinsics_l = ', var_dict_np['intrinsics_l'])
            print('filename_tgt = ', var_dict_np['filename_tgt'])
            if FLAGS.with_gt_pose:
                print('gt_trg_pose = ', var_dict_np['gt_trg_pose'])
            if FLAGS.with_gt_depth:
                print('gt_depth_l = ', var_dict_np['gt_depth_l'][0])
    elif not FLAGS.stereo and FLAGS.seq_length > 1:
        for i, var_dict_np in enumerate(train_loader):
            print('\n', i, ' filename_tgt = ', var_dict_np['filename_tgt'])
            print('filename_ref = ', var_dict_np['filename_ref'])
            print('mean = ', torch.mean(var_dict_np['tgt_img_l']), torch.std(var_dict_np['tgt_img_l']),
                  torch.min(var_dict_np['tgt_img_l']), torch.max(var_dict_np['tgt_img_l']))
            print('mean = ', torch.mean(var_dict_np['ref_imgs_l'][0]), torch.std(var_dict_np['ref_imgs_l'][0]),
                  torch.min(var_dict_np['ref_imgs_l'][0]), torch.max(var_dict_np['ref_imgs_l'][0]))
            print('tgt_img_l = ', var_dict_np['tgt_img_l'].shape)
            print('ref_imgs_l = ', len(var_dict_np['ref_imgs_l']), var_dict_np['ref_imgs_l'][0].shape)
            print('intrinsics_l = ', var_dict_np['intrinsics_l'])
            if FLAGS.with_gt_pose:
                print('gt_trg_pose = ', var_dict_np['gt_trg_pose'])
                print('gt_ref_poses = ', var_dict_np['gt_ref_poses'])
            if FLAGS.with_gt_depth:
                print('gt_depth_l = ', var_dict_np['gt_depth_l'][0])
    elif FLAGS.stereo and FLAGS.seq_length == 1:
        for i, var_dict_np in enumerate(train_loader):
            print('\n', i, ' filename_tgt = ', var_dict_np['filename_tgt'])
            print('tgt_img_l = ', var_dict_np['tgt_img_l'].shape)
            print('tgt_img_r = ', var_dict_np['tgt_img_r'].shape)
            print('intrinsics_l = ', var_dict_np['intrinsics_l'])
            print('intrinsics_r = ', var_dict_np['intrinsics_r'])
            print('baseline = ', var_dict_np['baseline'])
            if FLAGS.with_gt_pose:
                print('gt_trg_pose = ', var_dict_np['gt_trg_pose'])
            if FLAGS.with_gt_depth:
                print('gt_depth_l = ', len(var_dict_np['gt_depth_l']), var_dict_np['gt_depth_l'][0].shape)
                print('gt_depth_l = ', var_dict_np['gt_depth_l'][0])
    elif FLAGS.stereo and FLAGS.seq_length > 1:
        for i, var_dict_np in enumerate(train_loader):
            print('\n', i, ' filename_tgt = ', var_dict_np['filename_tgt'])
            print('filename_ref = ', var_dict_np['filename_ref'])
            print('tgt_img_l = ', var_dict_np['tgt_img_l'].shape)
            print('mean = ', torch.mean(var_dict_np['tgt_img_l']), torch.std(var_dict_np['tgt_img_l']),
                  torch.min(var_dict_np['tgt_img_l']), torch.max(var_dict_np['tgt_img_l']))
            print('ref_imgs_l = ', len(var_dict_np['ref_imgs_l']), var_dict_np['ref_imgs_l'][0].shape)
            print('mean = ', torch.mean(var_dict_np['ref_imgs_l'][0]), torch.std(var_dict_np['ref_imgs_l'][0]),
                  torch.min(var_dict_np['ref_imgs_l'][0]), torch.max(var_dict_np['ref_imgs_l'][0]))
            print('tgt_img_r = ', var_dict_np['tgt_img_r'].shape)
            print('ref_imgs_r = ', len(var_dict_np['ref_imgs_r']), var_dict_np['ref_imgs_r'][0].shape)
            print('intrinsics_l = ', var_dict_np['intrinsics_l'])
            print('intrinsics_r = ', var_dict_np['intrinsics_r'])
            print('baseline = ', var_dict_np['baseline'])
            if FLAGS.with_gt_pose:
                print('gt_trg_pose = ', var_dict_np['gt_trg_pose'])
                print('gt_ref_poses = ', var_dict_np['gt_ref_poses'])
            if FLAGS.with_gt_depth:
                print('gt_depth_l = ', var_dict_np['gt_depth_l'][0])
                print('gt_depth_r = ', var_dict_np['gt_depth_r'][0])


































