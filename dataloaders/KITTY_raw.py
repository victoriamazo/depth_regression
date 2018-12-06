import torch.utils.data as data
import numpy as np
from path import Path
import random
import os

import dataloaders.img_transforms as transforms
from utils.auxiliary import load_as_float, generate_mask, generate_depth_map, get_intrinsics_baseline, get_displacements

class KITTY_raw(data.Dataset):
    """
        Returns KITTY raw images depending on the input parameters.
        KITTY_raw is a dataset that includes serieses of stereo images with camera calibrations and depth.

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
                 - with_gt_depth: returnsalso depth GT
                 - hflip: if True, images are randomly flipped (if stereo, left and right images are interchanged)
                 - rand_crop: if True, image will be randomly cropped (up to 15%) (and resized)
                 - seed: random seed
        - mode: 'train', 'test' and 'eval'.

        Output:
            if stereo is False and seq_length = 1 every call returns a dictionary of variables:
                {tgt_img_l, intrinsics, intrinsics_inv, gt_depth_l, filename_tgt}
            if stereo is False and seq_length > 1 every call returns a dictionary of variables:
                {tgt_img_l, ref_imgs_l, intrinsics, intrinsics_inv, gt_depth_l, filename_tgt, filenames_ref}
            if stereo is True and seq_length = 1 every call returns a dictionary of variables:
                {tgt_img_l, tgt_img_r, intrinsics_l, intrinsics_r, intrinsics_inv_l, intrinsics_inv_r, gt_depth_l,
                gt_depth_r, filename_tgt, baseline}
            if stereo is True and seq_length > 1 every call returns a dictionary of variables:
                {tgt_img_l, ref_imgs_l, tgt_img_r, ref_imgs_r, intrinsics_l, intrinsics_r, intrinsics_inv_l,
                intrinsics_inv_r, gt_depth_l, gt_depth_r, filename_tgt, filenames_ref, baseline}

            - image is a PyTorch tensor in format (B, C, H, W)
            - ref_imgs_l, ref_imgs_r, filenames_ref and gt_ref_poses are lists (of length seq_lenth-1)
            - intrinsics is a PyTorch tensor (B, 3, 3)
            - filename is of shape (B, 1), it is a full path to an image (in format Path)
            - left (and right) depths are in the original sizes as tensor, fliiping is allowed, but
                cropping is not allowed if 'with_gt_depth'

    """
    def __init__(self, FLAGS, mode):
        super(KITTY_raw, self).__init__()
        seed = FLAGS.seed
        if mode == 'train':
            self.stereo = FLAGS.stereo
        else:
            self.stereo = FLAGS.stereo_test
        self.seq_length = FLAGS.seq_length
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.with_gt_depth = FLAGS.with_gt_depth
        self.min_depth, self.max_depth = 1e-3, 100
        if hasattr(FLAGS, 'min_depth'):
            self.min_depth = FLAGS.min_depth
        if hasattr(FLAGS, 'max_depth'):
            self.max_depth = FLAGS.max_depth
        if hasattr(FLAGS, 'skip'):
            self.skip = FLAGS.skip
        else:
            self.skip = 1
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(FLAGS.data_dir)
        self.euler_angles = FLAGS.euler_angles
        self.scenes = []

        # transforms (no cropping and resize if with_gt_depth)
        normalize = transforms.NormalizeStereo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = []
        if FLAGS.hflip:
            transform.append(transforms.RandomHorizontalFlipStereo())
        if FLAGS.rand_crop and not self.with_gt_depth:
            transform.append(transforms.RandomScaleCropResizeStereo())
        else:
            transform.append(transforms.ResizeStereo())
        transform.append(transforms.ArrayToTensorStereo())
        transform.append(normalize)
        self.transform = transforms.Compose(transform)
        if self.with_gt_depth:
            self.depth_transform = transforms.Compose([transforms.DepthFlipToTensorStereo()])

        assert os.path.isfile(self.root / 'test_files_eigen.txt')
        with open(self.root / 'test_files_eigen.txt', 'r') as f:
            self.test_files_list = list(f.read().splitlines())
        if mode == 'train':
            self.samples = self._crawl_folders()
            # with open(self.root/'train_files_eigen.txt', 'r') as f:
            #     files_list = list(f.read().splitlines())
        else:
            self.samples = self._read_scene_data()


    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img_l = load_as_float(sample['tgt_img_l'])
        ref_imgs_l = [load_as_float(img) for img in sample['ref_imgs_l']]
        tgt_img_r = load_as_float(sample['tgt_img_r'])
        ref_imgs_r = [load_as_float(img) for img in sample['ref_imgs_r']]
        intrinsics_l = np.copy(sample['intrinsics_l'])
        intrinsics_r = np.copy(sample['intrinsics_r'])
        filename_tgt = sample['tgt_img_l']
        if self.with_gt_depth:
            gt_depth_l = generate_depth_map(sample['calib_dir'], sample['gt_trg_depth'], tgt_img_l.shape[:2], cam=2)
            print('gt_depth_l nonzero = ', np.count_nonzero(gt_depth_l), 'gt_depth_l nonzero = ', np.count_nonzero(gt_depth_l==0))
            if self.stereo:
                gt_depth_r = generate_depth_map(sample['calib_dir'], sample['gt_trg_depth'], tgt_img_l.shape[:2], cam=3)
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
        flip = False
        args = [intrinsics, self.stereo, self.height, self.width, flip]
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
            gt_depth_l = depth_list_out[0][:370, :1224]
            if self.stereo:
                gt_depth_r = depth_list_out[1][:370, :1224]

        var_dict_np = {'tgt_img_l': tgt_img_l, 'intrinsics_l': intrinsics_l,
                       'intrinsics_inv_l': np.linalg.inv(intrinsics_l), 'filename_tgt': filename_tgt}
        if self.with_gt_depth:
            var_dict_np['gt_depth_l'] = gt_depth_l
        if self.seq_length > 1:
            var_dict_np['ref_imgs_l'] = images_out[1:1+num_ref_img]
            var_dict_np['filename_ref'] = sample['ref_imgs_l']
        if self.stereo:
            var_dict_np['intrinsics_r'] = intrinsics_r
            var_dict_np['intrinsics_inv_r'] = np.linalg.inv(intrinsics_r)
            var_dict_np['baseline'] = baseline
            if self.with_gt_depth:
                var_dict_np['gt_depth_r'] = gt_depth_r
            if self.seq_length == 1:
                var_dict_np['tgt_img_r'] = images_out[1]
            else:
                var_dict_np['tgt_img_r'] = images_out[1+num_ref_img]
                var_dict_np['ref_imgs_r'] = images_out[2+num_ref_img:2+2*num_ref_img]

        return var_dict_np


    def __len__(self):
        return len(self.samples)


    def _read_scene_data(self):
        '''Getting samples for Eigen split (test files)'''
        data_root = Path(self.root)
        sequence_set = []
        demi_length = (self.seq_length - 1) // 2
        demi_length_min, demi_length_max = demi_length, demi_length
        if self.seq_length % 2 == 0:
            demi_length_max = demi_length + 1
        shift_range = [self.skip * i for i in (list(range(-demi_length_min, 0)) + list(range(1, demi_length_max + 1)))]

        print('Eigen split test files_list length: ', len(self.test_files_list))
        for i, file in enumerate(self.test_files_list):
            tgt_img_l_path = data_root / file
            date, scene, cam_id, _, index = file[:-4].split('/')
            if int(index) >= self.skip * demi_length_min:
                ref_imgs_l_path = [tgt_img_l_path.dirname() / '{:010d}.png'.format(int(index) + shift) for shift in shift_range]
                # check if all ref_imgs exist
                ref_imgs_exist = [ref_img_l_path.isfile() for ref_img_l_path in ref_imgs_l_path]
                tgt_img_r_path = data_root / date / scene / 'image_03/data/{}.png'.format(index[:10])
                ref_imgs_r_path = [tgt_img_r_path.dirname() / '{:010d}.png'.format(int(index) + shift) for shift in shift_range]
                vel_path = data_root / date / scene / 'velodyne_points' / 'data' / '{}.bin'.format(index[:10])
                calib_dir = data_root / date

                # ensures ref_imgs are present, if not, set shift to 0 so that it will be discarded later
                caped_shift_range = shift_range[:]
                for i, img in enumerate(ref_imgs_l_path):
                    if not img.isfile():
                        ref_imgs_l_path[i] = tgt_img_l_path
                        caped_shift_range[i] = 0

                if tgt_img_l_path.isfile() and sum(ref_imgs_exist) == len(ref_imgs_l_path):
                    # displacement = self.get_displacements(data_root / date / scene / 'oxts', int(index), caped_shift_range)
                    intrinsics_l, intrinsics_r, baseline = get_intrinsics_baseline(calib_dir)
                    sample = {'tgt_img_l': tgt_img_l_path, 'ref_imgs_l': ref_imgs_l_path, 'gt_trg_depth': vel_path,
                              'intrinsics_l': intrinsics_l, 'intrinsics_r': intrinsics_r, 'tgt_img_r': tgt_img_r_path,
                              'ref_imgs_r': ref_imgs_r_path, 'baseline': baseline, 'calib_dir': calib_dir}
                    sequence_set.append(sample)
                else:
                    print('{} missing'.format(tgt_img_l_path))

        return sequence_set


    def _crawl_folders(self):
        data_root = Path(self.root)
        sequence_set = []
        demi_length = (self.seq_length - 1) // 2
        demi_length_min, demi_length_max = demi_length, demi_length
        if self.seq_length % 2 == 0:
            demi_length_min = demi_length + 1
        shift_range = [self.skip * i for i in (list(range(-demi_length_min, 0)) + list(range(1, demi_length_max + 1)))]

        for date in os.listdir(data_root):
            if os.path.isdir(data_root / date):
                print('date: ', date, ' N scenes: ', len(os.listdir(data_root / date)))
                calib_dir = data_root / date
                intrinsics_l, intrinsics_r, baseline = get_intrinsics_baseline(calib_dir)
                for scene in os.listdir(data_root / date):
                    if os.path.isdir(data_root / date / scene):
                        self.scenes.append(scene)
                        img_dir = date + '/' + scene + '/image_02/data'
                        file_list = os.listdir(data_root / img_dir)
                        file_list_sort = sorted(file_list)
                        for file in file_list_sort:
                            file_path_rel = img_dir + '/' + file
                            index, ext = file.split('.')
                            if ext == 'png' and file_path_rel not in self.test_files_list:
                                tgt_img_l_path = data_root / file_path_rel
                                if int(index) >= self.skip * demi_length_min and \
                                                int(index) < (len(file_list_sort) - self.skip * demi_length_max):
                                    ref_imgs_l_path = [
                                        tgt_img_l_path.dirname() / '{:010d}.png'.format(int(index) + shift) for shift
                                        in shift_range]
                                    tgt_img_r_path = data_root / date / scene / 'image_03/data/{}.png'.format(index[:10])
                                    ref_imgs_r_path = [
                                        tgt_img_r_path.dirname() / '{:010d}.png'.format(int(index) + shift) for
                                        shift in shift_range]
                                    vel_path = data_root / date / scene / 'velodyne_points' / 'data' / '{}.bin'.format(
                                        index[:10])

                                    # ensures ref_imgs are present, if not, set shift to 0 so that it will be discarded later
                                    caped_shift_range = shift_range[:]
                                    for i, img in enumerate(ref_imgs_l_path):
                                        if not img.isfile():
                                            ref_imgs_l_path[i] = tgt_img_l_path
                                            caped_shift_range[i] = 0

                                    if tgt_img_l_path.isfile() and vel_path.isfile():
                                    #     displacement = self.get_displacements(data_root / date / scene / 'oxts',
                                    #                                           int(index), caped_shift_range)
                                        sample = {'tgt_img_l': tgt_img_l_path, 'ref_imgs_l': ref_imgs_l_path,
                                                  'gt_trg_depth': vel_path, 'intrinsics_l': intrinsics_l,
                                                  'intrinsics_r': intrinsics_r, 'tgt_img_r': tgt_img_r_path,
                                                  'ref_imgs_r': ref_imgs_r_path, 'baseline': baseline,
                                                  'calib_dir': calib_dir}
                                        sequence_set.append(sample)
        return sequence_set








if __name__ == '__main__':
    import multiprocessing
    class FLAGS():
        def __init__(self):
            self.batch_size = 4
            self.data_dir = '/media/victoria/d/data/KITTI_raw'
            self.height = 128
            self.width = 416
            self.seed = 127
            self.stereo = 1
            self.stereo_test = 1
            self.seq_length = 2
            self.euler_angles = True
            self.with_gt_depth = 1
            self.hflip = 1
            self.rand_crop = 1
            self.shuffle = True
    FLAGS = FLAGS()
    mode = 'train'
    train_set = KITTY_raw(FLAGS, mode)
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('stereo: ', bool(FLAGS.stereo), ", seq_length = ", FLAGS.seq_length, 'with_gt_depth:', bool(FLAGS.with_gt_depth))

    train_loader = data.DataLoader(train_set, batch_size=FLAGS.batch_size, shuffle=FLAGS.shuffle,
        num_workers=multiprocessing.cpu_count()-2, pin_memory=True)
    epoch_size = len(train_loader)
    print('epoch_size = ', epoch_size)

    if not FLAGS.stereo and FLAGS.seq_length == 1:
        for i, var_dict_np in enumerate(train_loader):
            print('\n', i, 'tgt_img_l = ', var_dict_np['tgt_img_l'].shape)  # tgt_img is a PyTorch tensor
            print('intrinsics_l = ', var_dict_np['intrinsics_l'])
            print('filename_tgt = ', var_dict_np['filename_tgt'])
            if FLAGS.with_gt_depth:
                print('gt_depth_l = ', var_dict_np['gt_depth_l'][0])
    elif not FLAGS.stereo and FLAGS.seq_length > 1:
        for i, var_dict_np in enumerate(train_loader):
            print('\n', i, 'tgt_img_l = ', var_dict_np['tgt_img_l'].shape)
            print('ref_imgs_l = ', len(var_dict_np['ref_imgs_l']), var_dict_np['ref_imgs_l'][0].shape)
            print('intrinsics_l = ', var_dict_np['intrinsics_l'])
            print('filename_tgt = ', var_dict_np['filename_tgt'])
            print('filename_ref = ', var_dict_np['filename_ref'])
            if FLAGS.with_gt_depth:
                print('gt_depth_l = ', var_dict_np['gt_depth_l'][0])
    elif FLAGS.stereo and FLAGS.seq_length == 1:
        for i, var_dict_np in enumerate(train_loader):
            print('\n', i, 'tgt_img_l = ', var_dict_np['tgt_img_l'].shape)
            print('tgt_img_r = ', var_dict_np['tgt_img_r'].shape)
            print('intrinsics_l = ', var_dict_np['intrinsics_l'])
            print('intrinsics_r = ', var_dict_np['intrinsics_r'])
            print('baseline = ', var_dict_np['baseline'])
            print('filename_tgt = ', var_dict_np['filename_tgt'])
            if FLAGS.with_gt_depth:
                print('gt_depth_l = ', var_dict_np['gt_depth_l'][0])
                print('gt_depth_r = ', var_dict_np['gt_depth_r'][0])
    elif FLAGS.stereo and FLAGS.seq_length > 1:
        for i, var_dict_np in enumerate(train_loader):
            print('\n', i, 'tgt_img_l = ', var_dict_np['tgt_img_l'].shape)
            print('ref_imgs_l = ', len(var_dict_np['ref_imgs_l']), var_dict_np['ref_imgs_l'][0].shape)
            print('tgt_img_r = ', var_dict_np['tgt_img_r'].shape)
            print('ref_imgs_r = ', len(var_dict_np['ref_imgs_r']), var_dict_np['ref_imgs_r'][0].shape)
            if FLAGS.with_gt_depth:
                print('gt_depth_l = ', var_dict_np['gt_depth_l'][0])
                print('gt_depth_r = ', var_dict_np['gt_depth_r'][0])
            print('intrinsics_l = ', var_dict_np['intrinsics_l'])
            print('intrinsics_r = ', var_dict_np['intrinsics_r'])
            print('baseline = ', var_dict_np['baseline'])
            print('filename_tgt = ', var_dict_np['filename_tgt'])
            print('filename_ref = ', var_dict_np['filename_ref'])


































