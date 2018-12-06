import torch.utils.data as data
import numpy as np
from path import Path
import random
import yaml
from PIL import Image

import dataloaders.img_transforms as transforms
from utils.auxiliary import load_as_float, intrinsics_matrix, read_pose_csv

class EuRoc(data.Dataset):
    # TODO: to convert ref_imgs_l, ref_imgs_r, filenames_ref and ref_poses to lists
    """
        Returns EuRoc_MAV images depending on the input parameters.
        EuRoc_MAV is a dataset that includes serieses of stereo images with GT pose,
        camera calibrations and IMU measurements (angular velicities and accelerations).
        No baseline is found in the camera data (baseline = -1 and it is returned to be
        consistent with the KITTY odometru dataset).

        Input:
         - FLAGS: class that includes the following attributes:
                 - data_dir: root directory of EuRoc_MAV
                 - stereo: True or False
                 - seq_length: 1 or 2 (returns either only target image i, or, if 2, return target image i and
                 reference image i+skip)
                 - height: resize height (if smaller than input height, image height stays the same)
                 - width: resize width (if smaller than input width, image width stays the same)
                 - with_gt: only samples with GT
                 - hflip: if True, images are randomly flipped (if stereo, left and right images are interchanged)
                 - rand_crop: if True, image will be randomly cropped (up to 15%) (and resized)
                 - seed: random seed
        - mode: 'train', 'test' and 'eval'.

        Output:
            if stereo is False and seq_length = 1 every call returns tuple:
                (tgt_img, intrinsics, intrinsics_inv, trg_pose, filename_tgt)
            if stereo is False and seq_length = 2 every call returns tuple:
                (tgt_img, ref_imgs, intrinsics, intrinsics_inv, trg_pose, ref_pose, filename_tgt, filename_ref)
            if stereo is True and seq_length = 1 every call returns tuple:
                (tgt_img_l, tgt_img_r, intrinsics_l, intrinsics_r, intrinsics_inv_l, intrinsics_inv_r, trg_pose,
                filename_tgt, baseline)
            if stereo is True and seq_length = 2 every call returns tuple:
                (tgt_img_l, ref_img_l, tgt_img_r, ref_imgs_r, intrinsics_l, intrinsics_r, intrinsics_inv_l,
                intrinsics_inv_r, trg_pose, ref_pose, filename_tgt, filename_ref, baseline)

            - image is a PyTorch tensor in format (B, C, H, W)
            - intrinsics is a PyTorch tensor (B, 3, 3)
            - pose is a PyTorch tensor of shape (B, 6), if euler_angles is True 6dof is in format (tx, ty, tz, rx, ry, rz),
                    otherwise in format (tx, ty, tz, qx, qy, qz, qw)
            - filename is of shape (B, 1), it is a full path to an image (in format Path)

    """
    def __init__(self, FLAGS, mode):
        super(EuRoc, self).__init__()
        seed = FLAGS.seed
        self.stereo = FLAGS.stereo
        self.seq_length = FLAGS.seq_length
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.with_gt = FLAGS.with_gt
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
        if eval:
            scene_list_path = self.root/'eval.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]


        # transforms
        normalize = transforms.NormalizeStereo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = []
        if FLAGS.hflip:
            transform.append(transforms.RandomHorizontalFlipStereo())
        if FLAGS.rand_crop:
            transform.append(transforms.RandomScaleCropResizeStereo())
        else:
            transform.append(transforms.ResizeStereo())
        transform.append(transforms.ArrayToTensorStereo())
        transform.append(normalize)
        self.transform = transforms.Compose(transform)

        self.samples = self._crawl_folders()


    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_l = load_as_float(sample['tgt_l'])
        ref_img_l = load_as_float(sample['ref_img_l'])
        tgt_r = load_as_float(sample['tgt_r'])
        ref_img_r = load_as_float(sample['ref_img_r'])
        intrinsics_l = np.copy(sample['intrinsics_l'])
        intrinsics_r = np.copy(sample['intrinsics_r'])
        filename_tgt = sample['filename_tgt']
        trg_pose = sample['trg_pose']
        ref_pose = sample['ref_pose']
        baseline = sample['baseline']
        images = [tgt_l]
        intrinsics = [intrinsics_l]
        if ref_img_l is not None:
            images.append(ref_img_l)
        if tgt_r is not None:
            images.append(tgt_r)
            intrinsics.append(intrinsics_r)
        if ref_img_r is not None:
            images.append(ref_img_r)

        # transform
        if self.transform is not None:
            args = [intrinsics, self.stereo, self.height, self.width]
            images_out, args_out = self.transform(images, args)
            intrinsics_l = args_out[0][0]
            if tgt_r is not None:
                intrinsics_r = args_out[0][1]
            tgt_l = images_out[0]
        else:
            images_out = images

        if self.seq_length == 1 and not self.stereo:
            return [tgt_l, intrinsics_l, np.linalg.inv(intrinsics_l), trg_pose, filename_tgt]
        if self.seq_length == 2 and not self.stereo:
            return [tgt_l, images_out[1], intrinsics_l, np.linalg.inv(intrinsics_l), trg_pose, ref_pose, filename_tgt,
                    sample['filename_ref']]
        if self.seq_length == 1 and self.stereo:
            return [tgt_l, images_out[1], intrinsics_l, intrinsics_r, np.linalg.inv(intrinsics_l), np.linalg.inv(intrinsics_r), \
                   trg_pose, filename_tgt, baseline]
        if self.seq_length == 2 and self.stereo:
            return [tgt_l, images_out[1], images_out[2], images_out[3], intrinsics_l, intrinsics_r, np.linalg.inv(intrinsics_l), \
                   np.linalg.inv(intrinsics_r), trg_pose, ref_pose, filename_tgt, sample['filename_ref'], baseline]


    def __len__(self):
        return len(self.samples)


    def _crawl_folders(self):
        '''
            sequence_length can be either 1 or 2
        '''
        sequence_set = []
        for folder_path in self.scenes:

            # left imgs
            left_imgs_path = folder_path/'cam0'/'data'
            imgs_left = sorted(left_imgs_path.files('*.png'))
            intrinsics_l_path = folder_path / 'cam0' / 'sensor.yaml'
            intrinsics_dict = yaml.load(open(intrinsics_l_path))
            intrinsics_np = np.array(intrinsics_dict['intrinsics']).astype(np.float32)
            assert len(intrinsics_np) == 4
            intrinsics_l = intrinsics_matrix(intrinsics_np)
            if len(imgs_left) < self.seq_length:
                continue

            # right imgs
            baseline = -1
            if self.stereo:
                right_imgs_path = folder_path /'cam1'/'data'
                imgs_right = sorted(right_imgs_path.files('*.png'))
                assert len(imgs_left) == len(imgs_right), '{}: imgs_left = {}, imgs_right = {}'.format(folder_path,
                       len(imgs_left), len(imgs_right))
                intrinsics_r_path = folder_path /'cam1'/'sensor.yaml'
                intrinsics_dict = yaml.load(open(intrinsics_r_path))
                intrinsics_np = np.array(intrinsics_dict['intrinsics']).astype(np.float32)
                assert len(intrinsics_np) == 4
                intrinsics_right = intrinsics_matrix(intrinsics_np)
                if len(imgs_right) < self.seq_length:
                    continue

            # GT
            pose_gt_csv_path = folder_path / 'state_groundtruth_estimate0/data.csv'
            filename2time_csv_path = folder_path/'cam0'/'data.csv'
            filenames_with_gt, angles, tx, ty, tz = read_pose_csv(pose_gt_csv_path, filename2time_csv_path,
                                                                  euler_angles=self.euler_angles)
            imgs_left_np = [(imgs_left[i]).split("/")[-1] for i in range(len(imgs_left))]

            # skip first 120 images (they are static)
            for i in range(120, len(imgs_left) - (self.seq_length-1) - (self.skip-1), 1):
                ref_img_l, intrinsics_r, tgt_r, ref_img_r, filename_ref = None, None, None, None, None
                trg_pose, ref_pose = [], []
                trg_with_gt_bool = imgs_left_np[i] in filenames_with_gt
                ref_with_gt_bool = False
                filename_tgt = imgs_left[i]
                if self.seq_length == 2:
                    ref_img_l = imgs_left[i+self.skip]
                    filename_ref = imgs_left[i+self.skip]
                if self.stereo:
                    tgt_r = imgs_right[i]
                    intrinsics_r = intrinsics_right
                    if self.seq_length == 2:
                        ref_img_r = imgs_right[i+self.skip]

                # with pose GT
                if trg_with_gt_bool:
                    filenames_gt_idx_i = filenames_with_gt.index(imgs_left_np[i])
                    if self.euler_angles:
                        rx, ry, rz = angles
                        trg_pose = np.array([tx[filenames_gt_idx_i], ty[filenames_gt_idx_i], tz[filenames_gt_idx_i],
                                             rx[filenames_gt_idx_i], ry[filenames_gt_idx_i], rz[filenames_gt_idx_i]])
                    else:
                        qx, qy, qz, qw = angles
                        trg_pose = np.array([tx[filenames_gt_idx_i], ty[filenames_gt_idx_i], tz[filenames_gt_idx_i],
                                             qx[filenames_gt_idx_i], qy[filenames_gt_idx_i], qz[filenames_gt_idx_i],
                                             qw[filenames_gt_idx_i]])
                    if self.seq_length == 2:
                        ref_with_gt_bool = imgs_left_np[i+self.skip] in filenames_with_gt
                        if ref_with_gt_bool:
                            filenames_gt_idx_ip1 = filenames_with_gt.index(imgs_left_np[i+self.skip])
                            if self.euler_angles:
                                ref_pose = np.array([tx[filenames_gt_idx_ip1], ty[filenames_gt_idx_ip1], tz[filenames_gt_idx_ip1],
                                                     rx[filenames_gt_idx_ip1], ry[filenames_gt_idx_ip1], rz[filenames_gt_idx_ip1]])
                            else:
                                ref_pose = np.array([tx[filenames_gt_idx_ip1], ty[filenames_gt_idx_ip1], tz[filenames_gt_idx_ip1],
                                     qx[filenames_gt_idx_ip1], qy[filenames_gt_idx_ip1], qz[filenames_gt_idx_ip1],
                                     qw[filenames_gt_idx_ip1]])
                # without pose GT
                if not self.with_gt:
                    trg_pose, ref_pose = [], []

                sample = {'intrinsics_l': intrinsics_l, 'tgt_l': imgs_left[i], 'ref_img_l': ref_img_l,
                          'intrinsics_r': intrinsics_r, 'tgt_r': tgt_r, 'ref_img_r': ref_img_r,
                          'trg_pose': trg_pose, 'ref_pose': ref_pose, 'baseline': baseline,
                          'filename_tgt': filename_tgt, 'filename_ref': filename_ref}
                # if with_gt, add only samples with gt pose
                if (not self.with_gt) or (trg_with_gt_bool and self.seq_length==1) or (trg_with_gt_bool and ref_with_gt_bool and self.seq_length==2):
                    sequence_set.append(sample)

        return sequence_set






if __name__ == '__main__':
    import multiprocessing
    class FLAGS():
        def __init__(self):
            self.batch_size = 7
            self.data_dir = '/media/victoria/d/data/EuRoC_MAV'
            self.height = 240
            self.width = 376
            self.seed = 127
            self.stereo = 1
            self.seq_length = 2
            self.euler_angles = True
            self.with_gt = 1
            self.hflip = 1
            self.rand_crop = 1
    FLAGS = FLAGS()
    mode = 'train'
    train_set = EuRoc(FLAGS, mode)
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('stereo: ', bool(FLAGS.stereo), ", seq_length = ", FLAGS.seq_length, 'with_gt:', bool(FLAGS.with_gt))

    train_loader = data.DataLoader(
        train_set, batch_size=FLAGS.batch_size, shuffle=False,
        num_workers=multiprocessing.cpu_count()-2, pin_memory=True)
    epoch_size = len(train_loader)
    print('epoch_size = ', epoch_size)

    if not FLAGS.stereo and FLAGS.seq_length == 1:
        for i, var_list in enumerate(train_loader):
            tgt_img_l, intrinsics_l, intrinsics_inv, trg_pose, filename_tgt = var_list
            print('\n', i, 'tgt_img_l = ', tgt_img_l.shape)  # tgt_img is a PyTorch tensor
            print('intrinsics_l = ', intrinsics_l)
            print('filename_tgt = ', filename_tgt)
            if trg_pose is not None:
                print('trg_pose = ', trg_pose)
    elif not FLAGS.stereo and FLAGS.seq_length > 1:
        for i, var_list in enumerate(train_loader):
            tgt_img_l, ref_imgs_l, intrinsics_l, intrinsics_inv, trg_pose, ref_poses, filename_tgt, filenames_ref = var_list
            print('\n', i, 'tgt_img_l = ', tgt_img_l.shape)
            print('ref_imgs_l = ', len(ref_imgs_l), ref_imgs_l[0].shape)
            print('intrinsics_l = ', intrinsics_l)
            print('filename_tgt = ', filename_tgt)
            if trg_pose is not None:
                print('trg_pose = ', trg_pose)
            print('filenames_ref = ', filenames_ref)
            if len(ref_poses) > 0:
                print('ref_poses = ', ref_poses)
    elif FLAGS.stereo and FLAGS.seq_length == 1:
        for i, var_list in enumerate(train_loader):
            tgt_img_l, tgt_img_r, intrinsics_l, intrinsics_r, intrinsics_inv_l, intrinsics_inv_r, trg_pose, \
            filename_tgt, baseline = var_list
            print('\n', i, 'tgt_img_l = ', tgt_img_l.shape)
            print('tgt_img_r = ', tgt_img_r.shape)
            print('intrinsics_l = ', intrinsics_l)
            print('intrinsics_r = ', intrinsics_r)
            print('baseline = ', baseline)
            print('filename_tgt = ', filename_tgt)
            if trg_pose is not None:
                print('trg_pose = ', trg_pose)
    elif FLAGS.stereo and FLAGS.seq_length > 1:
        for i, var_list in enumerate(train_loader):
            tgt_img_l, ref_imgs_l, tgt_img_r, ref_imgs_r, intrinsics_l, intrinsics_r, intrinsics_inv_l, \
            intrinsics_inv_r, trg_pose, ref_poses, filename_tgt, filenames_ref, baseline = var_list
            print('\n', i, 'tgt_img_l = ', tgt_img_l.shape)
            print('ref_imgs_l = ', len(ref_imgs_l), ref_imgs_l[0].shape)
            print('tgt_img_r = ', tgt_img_r.shape)
            print('ref_imgs_r = ', len(ref_imgs_r), ref_imgs_r[0].shape)
            print('intrinsics_l = ', intrinsics_l)
            print('intrinsics_r = ', intrinsics_r)
            print('baseline = ', baseline)
            print('filename_tgt = ', filename_tgt)
            if trg_pose is not None:
                print('trg_pose = ', trg_pose)
            print('filenames_ref = ', filenames_ref)
            if len(ref_poses) > 0:
                print('ref_poses = ', ref_poses)

























# import torch.utils.data as data
# import numpy as np
# from path import Path
# import random
# import yaml
# from PIL import Image
#
# import dataloaders.img_transforms as transforms
# from utils.auxiliary import load_as_float, intrinsics_matrix, read_pose_csv
#
# class EuRoc(data.Dataset):
#     # TODO: to convert ref_imgs_l, ref_imgs_r, filenames_ref and ref_poses to lists (as in KITTY_odom)
#     """
#         Returns EuRoc_MAV images depending on the input parameters.
#         EuRoc_MAV is a dataset that includes serieses of stereo images with GT pose,
#         camera calibrations and IMU measurements (angular velicities and accelerations).
#         No baseline is found in the camera data (baseline = -1 and it is returned to be
#         consistent with the KITTY odometru dataset).
#
#         Input:
#          - FLAGS: class that includes the following attributes:
#                  - data_dir: root directory of EuRoc_MAV
#                  - stereo: True or False
#                  - seq_length: 1 or 2 (returns either only target image i, or, if 2, return target image i and
#                  reference image i+skip)
#                  - height: resize height (if smaller than input height, image height stays the same)
#                  - width: resize width (if smaller than input width, image width stays the same)
#                  - with_gt: only samples with GT
#                  - hflip: if True, images are randomly flipped (if stereo, left and right images are interchanged)
#                  - rand_crop: if True, image will be randomly cropped (up to 15%) (and resized)
#                  - seed: random seed
#         - mode: 'train', 'test' and 'eval'.
#
#         Output:
#             if stereo is False and seq_length = 1 every call returns tuple:
#                 (tgt_img, intrinsics, intrinsics_inv, trg_pose, filename_tgt)
#             if stereo is False and seq_length = 2 every call returns tuple:
#                 (tgt_img, ref_imgs, intrinsics, intrinsics_inv, trg_pose, ref_poses, filename_tgt, filename_ref)
#             if stereo is True and seq_length = 1 every call returns tuple:
#                 (tgt_img_l, tgt_img_r, intrinsics_l, intrinsics_r, intrinsics_inv_l, intrinsics_inv_r, trg_pose,
#                 filename_tgt, baseline)
#             if stereo is True and seq_length = 2 every call returns tuple:
#                 (tgt_img_l, ref_imgs_l, tgt_img_r, ref_imgs_r, intrinsics_l, intrinsics_r, intrinsics_inv_l,
#                 intrinsics_inv_r, trg_pose, ref_poses, filename_tgt, filename_ref, baseline)
#
#             - image is a PyTorch tensor in format (B, C, H, W)
#             - intrinsics is a PyTorch tensor (B, 3, 3)
#             - pose is a PyTorch tensor of shape (B, 6), if euler_angles is True 6dof is in format (tx, ty, tz, rx, ry, rz),
#                     otherwise in format (tx, ty, tz, qx, qy, qz, qw)
#             - filename is of shape (B, 1), it is a full path to an image (in format Path)
#
#     """
#     def __init__(self, FLAGS, mode):
#         super(EuRoc, self).__init__()
#         seed = FLAGS.seed
#         self.stereo = FLAGS.stereo
#         self.seq_length = FLAGS.seq_length
#         self.height = FLAGS.height
#         self.width = FLAGS.width
#         self.with_gt = FLAGS.with_gt
#         if hasattr(FLAGS, 'skip'):
#             self.skip = FLAGS.skip
#         else:
#             self.skip = 1
#         np.random.seed(seed)
#         random.seed(seed)
#         self.root = Path(FLAGS.data_dir)
#         self.euler_angles = FLAGS.euler_angles
#         train, eval = False, False
#         if mode == 'train':
#             train = True
#         if mode == 'eval':
#             eval = True
#         scene_list_path = self.root/'train.txt' if train else self.root/'test.txt'
#         if eval:
#             scene_list_path = self.root/'eval.txt'
#         self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
#
#
#         # transforms
#         normalize = transforms.NormalizeStereo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         transform = []
#         if FLAGS.hflip:
#             transform.append(transforms.RandomHorizontalFlipStereo())
#         if FLAGS.rand_crop:
#             transform.append(transforms.RandomScaleCropResizeStereo())
#         else:
#             transform.append(transforms.ResizeStereo())
#         transform.append(transforms.ArrayToTensorStereo())
#         transform.append(normalize)
#         self.transform = transforms.Compose(transform)
#
#         self.samples = self._crawl_folders()
#
#
#     def __getitem__(self, index):
#         sample = self.samples[index]
#         tgt_img_l = load_as_float(sample['tgt_img_l'])
#         ref_imgs_l = [load_as_float(img) for img in sample['ref_imgs_l']]
#         if sample['tgt_img_r'] is not None:
#             tgt_img_r = load_as_float(sample['tgt_img_r'])
#         else:
#             tgt_img_r = []
#         ref_imgs_r = [load_as_float(img) for img in sample['ref_imgs_r']]
#         intrinsics_l = np.copy(sample['intrinsics_l'])
#         intrinsics_inv_l = np.linalg.inv(intrinsics_l)
#         if sample['intrinsics_r'] is not None:
#             intrinsics_r = np.copy(sample['intrinsics_r'])
#             intrinsics_inv_r = np.linalg.inv(intrinsics_r)
#         else:
#             intrinsics_r = []
#             intrinsics_inv_r = []
#         filename_tgt = sample['tgt_img_l']
#         filenames_ref = sample['ref_imgs_l']
#         trg_pose = sample['trg_pose']
#         ref_poses = sample['ref_poses']
#         baseline = sample['baseline']
#
#         # transform
#         images = [tgt_img_l]
#         intrinsics = [intrinsics_l]
#         num_ref_img = 0
#         if len(ref_imgs_l) > 0:
#             num_ref_img = len(ref_imgs_l)
#             for img in ref_imgs_l:
#                 images.append(img)
#         if sample['tgt_img_r'] is not None:
#             images.append(tgt_img_r)
#             intrinsics.append(intrinsics_r)
#         if len(ref_imgs_r) > 0:
#             for img in ref_imgs_r:
#                 images.append(img)
#         if self.transform is not None:
#             args = [intrinsics, self.stereo, self.height, self.width]
#             images_out, args_out = self.transform(images, args)
#             intrinsics_l = args_out[0][0]
#             if sample['tgt_img_r'] is not None:
#                 intrinsics_r = args_out[0][1]
#             tgt_img_l = images_out[0]
#         else:
#             images_out = images
#
#         if self.seq_length > 1 and not self.stereo:
#             ref_imgs_l = images_out[1:]
#         if self.seq_length == 1 and self.stereo:
#             tgt_img_r = images_out[1]
#         if self.seq_length > 1 and self.stereo:
#             ref_imgs_l = images_out[1:1+num_ref_img]
#             tgt_img_r = images_out[1+num_ref_img]
#             ref_imgs_r = images_out[1+num_ref_img+1:]
#
#         return tgt_img_l, ref_imgs_l, tgt_img_r, ref_imgs_r, intrinsics_l, intrinsics_r, intrinsics_inv_l, \
#                intrinsics_inv_r, trg_pose, ref_poses, filename_tgt, filenames_ref, baseline
#
#
#     def __len__(self):
#         return len(self.samples)
#
#
#     def _crawl_folders(self):
#         '''
#             sequence_length can be either 1 or 2
#         '''
#         sequence_set = []
#         for folder_path in self.scenes:
#
#             # left imgs
#             left_imgs_path = folder_path/'cam0'/'data'
#             imgs_left = sorted(left_imgs_path.files('*.png'))
#             intrinsics_l_path = folder_path / 'cam0' / 'sensor.yaml'
#             intrinsics_dict = yaml.load(open(intrinsics_l_path))
#             intrinsics_np = np.array(intrinsics_dict['intrinsics']).astype(np.float32)
#             assert len(intrinsics_np) == 4
#             intrinsics_l = intrinsics_matrix(intrinsics_np)
#             if len(imgs_left) < self.seq_length:
#                 continue
#
#             # right imgs
#             baseline = -1
#             if self.stereo:
#                 right_imgs_path = folder_path /'cam1'/'data'
#                 imgs_right = sorted(right_imgs_path.files('*.png'))
#                 assert len(imgs_left) == len(imgs_right), '{}: imgs_left = {}, imgs_right = {}'.format(folder_path,
#                        len(imgs_left), len(imgs_right))
#                 intrinsics_r_path = folder_path /'cam1'/'sensor.yaml'
#                 intrinsics_dict = yaml.load(open(intrinsics_r_path))
#                 intrinsics_np = np.array(intrinsics_dict['intrinsics']).astype(np.float32)
#                 assert len(intrinsics_np) == 4
#                 intrinsics_right = intrinsics_matrix(intrinsics_np)
#                 if len(imgs_right) < self.seq_length:
#                     continue
#
#             # GT
#             pose_gt_csv_path = folder_path / 'state_groundtruth_estimate0/data.csv'
#             filename2time_csv_path = folder_path/'cam0'/'data.csv'
#             filenames_with_gt, angles, tx, ty, tz = read_pose_csv(pose_gt_csv_path, filename2time_csv_path,
#                                                                   euler_angles=self.euler_angles)
#             imgs_left_np = [(imgs_left[i]).split("/")[-1] for i in range(len(imgs_left))]
#
#             # skip first 120 images (they are static)
#             for i in range(120, len(imgs_left) - (self.seq_length-1) - (self.skip-1), 1):
#                 tgt_img_r, intrinsics_r = None, None
#                 ref_imgs_l, ref_imgs_r, trg_pose, ref_poses = [], [], [], []
#
#                 trg_with_gt_bool = imgs_left_np[i] in filenames_with_gt
#                 ref_with_gt_bool = False
#                 if self.seq_length == 2:
#                     ref_imgs_l = imgs_left[i+self.skip]
#                 if self.stereo:
#                     tgt_img_r = imgs_right[i]
#                     intrinsics_r = intrinsics_right
#                     if self.seq_length == 2:
#                         ref_imgs_r = imgs_right[i+self.skip]
#
#                 # with pose GT
#                 if trg_with_gt_bool:
#                     filenames_gt_idx_i = filenames_with_gt.index(imgs_left_np[i])
#                     if self.euler_angles:
#                         rx, ry, rz = angles
#                         trg_pose = np.array([tx[filenames_gt_idx_i], ty[filenames_gt_idx_i], tz[filenames_gt_idx_i],
#                                              rx[filenames_gt_idx_i], ry[filenames_gt_idx_i], rz[filenames_gt_idx_i]])
#                     else:
#                         qx, qy, qz, qw = angles
#                         trg_pose = np.array([tx[filenames_gt_idx_i], ty[filenames_gt_idx_i], tz[filenames_gt_idx_i],
#                                              qx[filenames_gt_idx_i], qy[filenames_gt_idx_i], qz[filenames_gt_idx_i],
#                                              qw[filenames_gt_idx_i]])
#                     if self.seq_length == 2:
#                         ref_with_gt_bool = imgs_left_np[i+self.skip] in filenames_with_gt
#                         if ref_with_gt_bool:
#                             filenames_gt_idx_ip1 = filenames_with_gt.index(imgs_left_np[i+self.skip])
#                             if self.euler_angles:
#                                 ref_poses = np.array([tx[filenames_gt_idx_ip1], ty[filenames_gt_idx_ip1], tz[filenames_gt_idx_ip1],
#                                                      rx[filenames_gt_idx_ip1], ry[filenames_gt_idx_ip1], rz[filenames_gt_idx_ip1]])
#                             else:
#                                 ref_poses = np.array([tx[filenames_gt_idx_ip1], ty[filenames_gt_idx_ip1], tz[filenames_gt_idx_ip1],
#                                      qx[filenames_gt_idx_ip1], qy[filenames_gt_idx_ip1], qz[filenames_gt_idx_ip1],
#                                      qw[filenames_gt_idx_ip1]])
#                 # without pose GT
#                 if not self.with_gt:
#                     trg_pose, ref_poses = [], []
#
#                 sample = {'intrinsics_l': intrinsics_l, 'tgt_img_l': imgs_left[i], 'ref_imgs_l': [ref_imgs_l],
#                           'intrinsics_r': intrinsics_r, 'tgt_img_r': tgt_img_r, 'ref_imgs_r': [ref_imgs_r],
#                           'trg_pose': trg_pose, 'ref_poses': [ref_poses], 'baseline': baseline}
#                 # if with_gt, add only samples with gt pose
#                 if (not self.with_gt) or (trg_with_gt_bool and self.seq_length==1) or (trg_with_gt_bool and ref_with_gt_bool and self.seq_length==2):
#                     sequence_set.append(sample)
#
#         return sequence_set
#
#
#
#
#
#
# if __name__ == '__main__':
#     import multiprocessing
#     class FLAGS():
#         def __init__(self):
#             self.batch_size = 7
#             self.data_dir = '/media/victoria/d/data/EuRoC_MAV'
#             self.height = 240
#             self.width = 376
#             self.seed = 127
#             self.stereo = 1
#             self.seq_length = 2
#             self.euler_angles = True
#             self.with_gt = 1
#             self.hflip = 1
#             self.rand_crop = 1
#     FLAGS = FLAGS()
#     mode = 'train'
#     train_set = EuRoc(FLAGS, mode)
#     print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
#     print('stereo: ', bool(FLAGS.stereo), ", seq_length = ", FLAGS.seq_length, 'with_gt:', bool(FLAGS.with_gt))
#
#     train_loader = data.DataLoader(
#         train_set, batch_size=FLAGS.batch_size, shuffle=False,
#         num_workers=multiprocessing.cpu_count()-2, pin_memory=True)
#     epoch_size = len(train_loader)
#     print('epoch_size = ', epoch_size)
#
#     for i, (tgt_img_l, ref_imgs_l, tgt_img_r, ref_imgs_r, intrinsics_l, intrinsics_r, intrinsics_inv_l, \
#            intrinsics_inv_r, trg_pose, ref_poses, filename_tgt, filenames_ref, baseline) in enumerate(train_loader):
#         print('\n', i, 'tgt_img_l = ', tgt_img_l.shape)  # tgt_img is a PyTorch tensor
#         print('filename_tgt = ', filename_tgt)
#         print('intrinsics_l = ', intrinsics_l)
#         if len(trg_pose) > 0:
#             print('trg_pose = ', trg_pose)
#         if not FLAGS.stereo and FLAGS.seq_length > 1:
#             print('ref_imgs_l = ', len(ref_imgs_l), ref_imgs_l[0].shape)
#             print('filenames_ref = ', filenames_ref)
#             if len(ref_poses) > 0:
#                 print('ref_poses = ', ref_poses)
#         elif FLAGS.stereo and FLAGS.seq_length == 1:
#             print('tgt_img_r = ', tgt_img_r.shape)
#             print('intrinsics_r = ', intrinsics_r)
#             print('baseline = ', baseline)
#         elif FLAGS.stereo and FLAGS.seq_length > 1:
#             print('ref_imgs_l = ', len(ref_imgs_l), ref_imgs_l[0].shape)
#             print('tgt_img_r = ', tgt_img_r.shape)
#             print('ref_imgs_r = ', len(ref_imgs_r), ref_imgs_r[0].shape)
#             print('intrinsics_r = ', intrinsics_r)
#             print('baseline = ', baseline)
#             print('filenames_ref = ', filenames_ref)
#             if len(ref_poses) > 0:
#                 print('ref_poses = ', ref_poses)






















