import torch.utils.data as data
import numpy as np
from path import Path
import random

import dataloaders.img_transforms as transforms
from utils.auxiliary import load_as_float


class KITTY_formatted(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """
    def __init__(self, FLAGS, mode):
        super(KITTY_formatted, self).__init__()
        np.random.seed(FLAGS.seed)
        random.seed(FLAGS.seed)
        self.root = Path(FLAGS.data_dir)
        self.stereo = FLAGS.stereo
        self.height = FLAGS.height
        self.width = FLAGS.width
        train = False
        if mode == 'train':
            train = True
        sequence_length = FLAGS.seq_length
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'

        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.samples = self._crawl_folders(self.scenes, sequence_length)

        # normalize = transforms.NormalizeStereo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # self.transform = transforms.Compose([
        #     transforms.RandomHorizontalFlipStereo(),
        #     transforms.RandomScaleCropResizeStereo(),
        #     transforms.ArrayToTensorStereo(),
        #     normalize])
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


    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            args = [[np.copy(sample['intrinsics'])], self.stereo, self.height, self.width]
            imgs, args_out = self.transform([tgt_img] + ref_imgs, args)
            intrinsics = args_out[0][0]
            tgt_img = imgs[0]
            ref_imgs = imgs[1]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        trg_pose, ref_pose = [], []
        filename_tgt = sample['tgt']
        filename_ref = sample['ref_imgs'][0]
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), trg_pose, ref_pose, filename_tgt, filename_ref


    def __len__(self):
        return len(self.samples)


    def _crawl_folders(self, folders_list, sequence_length):
        sequence_set = []
        demi_length = 1
        for folder in folders_list:
            intrinsics = np.genfromtxt(folder / 'cam.txt', delimiter=',').astype(np.float32).reshape((3, 3))
            imgs = sorted(folder.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs) - demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in range(-demi_length, demi_length):
                    if j != 0:
                        sample['ref_imgs'].append(imgs[i + j])
                sequence_set.append(sample)
        return sequence_set







if __name__ == '__main__':
    import multiprocessing
    import torch

    class FLAGS():
        def __init__(self):
            self.batch_size = 1
            self.data_dir = '/media/victoria/d/data/KITTI_formatted'
            self.height = 128
            self.width = 416
            self.seq_length = 2
            self.seed = 127
            self.stereo = 1
            self.euler_angles = True
            self.with_gt = 1
            self.hflip = 1
            self.rand_crop = 1


    FLAGS = FLAGS()
    mode = 'train'
    train_set = KITTY_formatted(FLAGS, mode)
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))

    train_loader = data.DataLoader(
        train_set, batch_size=FLAGS.batch_size, shuffle=False,
        num_workers=multiprocessing.cpu_count() - 2, pin_memory=True)
    epoch_size = len(train_loader)
    print('epoch_size = ', epoch_size)

    for i, (tgt_img_l, ref_img_l, intrinsics, intrinsics_inv, trg_pose, ref_pose, filename_tgt, filename_ref) \
            in enumerate(train_loader):
        print('\n', i, 'filename_tgt = ', filename_tgt)
        print('filename_ref = ', filename_ref)
        print('tgt_img_l = ', tgt_img_l.shape, torch.min(tgt_img_l), torch.max(tgt_img_l), torch.mean(tgt_img_l))
        print('ref_img_l = ', ref_img_l.shape, torch.min(ref_img_l), torch.max(ref_img_l), torch.mean(ref_img_l))
        print('intrinsics_l = ', intrinsics)


























