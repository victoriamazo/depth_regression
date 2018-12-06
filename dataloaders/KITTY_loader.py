import itertools
import os

import numpy as np
from PIL import Image

from dataloaders import img_transforms as transforms
from dataloaders.dataloader_builder import DataLoader


class KITTY_loader(DataLoader):
    def __init__(self, FLAGS, mode):
        super(KITTY_loader, self).__init__(FLAGS, mode)
        self.mode = mode
        self.height = FLAGS.height
        self.width = FLAGS.width

        if hasattr(FLAGS, 'series_list'):
            self.series_list = FLAGS.series_list
        else:
            self.series_list = os.listdir(self.subset_datadir)

        if hasattr(FLAGS, 'debug'):
            self.debug = int(FLAGS.debug)
        else:
            self.debug = False

        if hasattr(FLAGS, 'crop_min_h'):
            self.crop_min_h = FLAGS.crop_min_h
        else:
            self.crop_min_h = 1.0
        if hasattr(FLAGS, 'crop_min_w'):
            self.crop_min_w = FLAGS.crop_min_w
        else:
            self.crop_min_w = 1.0

        if hasattr(FLAGS, 'hflip'):
            self.hflip = FLAGS.hflip
        else:
            self.hflip = False

        self.shuffle = True
        if (hasattr(FLAGS, 'shuffle') and FLAGS.shuffle == False) or mode != 'train':
            self.shuffle = False

        self.transforms = transforms.Compose([
            transforms.RandCrop((self.crop_min_h, self.crop_min_w)),
            transforms.Resize((self.height, self.width)),
            transforms.RandHFlip(self.hflip),
        ])

        self.imageL_path_list = list()
        self.imageR_path_list = list()

        # create lists of all images in the given mode (train/test)
        self.num_samples = self._read_lists()


    def build(self):
        num_outputs = 3
        if self.debug:
            num_outputs = 5

        gen = self.batch_generator(self._imgpair_generator(), num_outputs=num_outputs)
        return gen, self.num_samples


    def len(self):
        return self.num_samples


    def _read_lists(self):
        '''
            Create lists of all images in the given mode (train/test)
        '''
        for sequence_dir in self.series_list:
            sequence_dir_path = os.path.join(self.subset_datadir, sequence_dir)
            imageL_dir_path = os.path.join(sequence_dir_path, 'image_2')
            assert os.path.exists(imageL_dir_path)
            self.imageL_path_list.append([os.path.join(imageL_dir_path, image) for image in os.listdir(imageL_dir_path)][:])
            imageR_dir_path = os.path.join(sequence_dir_path, 'image_3')
            assert os.path.exists(imageR_dir_path)
            self.imageR_path_list.append([os.path.join(imageR_dir_path, image) for image in os.listdir(imageR_dir_path)][:])

        self.imageL_path_list = list(itertools.chain(*self.imageL_path_list))
        self.imageR_path_list = list(itertools.chain(*self.imageR_path_list))
        assert len(self.imageL_path_list) == len(self.imageR_path_list)

        print('num sequences in {} mode is {}'.format(self.mode, len(self.series_list)))
        print('num images in {} mode is {}'.format(self.mode, len(self.imageL_path_list)))

        return len(self.imageL_path_list)


    def _imgpath_generator(self):
        '''Before each epoch shuffles image order.
        Each train/test set is given in a flat directory.
        Yields next_filename'''
        # infinite loop over epochs
        for epoch in itertools.count():
            if self.shuffle:
                file_set_idxs = np.random.permutation(self.num_samples)
            else:
                file_set_idxs = range(self.num_samples)

            # loop over all training set (one epoch)
            for idx in file_set_idxs:
                yield self.imageL_path_list[idx], self.imageR_path_list[idx]


    def _imgpair_generator(self):
        '''Receives image, label (from generator).
        image_dir should be a full path.
        Calls transform_image, which returns transformed_image of shape (Img_H, Img_W, n_channels).
        Yields (x, y) of shapes:
            - x (1, Img_H, Img_W, n_channels)
            - y (label)
        '''
        for imageL_path, imageR_path in self._imgpath_generator():
            img1 = Image.open(imageL_path)
            img2 = Image.open(imageR_path)
            filename = (imageL_path.split('/')[-1]).split('.')[0]

            # image augmentation
            img1_trans, img2_trans = list(self.transforms(img1, img2))

            if self.debug:
                yield img1_trans, img2_trans, filename, img1, img2

            else:
                yield img1_trans, img2_trans, filename





if __name__ == "__main__":
    class FLAGS():
        def __init__(self):
            self.batch_size = 1
            self.data_dir = '/media/victoria/d/data/KITTI_odom/data_odometry_color/sequences'
            self.series_list = ['00']
            self.crop_min_h = 0.9
            self.crop_min_w = 0.9
            self.hflip = True
            self.shuffle = True
            self.height = 416
            self.width = 128

            self.debug = True

    FLAGS = FLAGS()
    mode = 'train'

    # generate batch of stereo images
    dataloader = KITTY_loader(FLAGS, mode)
    gen, num_samples = dataloader.build()
    img1_trans, img2_trans, filename, img1, img2 = gen.next()

    # show images
    idx = 0
    print('img1_trans = {}, img2_trans = {}, filename = {}'.format(img1_trans.shape, img2_trans.shape, filename))
    im1 = Image.fromarray(img1_trans[idx])
    im1_name = 'img1_trans_{}.png'.format(filename[idx])
    im1.save('../utils/debug/{}'.format(im1_name))
    os.system('xdg-open ../debug/{}'.format(im1_name))
    im2 = Image.fromarray(img2_trans[idx])
    im2_name = 'img2_trans_{}.png'.format(filename[idx])
    im2.save('../utils/debug/{}'.format(im2_name))
    os.system('xdg-open ../debug/{}'.format(im2_name))
    im3 = Image.fromarray(img1[idx])
    im3_name = 'img1_{}.png'.format(filename[idx])
    im3.save('../utils/debug/{}'.format(im3_name))
    os.system('xdg-open ../debug/{}'.format(im3_name))
    im4 = Image.fromarray(img2[idx])
    im4_name = 'img2_{}.png'.format(filename[idx])
    im4.save('../utils/debug/{}'.format(im4_name))
    os.system('xdg-open ../debug/{}'.format(im4_name))













