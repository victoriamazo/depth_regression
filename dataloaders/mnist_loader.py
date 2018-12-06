import itertools
import os
import numpy as np
from PIL import Image
from dataloaders import img_transforms as transforms
from dataloaders.dataloader_builder import DataLoader
from utils.auxiliary import idx2label, convert_to_onehot


class mnist_loader(DataLoader):
    def __init__(self, FLAGS, mode):
        super(mnist_loader, self).__init__(FLAGS, mode)
        self.mode = mode
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.num_classes = FLAGS.num_classes
        if hasattr(FLAGS, 'crop'):
            self.crop = FLAGS.crop
        else:
            self.crop = False
        if hasattr(FLAGS, 'hflip'):
            self.hflip = FLAGS.hflip
        else:
            self.hflip = False
        if hasattr(FLAGS, 'vflip'):
            self.vflip = FLAGS.vflip
        else:
            self.vflip = False
        if hasattr(FLAGS, 'crop_shift'):
            self.crop_shift = FLAGS.crop_shift
        else:
            self.crop_shift = False
        if hasattr(FLAGS, 'shuffle'):
            self.shuffle = FLAGS.shuffle
        else:
            if mode != 'train':
                self.shuffle = False
            else:
                self.shuffle = True
        self.label_dict = idx2label(self.FLAGS.data_dir)
        self.file_set, self.num_samples = self._filelist()

        self.transforms = transforms.Compose([
            transforms.Normalize(0.1307, 0.3081),
        ])

    def build(self):
        num_outputs = 3
        gen = self.batch_generator(self._image_generator(), num_outputs=num_outputs)
        return gen, self.num_samples


    def len(self):
        file_set, num_samples = self._filelist()
        return num_samples


    def _filelist(self):
        file_list = os.listdir(self.subset_datadir)
        file_set = sorted(list(set(file_list)))
        num_samples = len(file_set)
        return file_set, num_samples


    def _filename_generator(self):
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
                yield self.file_set[idx]


    def _image_generator(self):
        '''Receives image, label (from generator).
        image_dir should be a full path.
        Calls transform_image, which returns transformed_image of shape (Img_H, Img_W, n_channels).
        Yields (x, y) of shapes:
            - x (1, Img_H, Img_W, n_channels)
            - y (label)
        '''
        for filename in self._filename_generator():
            image = Image.open(os.path.join(self.subset_datadir, filename)).convert('L')
            label = int((filename.split('.')[0]).split('_')[-1])

            # image augmentation
            x = self.transforms(image)
            y = label            #convert_to_onehot(label, self.label_dict, self.num_classes)

            yield x, y, filename





if __name__ == "__main__":
    class FLAGS():
        def __init__(self):
            self.batch_size = 1
            self.data_dir = '/media/victoria/d/data/mnist'
            self.hflip = True
            self.shuffle = True
            self.height = 28
            self.width = 28
            self.num_classes = 10

            self.debug = True

    FLAGS = FLAGS()
    mode = 'train'

    # generate batch of stereo images
    dataloader = mnist_loader(FLAGS, mode)
    gen, num_samples = dataloader.build()
    img1_trans, y, filename = gen.next()

    # show images
    idx = 0
    print('img1_trans = {}, y = {}, filename = {}'.format(img1_trans.shape, y, filename))
    im1 = Image.fromarray(img1_trans[idx])
    im1_name = 'img1_trans_{}.png'.format(filename[idx])
    im1.save('../debug/{}'.format(im1_name))
    os.system('xdg-open ../debug/{}'.format(im1_name))






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    