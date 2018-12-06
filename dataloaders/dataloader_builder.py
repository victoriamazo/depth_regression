import os
import imp
import numpy as np
import itertools
import multiprocessing

import torch.utils.data as data



class DataLoader(object):
    def __init__(self, FLAGS, mode):
        self.FLAGS = FLAGS
        self.data_dir = FLAGS.data_dir
        assert os.path.isdir(self.data_dir), 'data_dir {} is not a directory'.format(self.data_dir)
        self.batch_size = FLAGS.batch_size
        self.mode = mode
        self.subset_datadir = os.path.join(self.data_dir, mode)
        assert os.path.isdir(self.subset_datadir), 'subset_datadir {} is not a directory'.format(self.subset_datadir)


    def batch_generator(self, sample_generator, num_outputs=2, seed=1331):
        '''Yields (X, y)
            - X has shape (batch_size, H, W, n_channels)
            - y has shape (batch_size)'''
        np.random.seed(seed)
        while True:
            batch = list(itertools.islice(sample_generator, self.batch_size))
            if num_outputs == 2:
                x, y = zip(*batch)
                yield np.stack(x), np.stack(y)
            elif num_outputs == 3:
                x, y, a = zip(*batch)
                yield np.stack(x), np.stack(y), np.stack(a)
            elif num_outputs == 4:
                x, y, a, b = zip(*batch)
                yield np.stack(x), np.stack(y), np.stack(a), np.stack(b)
            elif num_outputs == 5:
                x, y, a, b, c = zip(*batch)
                yield np.stack(x), np.stack(y), np.stack(a), np.stack(b), np.stack(c)
            else:
                print('{} number of outputs is not valid'.format(num_outputs))


    def build(self):
        raise NotImplementedError('DataLoader.build is not implemented')


    def len(self):
        raise NotImplementedError('DataLoader.len is not implemented')


    @classmethod
    def dataloader_builder(cls, target_class, FLAGS, mode, parent_class=None):
        path = os.path.dirname(os.path.realpath(__file__))
        for filename in os.listdir(path):
            prefix, suffix = filename.split('.')[0], filename.split('.')[-1]
            if suffix == 'py' and prefix != '__init__' and prefix != 'data_loader_builder' and prefix != 'preprocessing':
                path_to_module = os.path.join(path, filename)
                module_dir, module_file = os.path.split(path_to_module)
                module_name, module_ext = os.path.splitext(module_file)
                module_obj = imp.load_source(module_name, path_to_module)

                for name in dir(module_obj):
                    if name == target_class:
                        o = getattr(module_obj, name)
                        if parent_class is None:
                            try:
                                if issubclass(o, cls):
                                    return o(FLAGS, mode).build()
                            except TypeError:
                                pass
                        elif parent_class == data.Dataset:
                            try:
                                if issubclass(o, parent_class):

                                    dataset = o(FLAGS, mode)
                                    shuffle = False
                                    if mode == 'train' and hasattr(FLAGS, 'shuffle') and FLAGS.shuffle:
                                        shuffle = True
                                    dataloader = data.DataLoader(dataset,
                                                                 batch_size=FLAGS.batch_size,
                                                                 shuffle=shuffle,
                                                                 num_workers=int(multiprocessing.cpu_count()/4),
                                                                 pin_memory=True,
                                                                 drop_last=True)
                                    return dataloader, len(dataset)
                            except TypeError:
                                pass
                        else:
                            print('Unknown parent class of a dataset given.')






















