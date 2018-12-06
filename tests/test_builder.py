import os
import imp
import json
import glob


class Test(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.data_dir = FLAGS.data_dir
        self.train_dir = FLAGS.train_dir
        self.data_loader = FLAGS.data_loader
        self.batch_size = FLAGS.batch_size
        self.gpus = FLAGS.gpus
        self.ckpts_dir = os.path.join(self.train_dir, 'ckpts')
        self.load_ckpt = ''
        if hasattr(FLAGS, 'load_ckpt') and FLAGS.load_ckpt != '':
            self.load_ckpt = FLAGS.load_ckpt
        if hasattr(FLAGS, 'worker_num'):
            self.worker_num = int(FLAGS.worker_num)
        else:
            self.worker_num = None
        self.debug = FLAGS.debug


    def _check_args(self):
        assert os.path.isdir(self.data_dir), 'correct data_dir field is required'
        assert os.path.isdir(self.train_dir), 'train_dir field is required'
        args_path = os.path.join(self.train_dir, 'args_test.json')
        if self.worker_num != None:
            args_path = os.path.join(self.train_dir, 'args_test_{}.json'.format(self.worker_num))
        with open(args_path, 'wt') as r:
            json.dump({arg: getattr(self.FLAGS, arg) for arg in vars(self.FLAGS)}, r, indent=2)
        if self.debug:
            print('\nArguments: ')
            for arg in vars(self.FLAGS):
                print('   - ', arg, ': ', getattr(self.FLAGS, arg))


    def build(self):
        raise NotImplementedError('Test.build() is not implemented')


    @classmethod
    def test_builder(cls, FLAGS):
        path = os.path.dirname(os.path.realpath(__file__))
        for filename in os.listdir(path):
            prefix, suffix = filename.split('.')[0], filename.split('.')[-1]
            if suffix == 'py' and prefix != '__init__' and prefix != 'test_builder':
                path_to_module = os.path.join(path, filename)
                module_dir, module_file = os.path.split(path_to_module)
                module_name, module_ext = os.path.splitext(module_file)
                module_obj = imp.load_source(module_name, path_to_module)

                for name in dir(module_obj):
                    if name == FLAGS.version:
                        o = getattr(module_obj, name)
                        try:
                            if issubclass(o, cls):
                                return o(FLAGS).build()
                        except TypeError:
                            print('No correct test class found')


