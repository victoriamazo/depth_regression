import os
import imp

import torch.nn as nn


class Model(object):
    def __init__(self):
        pass


    @classmethod
    def model_builder(cls, target_class, FLAGS, parent_class=nn.Module):
        path = os.path.dirname(os.path.realpath(__file__))
        for filename in os.listdir(path):
            prefix, suffix = filename.split('.')[0], filename.split('.')[-1]
            if suffix == 'py' and prefix != '__init__' and prefix != 'layers' and prefix != 'model_builder':
                path_to_module = os.path.join(path, filename)
                module_dir, module_file = os.path.split(path_to_module)
                module_name, module_ext = os.path.splitext(module_file)
                module_obj = imp.load_source(module_name, path_to_module)

                for name in dir(module_obj):
                    if str(name) == str(target_class):
                        o = getattr(module_obj, name)
                        try:
                            if issubclass(o, parent_class):
                                return o(FLAGS)
                        except TypeError:
                            pass




