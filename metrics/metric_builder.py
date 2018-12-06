import os
import imp



class Metric(object):
    def __init__(self, x, y, FLAGS):
        self.FLAGS = FLAGS


    def build(self):
        raise NotImplementedError('Metric.build() is not implemented')


    @classmethod
    def metric_builder(cls, target_class, x, y, FLAGS):
        path = os.path.dirname(os.path.realpath(__file__))
        for filename in os.listdir(path):
            prefix, suffix = filename.split('.')[0], filename.split('.')[-1]
            if suffix == 'py' and prefix != '__init__' and prefix != 'metric_builder':
                path_to_module = os.path.join(path, filename)
                module_dir, module_file = os.path.split(path_to_module)
                module_name, module_ext = os.path.splitext(module_file)
                module_obj = imp.load_source(module_name, path_to_module)

                for name in dir(module_obj):
                    if name == target_class:
                        o = getattr(module_obj, name)
                        try:
                            if issubclass(o, cls):
                                return o(x, y, FLAGS).build()
                        except TypeError:
                            pass

