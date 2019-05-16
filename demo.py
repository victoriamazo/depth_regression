import time
import argparse
import torch
from collections import OrderedDict
from PIL import Image
from path import Path
from scipy.misc import imresize
import numpy as np

from models.model_builder import Model
from utils.auxiliary import normalize_depth_for_display
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()

'''
Example:    
    python demo.py ckpts/DispNetS_ckpt_best.pth.tar --arch DispNetS --image images/000011.png 
'''

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', metavar='DIR', help="path to model file")
    parser.add_argument('--arch', type=str, default='DispNetS', help="network architecture: DispNetS or ResNet")
    parser.add_argument('--num_layers', type=int, default=34, help="number of layers in ResNet architecture")
    parser.add_argument('--image', type=str, default='', help="path to image file for depth inference")
    parser.add_argument('--output', type=str, default='', help="(optional) directory for output depth file (will be "
                                                               "save in the same dir as the input image if not given)")
    args = parser.parse_args()
    return args


def main():
    args = getArgs()

    class FLAGS():
        def __init__(self):
            self.batch_size = 1
            self.height = 116
            self.width = 612
            self.xy_cut = [0, 1224, 139, 369]
            self.seed = 127
            self.stereo = 0
            self.stereo_test = 0
            self.hflip = 0
            self.rand_crop = 0
            self.shuffle = 0
            self.num_layers = args.num_layers
    FLAGS = FLAGS()
    img_path = Path(args.image)

    # load model
    model = Model.model_builder(args.arch, FLAGS)
    if use_cuda:
        model = model.cuda()
    ckpt_dict = torch.load(args.model_path)
    if 'iteration' in ckpt_dict:
        train_iter = ckpt_dict['iteration']
    if 'epoch' in ckpt_dict:
        epoch = ckpt_dict['epoch']
    assert isinstance(ckpt_dict['state_dict'], (dict, OrderedDict)), type(ckpt_dict['state_dict'])
    model.load_state_dict(ckpt_dict['state_dict'], strict=False)
    print('{} ckpt loaded from {} (epoch {}, iter {})'.format(args.arch, args.model_path, epoch, train_iter))

    # convert image to tensor
    img = Image.open(img_path)
    img_view = np.copy(img)
    img = imresize(img, (FLAGS.height, FLAGS.width))
    img = np.array(img).astype(np.float32)
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img / 255).float()
    img = img.view(-1, img.size(0), img.size(1), img.size(2))
    if use_cuda:
        img = img.cuda()
    with torch.no_grad():
        img = Variable(img)

    # run inference
    t_begin = time.time()
    model.eval()
    disp = model(img)
    depth = 1.0 / (disp.data[0].cpu().squeeze().numpy() + 1e-6)

    # save output depth
    img_name, img_ext = img_path.basename().split('.')
    if args.output:
        save_path = Path(args.output) / img_name + '_depth.' + img_ext
    else:
        save_path = img_path.dirname() / img_name + '_depth.' + img_ext
    depth = normalize_depth_for_display(depth)
    if hasattr(FLAGS, 'xy_cut') and len(FLAGS.xy_cut) > 0:
        h, w, _ = img_view.shape
        img_view = img_view[FLAGS.xy_cut[2]:FLAGS.xy_cut[3], FLAGS.xy_cut[0]:FLAGS.xy_cut[1]]
        depth = imresize(depth, (h, w))
        depth = depth[FLAGS.xy_cut[2]:FLAGS.xy_cut[3], FLAGS.xy_cut[0]:FLAGS.xy_cut[1]]

    imgs_comb = np.hstack([img_view, depth])
    imgs_concat = Image.fromarray(imgs_comb)
    imgs_concat.save(save_path)
    print('Depth output is saved as {} (inference took {:.2f} sec)'.format(save_path, time.time()-t_begin))








if __name__ == '__main__':
    main()

















