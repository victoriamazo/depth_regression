# TODO: to implement rotation and color augmentation

import numbers
import random
from PIL import Image, ImageOps
import PIL
import numpy as np
import cv2
from scipy.misc import imresize, imsave
from scipy import ndimage
from skimage.transform import resize
import itertools
try:
    import accimage
except ImportError:
    accimage = None

import torch


class Compose(object):
    """
        Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class ToTensor(object):
    """Convert a batch of ``PIL.Images`` or ``numpy.ndarrays`` to tensor.

    Converts a batch of PIL.Images or numpy.ndarrays (batch_size x H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (batch_size x C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            batch of pics (PIL.Images or numpy.ndarrays): Images of shape (batch_size, H, W(, C))
            to be converted to tensor.
        Returns:
            Tensor: Converted batch of images of shape (batch_size, C, H, W).
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            if len(pic.shape) == 3:
                pic = pic.reshape(pic.shape[0], pic.shape[1], pic.shape[2], -1)

                img = torch.from_numpy(pic.transpose((0, 3, 1, 2)))
            # backward compatibility
            return img.float().div(255)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class Resize(object):
    """
        Resizes the given PIL.Image to the given size.
    size can be a tuple (target_height, target_width) or an integer,
    in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1, img2=None, *args):
        assert img2 is None or img1.size == img2.size

        w, h = img1.size
        resize_h, resize_w = self.size
        if w == resize_w and h == resize_h:
            return (img1, img2, args)

        results = [img1.resize(self.size, PIL.Image.ANTIALIAS)]
        if img2 is not None:
            results.append(img2.resize(self.size, PIL.Image.ANTIALIAS))

        results.extend(args)
        return results


class RandCrop(object):
    """
        Crops the given PIL.Image at a random location to have a region of
    crop_max in the range [crop_min_h, 1] and [crop_min_w, 1]
    (crop_min_h and crop_min_w should be in the range (0,1]).
    crop_max can be a tuple (crop_min_h, cropWmax) or an integer, in which case
    the target will be in the range [crop_min_h, 1] and [crop_min_h, 1]
    """

    def __init__(self, crop_max):
        if isinstance(crop_max, numbers.Number):
            self.crop_max = (int(crop_max), int(crop_max))
        else:
            self.crop_max = crop_max

    def __call__(self, img1, img2=None, *args):
        assert img2 is None or img1.size == img2.size

        crop_min_h, crop_min_w = self.crop_max
        assert crop_min_h > 0 and crop_min_w > 0 and crop_min_h <= 1.0 and crop_min_w <= 1.0
        if crop_min_h == 1.0 and crop_min_w == 1.0:
            return (img1, img2, args)

        w, h = img1.size
        rand_w = random.randint(int(crop_min_w*w), w)
        rand_h = random.randint(int(crop_min_h*h), h)
        x1 = random.randint(0, w - rand_w)
        y1 = random.randint(0, h - rand_h)
        results = [img1.crop((x1, y1, x1 + rand_w, y1 + rand_h))]
        if img2 is not None:
            results.append(img2.crop((x1, y1, x1 + rand_w, y1 + rand_h)))
        results.extend(args)
        return results


class RandHFlip(object):
    """
        Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, hflip):
        self.hflip = hflip


    def __call__(self, img1, img2=None, *args):
        if self.hflip and random.random() < 0.5:
            if img2 is not None:
                results = [img1.transpose(Image.FLIP_LEFT_RIGHT), img2.transpose(Image.FLIP_LEFT_RIGHT)]
            else:
                results = img1.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            if img2 is not None:
                results = [img1, img2]
            else:
                results = img1
        return results


class Normalize(object):
    """
        Normalizes a PIL image or np.aray with a given min and std.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        assert std > 0, 'std for image normalization is 0'

    def __call__(self, img1, img2=None, *args):
        img1 = np.array(img1).astype(np.float32)
        img1 = (img1 - self.mean)/self.std
        if img2 is not None:
            img2 = np.array(img2).astype(np.float32)
            img2 = (img2 - self.mean) / self.std
            results = [img1, img2]
        else:
            results = img1
        return results


class Pad(object):
    """
        Pads the given PIL.Image on all sides with the given "pad" value
    """

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or \
               isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, img1, img2=None, *args):
        if img2 is not None:
            img2 = ImageOps.expand(img2, border=self.padding, fill=255)
        if self.fill == -1:
            img1 = np.asarray(img1)
            img1 = cv2.copyMakeBorder(img1, self.padding, self.padding,
                                       self.padding, self.padding,
                                       cv2.BORDER_REFLECT_101)
            img1 = Image.fromarray(img1)
            return (img1, img2, args)
        else:
            return ImageOps.expand(img1, border=self.padding, fill=self.fill), img2


class RandomHorizontalFlipStereo(object):
    '''Randomly horizontally flips the given numpy array with a probability of 0.5.
        Input:
        - images: list of images of the same size, each image being a np array of shape (height, width, num_channels)
        - intrinsics: list of 3x3 camera intrinsics np arrays
        - stereo: if True, images will be horizontally flipped and interchanged (left and right)
        Output:
         - list of horizontally flipped images
         - correspondingly changed list of intrinsics matrices
    '''
    def __call__(self, images, args):
        intrinsics, stereo, _, _, transf_params = args
        assert stereo is not None
        assert len(intrinsics) > 0
        flip = False

        if random.random() < 0.5:
            flip = True
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
            for i in range(len(output_intrinsics)):
                output_intrinsics[i][0, 2] = w - output_intrinsics[i][0, 2]
            if stereo:
                # intechange left and right images
                assert len(output_images)%2 == 0
                output_images = [output_images[len(output_images)//2:], output_images[:len(output_images)//2]]
                output_images = list(itertools.chain(*output_images))
        else:
            output_images = images
            output_intrinsics = intrinsics
        transf_params_out = transf_params[:-1] + [flip]
        args_out = [output_intrinsics, args[1], args[2], args[3], transf_params_out]
        return output_images, args_out



class RandomScaleCropResizeStereo(object):
    '''If resize dimensions < input dimensions of images, then the images are resized
    and cropped to the resize dimension, otherwise images are randomly zoomed (up to 15%)
    and cropped to keep same size as before. Intrinsics matrix is changes accordingly.
        Input:
        - images: list of images of the same size, each image being a np array of shape (height, width, num_channels)
        - intrinsics: list of 3x3 camera intrinsics np arrays
        - resize height and resize width
        Output:
         - list of randomly cropped images
         - correspondingly changed list of intrinsics matrices
    '''
    def __call__(self, images, args):
        intrinsics, _, resize_h, resize_w, transf_params = args
        assert resize_h is not None
        assert resize_w is not None
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)
        in_h, in_w, _ = images[0].shape
        resize, crop = True, True
        flip = transf_params[-1]

        if resize_h < in_h and resize_w < in_w:
            offset_y = int(resize_h * np.random.uniform(1.0, 1.15) - resize_h)
            offset_x = int(resize_w * np.random.uniform(1.0, 1.15) - resize_w)

            # resize
            scaled_h, scaled_w = int(resize_h + offset_y), int(resize_w + offset_x)
            y_scaling, x_scaling = scaled_h / in_h, scaled_w/in_w
            # im = Image.fromarray(((images[0] / images[0].max()) * 255).astype('uint8'))
            # im.show()
            # im = Image.fromarray(((images[-1] / 80) * 255).astype('uint8'))
            # im.show()
            scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]
            # im1 = Image.fromarray(((scaled_images[-1] / 80) * 255).astype('uint8'))
            # im1.show()

            # crop
            cropped_images = [im[offset_y:offset_y + resize_h, offset_x:offset_x + resize_w] for im in scaled_images]
            # im2 = Image.fromarray(((cropped_images[-1] / 80) * 255).astype('uint8'))
            # im2.show()
            transf_params_out = [resize, scaled_h, scaled_w, crop, offset_y, offset_y + resize_h, offset_x,
                                 offset_x + resize_w, flip]
        else:
            # scale
            x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
            scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)
            scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

            # crop
            offset_y = np.random.randint(scaled_h - in_h + 1)
            offset_x = np.random.randint(scaled_w - in_w + 1)
            cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]
            transf_params_out = [resize, scaled_h, scaled_w, crop, offset_y, offset_y + in_h, offset_x, offset_x + in_w,
                                 flip]

        for i in range(len(output_intrinsics)):
            output_intrinsics[i][0] *= x_scaling
            output_intrinsics[i][1] *= y_scaling
            output_intrinsics[i][0, 2] -= offset_x
            output_intrinsics[i][1, 2] -= offset_y

        args_out = [output_intrinsics, args[1], args[2], args[3], transf_params_out]
        # imsave('/home/victoria/Dropbox/Neural_Networks/Projects/2017.10_PoseEst_pt/utils/debug/tgt.jpg', images[-1])
        # imsave('/home/victoria/Dropbox/Neural_Networks/Projects/2017.10_PoseEst_pt/utils/debug/tgt_scaled.jpg',
        #        scaled_images[-1])
        # imsave('/home/victoria/Dropbox/Neural_Networks/Projects/2017.10_PoseEst_pt/utils/debug/tgt_cropped.jpg',
        #        cropped_images[-1])
        return cropped_images, args_out


class ResizeStereo(object):
    '''If resize dimensions < input dimensions of images, then the images are resized
    to the resize dimension.
        Input:
        - images: list of images of the same size, each image being a np array of shape (height, width, num_channels)
        - resize height and resize width
        Output:
         - list of resized images
         - unchanged intrinsics matrices
    '''
    def __call__(self, images, args):
        intrinsics, _, resize_h, resize_w, transf_params = args
        assert resize_h is not None
        assert resize_w is not None
        in_h, in_w, _ = images[0].shape
        flip = transf_params[-1]

        if resize_h < in_h and resize_w < in_w:
            resized_images = [imresize(im, (resize_h, resize_w)) for im in images]
            resize, crop = True, False
            transf_params_out = [resize, resize_h, resize_w, crop, 0, 0, 0, 0, flip]

            y_scaling, x_scaling = resize_h / in_h, resize_w / in_w
            output_intrinsics = np.copy(intrinsics)
            for i in range(len(output_intrinsics)):
                output_intrinsics[i][0] *= x_scaling
                output_intrinsics[i][1] *= y_scaling
        else:
            resized_images = images
            output_intrinsics = intrinsics
            resize, crop = False, False
            transf_params_out = [resize, 0, 0, crop, 0, 0, 0, 0, flip]

        args_out = [output_intrinsics, args[1], args[2], args[3], transf_params_out]

        return resized_images, args_out


class ArrayToTensorStereo(object):
    '''
        Input:
        - images: list of images of the same size, each image being a np array of shape (height, width, num_channels)
        - intrinsics: list of 3x3 camera intrinsics np arrays
        Output:
         - list of torch.FloatTensor images of shape (C x H x W) normalized to be in the range [0,1]
         - unchanged list of intrinsics matrices
    '''
    def __call__(self, images, args):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im/255).float())
        return tensors, args


class NormalizeStereo(object):
    '''
        Input:
        - images: list of images of the same size, each image being a pytorch tensor of shape (H x W x C)
        - intrinsics: list of 3x3 camera intrinsics np arrays
        Output:
         - list of normalized images
         - unchanged list of intrinsics matrices
    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, args):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, args


class DepthRandomScaleCropResizeStereo(object):
    '''
        Performs depth smoothing, resizing and cropping od depth (with the same parameters as images)
        Input:
        - images: list of left (and right) depth np array of shape (height, width)
        Output:
         - list of left (and right) depth np array, maybe resized and/or cropped
         - unchanged list of args
    '''
    def __call__(self, images, args):
        transf_params = args[-1]
        resize, resize_h, resize_w, crop, offset_y_min, offset_y_max, offset_x_min, offset_x_max, _ = transf_params

        # depth smoothing
        output_images = [ndimage.filters.maximum_filter(im, (5, 5)) for im in images]

        # resize
        if resize:
            output_images = [imresize(im, (resize_h, resize_w)) for im in output_images]
        else:
            output_images = images

        # crop
        if crop:
            output_images = [im[offset_y_min: offset_y_max, offset_x_min: offset_x_max] for im in output_images]
        else:
            output_images = images

        # smooth_lidar_image = Image.fromarray(((output_images[0] / 80) * 255).astype('uint8'))
        # smooth_lidar_image.show()
        #
        # im1 = Image.fromarray(((images[0] / 80) * 255).astype('uint8'))
        # im1.show()

        return output_images, args


class DepthFlipToTensorStereo(object):
    '''
        Input:
        - images: list of left (and right) depth np array of shape (height, width)
        Output:
         - list of depth torch.FloatTensor of shape (H x W)
         - unchanged list of args
    '''
    def __call__(self, images, args):
        transf_params = args[-1]
        flip = transf_params[-1]

        # flip
        if flip:
            output_images = [np.copy(np.fliplr(im)) for im in images]
        else:
            output_images = images

        # depth = output_images[0]
        # im2 = ndimage.filters.maximum_filter(depth, (5, 5))
        # smooth_lidar_image = Image.fromarray(((im2 / 80) * 255).astype('uint8'))
        # smooth_lidar_image.show()
        #
        # im3 = imresize(smooth_lidar_image, (128, 416))
        # im3 = Image.fromarray(((im3 / 80) * 255).astype('uint8'))
        # im3.show()
        #
        # im1 = Image.fromarray(((depth / 80) * 255).astype('uint8'))
        # im1.show()

        # convert to tensors
        tensors = []
        for im in output_images:
            # handle numpy array
            tensors.append(torch.from_numpy(im).float())
        return tensors, args

















