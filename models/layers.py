import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(FLAGS):
    if not FLAGS is None:
        if hasattr(FLAGS, 'nonlinearity') and FLAGS.nonlinearity == 'elu':
            return nn.ELU(inplace=True)
        if hasattr(FLAGS, 'nonlinearity') and FLAGS.nonlinearity == 'lrelu':
            return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

def conv_ReLU(in_planes, out_planes, kernel_size=3, padding=1, stride=1, FLAGS=None):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride),
                         nn.ReLU(inplace=True))

def conv_bn(in_planes, out_planes, kernel_size=3, padding=1, stride=1, FLAGS=None):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(out_planes))


def conv_bn_ReLU(in_planes, out_planes, kernel_size=3, padding=1, stride=1, FLAGS=None):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


def upconv_ReLU(in_planes, out_planes, kernel_size=3, output_padding=0, FLAGS=None):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=1,
        output_padding=output_padding),
        nn.ReLU(inplace=True))


def downsample_conv(in_planes, out_planes, kernel_size=3, FLAGS=None):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ELU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True))


def predict_disp(in_planes, output=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, output, kernel_size=3, padding=1),
        nn.Sigmoid())


def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

def sigm():
    return nn.Sequential(nn.Sigmoid())


class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.zeros(num_channels, 1, stride, stride)
        self.weights[:, :, 0, 0] = 1

    def forward(self, x):
        # not compatible with running on CPU
        return F.conv_transpose2d(x, self.weights.cuda(), stride=self.stride, groups=self.num_channels)


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer = None

    def forward(self, x):
        x = self.layer(x)
        return x


class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size >= 2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2 * padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                (module_name, nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size,
                                                 stride, padding, output_padding, bias=False)),
                ('batchnorm', nn.BatchNorm2d(in_channels // 72)),
                ('relu', nn.ReLU(inplace=True)),
            ]))

        self.layer = convt(in_channels//2)


class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
            ('unpool', Unpool(in_channels)),
            ('conv', nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, stride=1, padding=2, bias=False)),
            ('batchnorm', nn.BatchNorm2d(in_channels // 2)),
            ('relu', nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer = self.upconv_module(in_channels//2)


class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels):
            super(UpProj.UpProjModule, self).__init__()
            out_channels = in_channels // 2
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)),
                ('batchnorm1', nn.BatchNorm2d(out_channels)),
                ('relu', nn.ReLU()),
                ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)),
                ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer = self.UpProjModule(in_channels)


def choose_decoder(decoder, in_channels):
    # iheight, iwidth = 10, 8
    if decoder[:6] == 'deconv':
        assert len(decoder) == 7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == "upproj":
        return UpProj(in_channels)
    elif decoder == "upconv":
        return UpConv(in_channels)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)










