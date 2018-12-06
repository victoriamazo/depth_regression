import torch.nn as nn


def conv_ReLU(FLAGS, in_planes, out_planes, kernel_size=3, padding=1, stride=1):
    if hasattr(FLAGS, 'nonlinearity') and FLAGS.nonlinearity == 'elu':
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ELU(inplace=True)
        )
    if hasattr(FLAGS, 'nonlinearity') and FLAGS.nonlinearity == 'lrelu':
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.LeakyReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True)
        )

def upconv_ReLU(FLAGS, in_planes, out_planes, kernel_size=3, output_padding=0):
    if hasattr(FLAGS, 'nonlinearity') and FLAGS.nonlinearity == 'elu':
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=1,
            output_padding=output_padding), nn.ELU(inplace=True)
        )
    if hasattr(FLAGS, 'nonlinearity') and FLAGS.nonlinearity == 'lrelu':
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=1,
            output_padding=output_padding), nn.LeakyReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=1,
            output_padding=output_padding), nn.ReLU(inplace=True)
        )

def downsample_conv(FLAGS, in_planes, out_planes, kernel_size=3):
    if hasattr(FLAGS, 'nonlinearity') and FLAGS.nonlinearity == 'elu':
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
            nn.ELU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ELU(inplace=True)
        )
    if hasattr(FLAGS, 'nonlinearity') and FLAGS.nonlinearity == 'lrelu':
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.LeakyReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True)
        )

def predict_disp(in_planes, output=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, output, kernel_size=3, padding=1),
        nn.Sigmoid()
    )

def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]