import torch
import torch.nn as nn
from models.layers import conv_ReLU, upconv_ReLU


class ConvNet(nn.Module):

    def __init__(self, FLAGS):
        super(ConvNet, self).__init__()

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv_ReLU(FLAGS, 6, conv_planes[0], kernel_size=7, padding=3, stride=2)
        self.conv2 = conv_ReLU(FLAGS, conv_planes[0], conv_planes[1], kernel_size=5, padding=2, stride=2)
        self.conv3 = conv_ReLU(FLAGS, conv_planes[1], conv_planes[2], stride=2)
        self.conv4 = conv_ReLU(FLAGS, conv_planes[2], conv_planes[3], stride=2)
        self.conv5 = conv_ReLU(FLAGS, conv_planes[3], conv_planes[4], stride=2)
        self.conv6 = conv_ReLU(FLAGS, conv_planes[4], conv_planes[5], stride=2)
        self.conv7 = conv_ReLU(FLAGS, conv_planes[5], conv_planes[6], stride=2)
        self.FCs = nn.Sequential(nn.Linear(1536, 64), nn.Linear(64, 6),)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, target_image, ref_img):
        # concatenate tgt and ref images (along the color channel) for input
        input = [target_image]
        input.extend(ref_img)
        input = torch.cat(input, 1)

        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)
        out_conv7 = out_conv7.view(out_conv7.size(0), -1)
        pose = self.FCs(out_conv7)
        pose = 0.01 * pose.view(pose.size(0), 1, 6)

        exp_mask1 = None
        return exp_mask1, pose



