from utils.auxiliary import make_loss_dict

import torch
import torch.nn as nn
from models.layers import conv_ReLU, upconv_ReLU


class PoseExpNet(nn.Module):

    def __init__(self, FLAGS):
        super(PoseExpNet, self).__init__()
        self.nb_ref_imgs = FLAGS.seq_length - 1
        self.output_exp = False
        self.loss_weights_dict, _ = make_loss_dict(FLAGS.loss_weights)
        if 'w_E' in self.loss_weights_dict and self.loss_weights_dict['w_E'] > 0:
            self.output_exp = True
        else:
            print("=> no mask loss, PoseExpnet will only output pose")

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv_ReLU(FLAGS, 3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7, padding=3, stride=2)
        self.conv2 = conv_ReLU(FLAGS, conv_planes[0], conv_planes[1], kernel_size=5, padding=2, stride=2)
        self.conv3 = conv_ReLU(FLAGS, conv_planes[1], conv_planes[2], stride=2)
        self.conv4 = conv_ReLU(FLAGS, conv_planes[2], conv_planes[3], stride=2)
        self.conv5 = conv_ReLU(FLAGS, conv_planes[3], conv_planes[4], stride=2)
        self.conv6 = conv_ReLU(FLAGS, conv_planes[4], conv_planes[5], stride=2)
        self.conv7 = conv_ReLU(FLAGS, conv_planes[5], conv_planes[6], stride=2)

        self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)

        if self.output_exp:
            upconv_planes = [256, 128, 64, 32, 16]
            self.upconv5 = upconv_ReLU(FLAGS, conv_planes[4],   upconv_planes[0], kernel_size=4)
            self.upconv4 = upconv_ReLU(FLAGS, upconv_planes[0], upconv_planes[1], kernel_size=4)
            self.upconv3 = upconv_ReLU(FLAGS, upconv_planes[1], upconv_planes[2], kernel_size=4)
            self.upconv2 = upconv_ReLU(FLAGS, upconv_planes[2], upconv_planes[3], kernel_size=4)
            self.upconv1 = upconv_ReLU(FLAGS, upconv_planes[3], upconv_planes[4], kernel_size=4)

            self.predict_mask4 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask3 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask2 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask1 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs, kernel_size=3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        if self.output_exp:
            out_upconv5 = self.upconv5(out_conv5  )[:, :, 0:out_conv4.size(2), 0:out_conv4.size(3)]
            out_upconv4 = self.upconv4(out_upconv5)[:, :, 0:out_conv3.size(2), 0:out_conv3.size(3)]
            out_upconv3 = self.upconv3(out_upconv4)[:, :, 0:out_conv2.size(2), 0:out_conv2.size(3)]
            out_upconv2 = self.upconv2(out_upconv3)[:, :, 0:out_conv1.size(2), 0:out_conv1.size(3)]
            out_upconv1 = self.upconv1(out_upconv2)[:, :, 0:input.size(2), 0:input.size(3)]

            exp_mask4 = nn.functional.sigmoid(self.predict_mask4(out_upconv4))
            exp_mask3 = nn.functional.sigmoid(self.predict_mask3(out_upconv3))
            exp_mask2 = nn.functional.sigmoid(self.predict_mask2(out_upconv2))
            exp_mask1 = nn.functional.sigmoid(self.predict_mask1(out_upconv1))
        else:
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None

        if self.training:
            return [exp_mask1, exp_mask2, exp_mask3, exp_mask4], pose
        else:
            return exp_mask1, pose



