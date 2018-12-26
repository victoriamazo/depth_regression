import torch
import torch.nn as nn
from models.layers import downsample_conv, predict_disp, conv_ReLU, upconv_ReLU, crop_like


class DispNetS(nn.Module):
    def __init__(self, FLAGS):
        super(DispNetS, self).__init__()

        self.alpha = 0.3  #as in Godard with edge-aware smoothing loss
        if hasattr(FLAGS, 'edge_aware') and FLAGS.edge_aware == 0:
            self.alpha = 10  # as in Zhou with plain smoothing loss
        self.beta = 0.01

        input = 3
        output = 1
        if FLAGS.stereo and hasattr(FLAGS, 'concat_LR') and FLAGS.concat_LR:
            print('input to DispNet is concatenated L and R images, output - L and R disparities')
            input = 6
            output = 2

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv(input, conv_planes[0], kernel_size=7, FLAGS=FLAGS)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5, FLAGS=FLAGS)
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2], FLAGS=FLAGS)
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3], FLAGS=FLAGS)
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4], FLAGS=FLAGS)
        self.conv6 = downsample_conv(conv_planes[4], conv_planes[5], FLAGS=FLAGS)
        self.conv7 = downsample_conv(conv_planes[5], conv_planes[6], FLAGS=FLAGS)

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv_ReLU(conv_planes[6],   upconv_planes[0], output_padding=1, FLAGS=FLAGS)
        self.upconv6 = upconv_ReLU(upconv_planes[0], upconv_planes[1], output_padding=1, FLAGS=FLAGS)
        self.upconv5 = upconv_ReLU(upconv_planes[1], upconv_planes[2], output_padding=1, FLAGS=FLAGS)
        self.upconv4 = upconv_ReLU(upconv_planes[2], upconv_planes[3], output_padding=1, FLAGS=FLAGS)
        self.upconv3 = upconv_ReLU(upconv_planes[3], upconv_planes[4], output_padding=1, FLAGS=FLAGS)
        self.upconv2 = upconv_ReLU(upconv_planes[4], upconv_planes[5], output_padding=1, FLAGS=FLAGS)
        self.upconv1 = upconv_ReLU(upconv_planes[5], upconv_planes[6], output_padding=1, FLAGS=FLAGS)

        self.iconv7 = conv_ReLU(upconv_planes[0] + conv_planes[5], upconv_planes[0], FLAGS=FLAGS)
        self.iconv6 = conv_ReLU(upconv_planes[1] + conv_planes[4], upconv_planes[1], FLAGS=FLAGS)
        self.iconv5 = conv_ReLU(upconv_planes[2] + conv_planes[3], upconv_planes[2], FLAGS=FLAGS)
        self.iconv4 = conv_ReLU(upconv_planes[3] + conv_planes[2], upconv_planes[3], FLAGS=FLAGS)
        self.iconv3 = conv_ReLU(output + upconv_planes[4] + conv_planes[1], upconv_planes[4], FLAGS=FLAGS)
        self.iconv2 = conv_ReLU(output + upconv_planes[5] + conv_planes[0], upconv_planes[5], FLAGS=FLAGS)
        self.iconv1 = conv_ReLU(output + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3], output=output)
        self.predict_disp3 = predict_disp(upconv_planes[4], output=output)
        self.predict_disp2 = predict_disp(upconv_planes[5], output=output)
        self.predict_disp1 = predict_disp(upconv_planes[6], output=output)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(nn.functional.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=True), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(nn.functional.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=True), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(nn.functional.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=True), x)
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1




























