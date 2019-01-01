from models.layers import crop_like, sigm, weights_init, choose_decoder

import torch
import torch.nn as nn
import torchvision.models


class ResNet(nn.Module):
    def __init__(self, FLAGS, layers=18, decoder='upproj', in_channels=3, out_channels=1, pretrained=True):

        if hasattr(FLAGS, 'num_layers'):
            layers = FLAGS.num_layers
        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet, self).__init__()
        # if FLAGS.load_ckpt != '':
        pretrained=False
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        self.alpha = 0.3  #as in Godard with edge-aware smoothing loss
        if hasattr(FLAGS, 'edge_aware') and FLAGS.edge_aware == 0:
            self.alpha = 10  # as in Zhou with plain smoothing loss
        self.beta = 0.01
        if FLAGS.stereo and hasattr(FLAGS, 'concat_LR') and FLAGS.concat_LR:
            print('input to ResNet is concatenated L and R images, output - L and R disparities')
            in_channels *= 2
            out_channels *= 2

        self.in_channels = in_channels
        in_channels_conv1 = 64
        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, in_channels_conv1, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(in_channels_conv1)

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

        self.conv_u4 = nn.Conv2d(num_channels, num_channels//2, kernel_size=1, bias=False)
        self.bn_u4 = nn.BatchNorm2d(num_channels // 2)
        self.conv_u3 = nn.Conv2d(num_channels//2, num_channels//4, kernel_size=1, bias=False)
        self.bn_u3 = nn.BatchNorm2d(num_channels // 4)
        self.conv_u2 = nn.Conv2d(num_channels//4, num_channels // 8, kernel_size=1, bias=False)
        self.bn_u2 = nn.BatchNorm2d(num_channels // 8)
        self.conv_u1 = nn.Conv2d(num_channels//16+in_channels_conv1, num_channels // 8, kernel_size=1, bias=False)
        self.bn_u1 = nn.BatchNorm2d(num_channels // 8)

        self.decoder4 = choose_decoder(decoder, num_channels)
        self.decoder3 = choose_decoder(decoder, num_channels//2)
        self.decoder2 = choose_decoder(decoder, num_channels//4)
        self.decoder1 = choose_decoder(decoder, num_channels//8)
        self.decoder0 = choose_decoder(decoder, num_channels//8)

        # setting bias=true doesn't improve accuracy
        self.conv_d4 = nn.Conv2d(num_channels//4,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv_d3 = nn.Conv2d(num_channels//8,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv_d2 = nn.Conv2d(num_channels//16,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv_d1 = nn.Conv2d(num_channels//16,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.sigm = sigm()


    def init_weights(self):
        if self.in_channels == 3:
            weights_init(self.conv1)
            weights_init(self.bn1)
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder1.apply(weights_init)
        self.decoder2.apply(weights_init)
        self.decoder3.apply(weights_init)
        self.decoder4.apply(weights_init)
        self.conv_d1.apply(weights_init)
        self.conv_d2.apply(weights_init)
        self.conv_d3.apply(weights_init)
        self.conv_d4.apply(weights_init)


    def forward(self, x):
        # encoder (resnet)
        conv1 = self.conv1(x)
        conv1_bn = self.bn1(conv1)
        conv1_bn_relu = self.relu(conv1_bn)
        pool1 = self.maxpool(conv1_bn_relu)
        layer1 = self.layer1(pool1)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        conv2 = self.conv2(layer4)
        conv2_bn = self.bn2(conv2)

        # decoder (5 blocks)
        up4 = self.decoder4(conv2_bn)
        up4 = crop_like(up4, layer3)
        concat3 = torch.cat((up4, layer3), 1)
        conv_u4 = self.conv_u4(concat3)
        conv_u4_bn = self.bn_u4(conv_u4)

        up3 = self.decoder3(conv_u4_bn)
        if self.training:
            conv_d4 = self.conv_d4(up3)
            disp4 = self.sigm(conv_d4)
        up3 = crop_like(up3, layer2)
        concat2 = torch.cat((up3, layer2), 1)
        conv_u3 = self.conv_u3(concat2)
        conv_u3_bn = self.bn_u3(conv_u3)

        up2 = self.decoder2(conv_u3_bn)
        if self.training:
            conv_d3 = self.conv_d3(up2)
            disp3 = self.sigm(conv_d3)
        up2 = crop_like(up2, layer1)
        concat1 = torch.cat((up2, layer1), 1)
        conv_u2 = self.conv_u2(concat1)
        conv_u2_bn = self.bn_u2(conv_u2)

        up1 = self.decoder1(conv_u2_bn)
        if self.training:
            conv_d2 = self.conv_d2(up1)
            disp2 = self.sigm(conv_d2)
        up1 = crop_like(up1, conv1_bn_relu)
        concat0 = torch.cat((up1, conv1_bn_relu), 1)
        conv_u1 = self.conv_u1(concat0)
        conv_u1_bn = self.bn_u1(conv_u1)

        up0 = self.decoder0(conv_u1_bn)
        conv_d1 = self.conv_d1(up0)
        disp1 = self.sigm(conv_d1)


        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1




















