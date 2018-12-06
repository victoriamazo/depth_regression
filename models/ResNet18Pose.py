import torch.nn as nn
import torch
import torchvision.models as models

# list of all pretrained models
# model_names = sorted(name for name in models.__dict__
# if name.islower() and not name.startswith("__"))


class ResNet18Pose(nn.Module):
    def __init__(self, FLAGS):
        super(ResNet18Pose, self).__init__()

        assert FLAGS.height == 240 and FLAGS.width == 376, 'Height is not 240 or width is not 376'

        print('=> using pre-trained ResNet18 for pose training')
        original_model = models.__dict__['resnet18'](pretrained=True)

        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.FCs = nn.Sequential(nn.Linear(12288, 128), nn.Linear(128, 6),)

        # # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False


    def init_weights(self):
        pass


    def forward(self, target_image, ref_imgs):
        exp_mask1, exp_mask2, exp_mask3, exp_mask4 = None, None, None, None

        f_tgt = self.features(target_image)
        f_ref = self.features(ref_imgs[0])
        resnet_out = torch.cat((f_tgt, f_ref), 1)
        resnet_out = resnet_out.view(resnet_out.size(0), -1)
        pose = self.FCs(resnet_out)
        pose = 0.01 * pose.view(pose.size(0), 1, 6)

        if self.training:
            return [exp_mask1, exp_mask2, exp_mask3, exp_mask4], pose
        else:
            return exp_mask1, pose

















