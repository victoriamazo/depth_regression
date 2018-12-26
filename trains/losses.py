from utils.auxiliary import generate_mask_tensor, generate_image_left, generate_image_right

import torch
from torch import nn



def compute_loss(var_dict_t, disp_l, depth_l, disp_r, depth_r, loss_weights_dict, loss_dict, loss_params_dict):
    '''
    :param var_dict_t: dictionary of all variables (pytorch tensors)
    :param disp_l: predicted left disparity
    :param disp_r: predicted right disparity
    :param loss_weights_dict: dictionary of loss weights
    :param loss_dict: dictionary of all losses
    :param loss_params_dict: dictionary of all auxiliary parameters for calculation losses (except loss weights)
    :return:
         - 'DS': depth supervised loss
         - 'O': left (and right if stereo) occlusion loss
         - 'S': left (and right if stereo) disparity smoothing loss
         - 'DC': disparity consistency loss
    '''

    # depth supervised loss
    if 'loss_DS' in loss_dict and loss_params_dict['mode'] == 'train' and loss_params_dict['with_gt_depth']:
        loss_dict['loss_DS'] = depth_supervised_loss(depth_l[0], var_dict_t['gt_depth_l'], loss_params_dict)
    elif 'loss_DS' in loss_dict:
        del loss_dict['loss_DS']

    # left disparity smoothing loss
    if 'loss_S' in loss_dict:
        loss_dict['loss_S'] = smooth_loss(disp_l, var_dict_t['tgt_img_l'], loss_params_dict)

    # occlusion loss
    if 'loss_O' in loss_dict:
        loss_dict['loss_O'] = occlusion_loss(depth_l[0])

    # disparity consistency loss
    if 'loss_DC' in loss_dict:
        assert disp_r is not None
        loss_dict['loss_DC'] = disp_consistency_loss(disp_l, disp_r, var_dict_t['tgt_img_l'].size(),
                                                     loss_params_dict['upscaling'], loss_params_dict['disp_norm'])

    if loss_params_dict['stereo']:
        # depth supervised loss
        if 'loss_DS' in loss_dict and loss_params_dict['mode'] == 'train' and loss_params_dict['with_gt_depth']:
            loss_dict['loss_DS'] = depth_supervised_loss(depth_r[0], var_dict_t['gt_depth_r'], loss_params_dict)
        elif 'loss_DS' in loss_dict:
            del loss_dict['loss_DS']

        # right disparity smoothing loss
        if 'loss_S' in loss_dict:
            loss_dict['loss_S'] += smooth_loss(disp_r, var_dict_t['tgt_img_r'], loss_params_dict)

        # occlusion loss
        if 'loss_O' in loss_dict:
            loss_dict['loss_O'] += occlusion_loss(depth_r[0])

    # compute total loss
    loss = 0
    loss_names = ['tot_loss']
    losses_list = [torch.zeros(1)]
    for loss_name, loss_value in loss_dict.items():
        loss += loss_weights_dict['w_{}'.format(loss_name[5:])] * loss_value
        loss_names.append(loss_name)
        losses_list.append(loss_value)
    losses_list[0] = loss                 # total loss

    return losses_list, loss_names


def depth_supervised_loss(depth, gt_depth, loss_params_dict):
    '''Input:
         - depth (B,1,in_h,in_w) (tensor) - depth from the finest layer (at the same resolution as the input image)
         - gt_depth (B,h,w) (tensor) - GT depth at the same resolution as depth'''
    gt_depth = gt_depth / 255 * 80
    depth = torch.squeeze(depth)
    assert depth.size() == gt_depth.size()
    b, h, w = gt_depth.size()
    gt_depth = gt_depth.cuda()

    if loss_params_dict['disp_norm']:
        if b == 1:
            normalized_disp = depth / depth.mean()
        else:
            normalized_disp_means = torch.mean(torch.mean(depth, dim=1), dim=1)
            normalized_disp = torch.stack([depth[i, :, :] / normalized_disp_means[i] for i in range(depth.size(0))])
    else:
        normalized_disp = depth

    normalized_disp = torch.clamp(normalized_disp, 1e-3, loss_params_dict['max_depth'])
    # mask = generate_mask_tensor(gt_depth, max_depth=loss_params_dict['max_depth'])
    # normalized_disp = torch.masked_select(normalized_disp, mask)
    # gt_depth = torch.masked_select(gt_depth, mask)

    # reversed Huber loss
    x = torch.abs(normalized_disp - gt_depth)
    c = 0.2 * torch.max(x)
    DS_loss_L1 = torch.masked_select(x, x <= c)
    DS_loss_L2 = torch.masked_select(x, x > c)
    DS_loss = DS_loss_L1.mean() + ((DS_loss_L2**2 + c**2)/(2*c)).mean()

    return DS_loss


def smooth_loss(disp, tgt_img, loss_params_dict):
    '''- downscaling of the tgt and ref imgs to disp size (default) or
            upscaling of disp to the tgt img size
        - optional disparity normalization
        - edge-aware smoothing loss or
            regular smoothing loss (as in Zhou et al.)'''
    assert 'disp_norm' in loss_params_dict
    assert 'upscaling' in loss_params_dict
    assert 'edge_aware' in loss_params_dict
    disp_norm = loss_params_dict['disp_norm']
    upscaling = loss_params_dict['upscaling']
    edge_aware = loss_params_dict['edge_aware']

    def gradient(pred):
        D_dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]
        D_dx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
        return D_dx, D_dy

    if type(disp) not in [tuple, list]:
        disp = [disp]

    b, _, h, w = tgt_img.size()
    im_dx, im_dy = gradient(tgt_img)
    weights_x = torch.exp(-torch.mean(im_dx.abs(), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(im_dy.abs(), 1, keepdim=True))
    loss, weight = 0, 1.0
    for s, scaled_disp in enumerate(disp):
        # downscale = h / scaled_disp.size(2)
        if disp_norm:
            if b == 1:
                scaled_disp /= scaled_disp.mean()
            else:
                normalized_disp_means = torch.mean(torch.mean(torch.mean(scaled_disp, dim=1), dim=1), dim=1)
                scaled_disp = torch.stack([scaled_disp[i, :, :, :] / normalized_disp_means[i] for i in range(scaled_disp.size(0))])

        if upscaling:
            if scaled_disp.size(2) < h or scaled_disp.size(3) < w:
                scaled_disp = torch.nn.functional.interpolate(scaled_disp, (h, w), align_corners=False)
        else:
            weights_x = nn.functional.adaptive_avg_pool2d(weights_x, (scaled_disp.size(2), scaled_disp.size(3)-1))
            weights_y = nn.functional.adaptive_avg_pool2d(weights_y, (scaled_disp.size(2)-1, scaled_disp.size(3)))

        if edge_aware:
            disp_dx, disp_dy = gradient(scaled_disp)
            smoothness_x = disp_dx * weights_x
            smoothness_y = disp_dy * weights_y
            loss += smoothness_x.abs().mean() + smoothness_y.abs().mean()
        else:
            dx, dy = gradient(scaled_disp)
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            loss += ((dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight)/(2**s) #/downscale
            weight /= 2.83  # 2sqrt(2)

    return loss


def occlusion_loss(depth):
    '''Input:
         - depth (B,1,in_h,in_w) (tensor) - depth from the finest layer (at the same resolution as the input image)
    '''
    occl_loss = torch.abs(depth).mean()

    return occl_loss


def disp_consistency_loss(disp_l, disp_r, img_size, upscaling=False, disp_norm=False):
    b, _, h, w = img_size

    def one_scale(scaled_disp_l, scaled_disp_r):

        # normalize disparity image by the mean of the disparity image (not by the batch mean)
        if disp_norm:
            if b == 1:
                normalized_disp_l = scaled_disp_l / scaled_disp_l.mean()
                normalized_disp_r = scaled_disp_r / scaled_disp_r.mean()
            else:
                normalized_disp_means_l = torch.mean(torch.mean(torch.mean(scaled_disp_l, dim=1), dim=1), dim=1)
                normalized_disp_l = torch.stack([scaled_disp_l[i,:,:,:]/normalized_disp_means_l[i] for i in
                                                 range(scaled_disp_l.size(0))])
                normalized_disp_means_r = torch.mean(torch.mean(torch.mean(scaled_disp_r, dim=1), dim=1), dim=1)
                normalized_disp_r = torch.stack([scaled_disp_r[i, :, :, :] / normalized_disp_means_r[i] for i in
                                                 range(scaled_disp_r.size(0))])
        else:
            normalized_disp_l = scaled_disp_l
            normalized_disp_r = scaled_disp_r

        # upscale disparity
        if upscaling and (normalized_disp_l.size(2) < h or normalized_disp_l.size(3) < w):
            upscaled_disp_l = torch.nn.functional.interpolate(normalized_disp_l, (h, w), align_corners=False)
            upscaled_disp_r = torch.nn.functional.interpolate(normalized_disp_r, (h, w), align_corners=False)
        else:
            upscaled_disp_l = normalized_disp_l
            upscaled_disp_r = normalized_disp_r

        right_to_left_disp = generate_image_left(upscaled_disp_r, upscaled_disp_l)
        left_to_right_disp = generate_image_right(upscaled_disp_l, upscaled_disp_r)

        dc_loss_l = (right_to_left_disp - upscaled_disp_l).abs().mean()
        dc_loss_r = (left_to_right_disp - upscaled_disp_r).abs().mean()

        return dc_loss_l + dc_loss_r

    if type(disp_l) not in [list, tuple]:
        disp_l = [disp_l]
    if type(disp_r) not in [list, tuple]:
        disp_r = [disp_r]

    DC_loss = 0
    # loss is a sum of losses over 4 scales
    for d_l, d_r in zip(disp_l, disp_r):
        DC_loss += one_scale(d_l, d_r)

    return DC_loss



































