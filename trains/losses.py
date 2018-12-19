from utils.inverse_warp import inverse_warp, generate_image_left, generate_image_right
from utils.auxiliary import generate_mask_tensor

import torch
from torch import nn
from torch.autograd import Variable


def compute_loss(var_dict_t, disp_l, depth_l, disp_r, explainability_mask, pose, loss_weights_dict, loss_dict, loss_params_dict):
    '''
    :param var_dict_t: dictionary of all variables (pytorch tensors)
    :param disp_l: predicted left disparity
    :param disp_r: predicted right disparity
    :param explainability_mask: predicted explainability mask
    :param pose: predicted pose
    :param loss_weights_dict: dictionary of loss weights
    :param loss_dict: dictionary of all losses
    :param loss_params_dict: dictionary of all auxiliary parameters for calculation losses (except loss weights)
    :return:
         - '12': temporal photometric loss
         - '12S': temporal SSIM loss
         - 'E': temporal explainability loss
         - 'S': left (and right if stereo) disparity smoothing loss
         - 'DC': disparity consistency loss
         - 'DS': depth supervised loss
         - 'LR': left->right photometric loss
         - 'RL': right->left photometric loss
         - 'LRS': left (and right if stereo) SSIM loss
    '''
    # temporal photometric loss
    if 'loss_12' in loss_dict:
        ssim = False
        if 'loss_12S' in loss_dict:
            ssim = True
        loss_dict['loss_12'], SSIM_loss = photometric_reconstruction_loss(var_dict_t['tgt_img_l'], var_dict_t['ref_imgs_l'],
                                                    var_dict_t['intrinsics_l'], var_dict_t['intrinsics_l_inv'], disp_l,
                                                    explainability_mask, pose, loss_params_dict, ssim=ssim, LR=False)
        # temporal SSIM loss
        if 'loss_12S' in loss_dict:
            loss_dict['loss_12S'] = SSIM_loss

    # temporal explainability loss
    if 'loss_E' in loss_dict:
        loss_dict['loss_E'] = explainability_loss(explainability_mask)

    # left disparity smoothing loss
    if 'loss_S' in loss_dict:
        loss_dict['loss_S'] = smooth_loss(disp_l, var_dict_t['tgt_img_l'], loss_params_dict)

    # disparity consistency loss
    if 'loss_DC' in loss_dict:
        assert disp_r is not None
        loss_dict['loss_DC'] = disp_consistency_loss(disp_l, disp_r, var_dict_t['tgt_img_l'].size(),
                                                     loss_params_dict['upscaling'], loss_params_dict['disp_norm'])

    # depth supervised loss
    if 'loss_DS' in loss_dict and loss_params_dict['mode'] == 'train' and loss_params_dict['with_gt_depth']:
        loss_dict['loss_DS'] = depth_supervised_loss(depth_l[0], var_dict_t['gt_depth_l'], loss_params_dict)
    elif 'loss_DS' in loss_dict:
        del loss_dict['loss_DS']

    if loss_params_dict['stereo']:
        # LR photometric loss
        if 'loss_LR' in loss_dict:
            ssim = False
            if 'loss_LRS' in loss_dict:
                ssim = True
            loss_dict['loss_LR'], SSIM_loss = photometric_reconstruction_loss(var_dict_t['tgt_img_l'],
                                                    [var_dict_t['tgt_img_r']], var_dict_t['intrinsics_l'],
                                                    var_dict_t['intrinsics_l_inv'], disp_l, explainability_mask,
                                                    var_dict_t['T_LR'], loss_params_dict, ssim=ssim, LR=True, left=True)
            # LR SSIM loss
            if 'loss_LRS' in loss_dict:
                loss_dict['loss_LRS'] = SSIM_loss

            # RL photometric loss
            if 'loss_RL' in loss_dict:
                assert disp_r is not None
                loss_dict['loss_RL'], SSIM_loss = photometric_reconstruction_loss(var_dict_t['tgt_img_r'],
                                                    [var_dict_t['tgt_img_l']], var_dict_t['intrinsics_r'],
                                                    var_dict_t['intrinsics_r_inv'], disp_r, explainability_mask,
                                                    var_dict_t['T_LR'], loss_params_dict, ssim=ssim, LR=True, left=False)
                # RL SSIM loss
                if 'loss_LRS' in loss_dict:
                    loss_dict['loss_LRS'] += SSIM_loss

                # right disparity smoothing loss
                if 'loss_S' in loss_dict:
                    loss_dict['loss_S'] += smooth_loss(disp_r, var_dict_t['tgt_img_r'], loss_params_dict)
    else:
        if 'loss_LR' in loss_dict:
            del loss_dict['loss_LR']
        if 'loss_LRS' in loss_dict:
            del loss_dict['loss_LRS']
        if 'loss_RL' in loss_dict:
            del loss_dict['loss_RL']

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


def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, intrinsics_inv, disp, explainability_mask, pose,
                                    loss_params_dict, padding_mode='zeros', ssim=False, LR=False, left=True):
    '''photometric loss with:
        - downscaling of the tgt and ref imgs to disp size (default) or
            upscaling of disp to the tgt img size
        - optional disparity normalization
    '''
    assert 'stereo' in loss_params_dict
    assert 'disp_norm' in loss_params_dict
    assert 'upscaling' in loss_params_dict
    assert 'rotation_mode' in loss_params_dict
    stereo = loss_params_dict['stereo']
    disp_norm = loss_params_dict['disp_norm']
    upscaling = loss_params_dict['upscaling']
    rotation_mode = loss_params_dict['rotation_mode']

    def one_scale_upscaling(scaled_disp, explainability_mask):
        b, _, h, w = tgt_img.size()
        assert(explainability_mask is None or scaled_disp.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        # normalize disparity image by the mean of the disparity image (not by the batch mean)
        if disp_norm:
            if b == 1:
                normalized_disp = scaled_disp / scaled_disp.mean()
            else:
                normalized_disp_means = torch.mean(torch.mean(torch.mean(scaled_disp, dim=1), dim=1), dim=1)
                normalized_disp = torch.stack([scaled_disp[i,:,:,:]/normalized_disp_means[i] for i in range(scaled_disp.size(0))])  #scaled_disp / scaled_disp.mean()
        else:
            normalized_disp = scaled_disp

        # upscaling of disparity (and explainability mask)
        if normalized_disp.size(2) < h or normalized_disp.size(3) < w:
            upscaled_disp = torch.nn.functional.upsample(normalized_disp, (h, w), mode='bilinear')
            if explainability_mask is not None:
                explainability_mask = torch.nn.functional.upsample(explainability_mask, (h, w), mode='bilinear')
        else:
            upscaled_disp = normalized_disp
        upscaled_depth = 1 / upscaled_disp

        reconstruction_loss, ssim_loss = 0, 0
        for i, ref_img in enumerate(ref_imgs):
            current_pose = pose[:, i]                                               # pose [b, num_src_imgs, 6]

            if LR:
                if left:
                    ref_img_warped = generate_image_left(ref_img, upscaled_disp)
                else:
                    ref_img_warped = generate_image_right(ref_img, upscaled_disp)
            else:
                ref_img_warped = inverse_warp(ref_img, upscaled_depth[:, 0], current_pose, intrinsics, intrinsics_inv,
                                                rotation_mode, padding_mode)
            out_of_bound = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            tgt_img_bound = tgt_img * out_of_bound
            ref_img_warped_bound = ref_img_warped * out_of_bound

            if explainability_mask is not None:
                tgt_img_bound = tgt_img_bound * explainability_mask[:, i:i+1].expand_as(tgt_img_bound)    #repeat for all 3 channels
                ref_img_warped_bound = ref_img_warped_bound * explainability_mask[:, i:i+1].expand_as(ref_img_warped_bound)    #repeat for all 3 channels

            diff = tgt_img_bound - ref_img_warped_bound
            reconstruction_loss += diff.abs().mean()
            assert((reconstruction_loss == reconstruction_loss).data[0] == 1)

            if ssim:
                ssim_loss += SSIM(tgt_img_bound, ref_img_warped_bound)

        return reconstruction_loss, ssim_loss

    def one_scale_downscaling(scaled_disp, explainability_mask):
        assert(explainability_mask is None or scaled_disp.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))
        reconstruction_loss, ssim_loss = 0, 0
        b, _, h, w = scaled_disp.size()
        downscale = tgt_img.size(2)/h

        if disp_norm:
            if b == 1:
                scaled_disp /= scaled_disp.mean()
            else:
                normalized_disp_means = torch.mean(torch.mean(torch.mean(scaled_disp, dim=1), dim=1), dim=1)
                scaled_disp = torch.stack([scaled_disp[i,:,:,:]/normalized_disp_means[i] for i in range(b)])
        depth = 1 / scaled_disp

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]                                               # pose [b, num_src_imgs, 6]

            if LR:
                # baseline = pose.view(-1)[0]
                # f = intrinsics.view(-1)[0]
                disp = 1 / depth             #baseline * f / depth
                if left:
                    ref_img_warped = generate_image_left(ref_img, disp)
                else:
                    ref_img_warped = generate_image_right(ref_img, disp)
            else:
                ref_img_warped = inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv,
                                                rotation_mode, padding_mode)
            out_of_bound = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            tgt_img_scaled_bound = tgt_img_scaled * out_of_bound
            ref_img_warped_bound = ref_img_warped * out_of_bound

            if explainability_mask is not None:
                tgt_img_scaled_bound = tgt_img_scaled_bound * explainability_mask[:, i:i+1].expand_as(tgt_img_scaled_bound)    #repeat for all 3 channels
                ref_img_warped_bound = ref_img_warped_bound * explainability_mask[:, i:i+1].expand_as(ref_img_warped_bound)    #repeat for all 3 channels

            diff = tgt_img_scaled_bound - ref_img_warped_bound   #(tgt_img_scaled - ref_img_warped) * out_of_bound
            reconstruction_loss += diff.abs().mean()
            assert((reconstruction_loss == reconstruction_loss).data[0] == 1)

            if ssim:
                ssim_loss += SSIM(ref_img_warped_bound, ref_img_warped_bound)

        return reconstruction_loss, ssim_loss

    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(disp) not in [list, tuple]:
        disp = [disp]

    L1_loss, SSIM_loss = 0, 0
    # loss is a sum of losses over 4 scales
    for d, mask in zip(disp, explainability_mask):
        if upscaling:
            l1_loss, ssim_loss = one_scale_upscaling(d, mask)
        else:
            l1_loss, ssim_loss = one_scale_downscaling(d, mask)
        L1_loss += l1_loss
        SSIM_loss += ssim_loss
    return L1_loss, SSIM_loss


def SSIM(tgt_img, ref_img):
    C1 = 0.01**2
    C2 = 0.03**2
    avg_pooling = torch.nn.AvgPool2d(3, stride=1)

    # loss is a sum of losses over 4 scales
    mu_x = avg_pooling(tgt_img)
    mu_y = avg_pooling(ref_img)

    sigma_x = avg_pooling(tgt_img**2) - mu_x**2
    sigma_y = avg_pooling(ref_img**2) - mu_y**2
    sigma_xy = avg_pooling(tgt_img * ref_img) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d
    SSIM_loss = torch.clamp((1 - SSIM) / 2, min=0, max=1).mean()

    return SSIM_loss


def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]                                           # list of 4 scaled masks [b,2,h',w']
    loss = 0
    for mask_scaled in mask:
        ones_var = Variable(torch.ones(1)).expand_as(mask_scaled).type_as(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss


def smooth_loss(disp, tgt_img, loss_params_dict):
    # TODO: to correct bug
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
    for scaled_disp in disp:
        # downscale = h / scaled_disp.size(2)
        if disp_norm:
            if b == 1:
                scaled_disp /= scaled_disp.mean()
            else:
                normalized_disp_means = torch.mean(torch.mean(torch.mean(scaled_disp, dim=1), dim=1), dim=1)
                scaled_disp = torch.stack([scaled_disp[i, :, :, :] / normalized_disp_means[i] for i in range(scaled_disp.size(0))])

        if upscaling:
            if scaled_disp.size(2) < h or scaled_disp.size(3) < w:
                scaled_disp = torch.nn.functional.upsample(scaled_disp, (h, w), mode='bilinear')
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
            loss += ((dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight) #/downscale
            weight /= 2.83  # 2sqrt(2)
    return loss


def plain_smooth_loss(pred_disp):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_disp) not in [tuple, list]:
        pred_disp = [pred_disp]

    loss = 0
    weight = 1.

    for scaled_disp in pred_disp:
        dx, dy = gradient(scaled_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.83  # 2sqrt(2)
    return loss


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
            upscaled_disp_l = torch.nn.functional.upsample(normalized_disp_l, (h, w), mode='bilinear')
            upscaled_disp_r = torch.nn.functional.upsample(normalized_disp_r, (h, w), mode='bilinear')
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
    mask = generate_mask_tensor(gt_depth, max_depth=loss_params_dict['max_depth'])
    normalized_disp = torch.masked_select(normalized_disp, mask)
    gt_depth = torch.masked_select(gt_depth, mask)

    # reversed Huber loss
    x = torch.abs(normalized_disp - gt_depth)
    # DS_loss = x.mean()
    c = 0.2 * torch.max(x)
    DS_loss_L1 = torch.masked_select(x, x <= c)
    DS_loss_L2 = torch.masked_select(x, x > c)
    DS_loss = DS_loss_L1.mean() + ((DS_loss_L2**2 + c**2)/(2*c)).mean()

    return DS_loss


def loss_regression(pred_pose_delta, gt_pose_delta, beta=0.5):
    '''
        Regression loss between predicted and GT posisions/angles.

    Input:
        - pred_pose_delta - pytorch tensor of shape (B, 6)
        - gt_pose_delta - pytorch tensor of shape (B, 6)
        - beta - coefficient for angles loss
    '''

    loss_pos = torch.norm(pred_pose_delta[:, :3] - gt_pose_delta[:, :3])
    loss_ang = torch.norm(pred_pose_delta[:, 3:] - gt_pose_delta[:, 3:])
    loss = loss_pos + beta * loss_ang

    return loss


def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]































