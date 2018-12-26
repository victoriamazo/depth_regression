import torch
import torch.nn.functional as F
from torch.autograd import Variable


def spatial_transformer(img, coords):
    """A wrapper over binlinear_sampler(), taking absolute coords as input."""
    img_height = float(img.size(2))
    img_width = float(img.size(3))
    px = coords[:, :, :, :1]
    py = coords[:, :, :, 1:]

    # Normalize coordinates to [-1, 1] to send to _bilinear_sampler.
    px = px / (img_width - 1) * 2.0 - 1.0
    py = py / (img_height - 1) * 2.0 - 1.0
    output_img, mask = bilinear_sampler_2d(img, px, py)
    return output_img, mask


def bilinear_sampler_2d(img, x, y):
    """Perform bilinear sampling on image given list of x, y coordinates.
    Implements the differentiable sampling mechanism with bilinear kernel
    in https://arxiv.org/abs/1506.02025.

    x,y are tensors specifying normalized coordinates [-1, 1] to be sampled on image.
    For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in img,
    and (1, 1) in (x, y) corresponds to the bottom right pixel in img.

    Input:
      img: Batch of images (pytorch tensor) with shape [B, h, w, channels].
      x: Tensor of normalized x coordinates in [-1, 1], with shape [B, h, w, 1].
      y: Tensor of normalized y coordinates in [-1, 1], with shape [B, h, w, 1].
      name: Name scope for ops.
    Returns:
      Sampled image with shape [B, h, w, channels].
      Principled mask with shape [B, h, w, 1], type:float32.  A value of 1.0
        in the mask indicates that the corresponding coordinate in the sampled
        image is valid.
    """
    x = x.view(-1) #.cuda()
    y = y.view(-1) #.cuda()

    # Constants.
    batch_size = img.size(0)
    _, channels, height, width = img.size()

    x = x.type(torch.FloatTensor)
    y = y.type(torch.FloatTensor)
    height_f = float(height)
    width_f = float(width)
    zero = torch.zeros(1)
    max_y = img.size(2)-1  #.type(torch.cuda.LongTensor)
    max_x = img.size(3)-1  #.type(torch.cuda.LongTensor)

    # Scale indices from [-1, 1] to [0, width - 1] or [0, height - 1].
    x = (x + 1.0) * (width_f - 1.0) / 2.0
    y = (y + 1.0) * (height_f - 1.0) / 2.0

    # Compute the coordinates of the 4 pixels to sample from
    x0 = torch.floor(x).type(torch.cuda.LongTensor)
    x1 = x0 + 1
    y0 = torch.floor(y).type(torch.cuda.LongTensor)
    y1 = y0 + 1

    mask = ((x0 >= 0) * (x1 <= max_x)) * ((y0 >= 0) * (y1 <= max_y)) #tf.logical_and(tf.logical_and(x0 >= zero, x1 <= max_x), tf.logical_and(y0 >= zero, y1 <= max_y))
    mask = mask.type(torch.FloatTensor)

    x0 = torch.clamp(x0, 0, max_x)  #tf.clip_by_value(x0, zero, max_x)
    x1 = torch.clamp(x1, 0, max_x)  #tf.clip_by_value(x1, zero, max_x)
    y0 = torch.clamp(y0, 0, max_y)  #tf.clip_by_value(y0, zero, max_y)
    y1 = torch.clamp(y1, 0, max_y)  #tf.clip_by_value(y1, zero, max_y)
    x0 = x0.type(torch.FloatTensor)
    x1 = x1.type(torch.FloatTensor)
    y0 = y0.type(torch.FloatTensor)
    y1 = y1.type(torch.FloatTensor)
    dim2 = width
    dim1 = width * height

    # Create base index
    base = torch.arange(batch_size) * dim1  #tf.range(batch_size) * dim1
    base = base.view(-1, 1)  #tf.reshape(base, [-1, 1])
    base = base.repeat(1, height * width)   #tf.tile(base, [1, height * width])
    base = base.view(-1)  # base = tf.reshape(base, [-1])
    base = Variable(base)

    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1
    idx_a = idx_a.type(torch.cuda.LongTensor)
    idx_b = idx_b.type(torch.cuda.LongTensor)
    idx_c = idx_c.type(torch.cuda.LongTensor)
    idx_d = idx_d.type(torch.cuda.LongTensor)

    # Use indices to lookup pixels in the flat image and restore channels dim
    im_flat = img.contiguous().view(-1, channels)   #tf.reshape(img, tf.stack([-1, channels]))
    im_flat = im_flat.type(torch.FloatTensor).cuda()       #tf.to_float(im_flat)

    pixel_a = torch.gather(im_flat, 0, idx_a.repeat(channels).view(-1, channels))   #tf.gather(im_flat, idx_a)
    pixel_b = torch.gather(im_flat, 0, idx_b.repeat(channels).view(-1, channels))  #tf.gather(im_flat, idx_b)
    pixel_c = torch.gather(im_flat, 0, idx_c.repeat(channels).view(-1, channels))  #tf.gather(im_flat, idx_c)
    pixel_d = torch.gather(im_flat, 0, idx_d.repeat(channels).view(-1, channels))  #tf.gather(im_flat, idx_d)

    x1_f = x1.type(torch.FloatTensor)  #tf.to_float(x1)
    y1_f = y1.type(torch.FloatTensor)  #tf.to_float(y1)

    # And finally calculate interpolated values
    wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1).cuda()                 #tf.expand_dims( ((x1_f - x) * (y1_f - y)) , 1)
    wb = ((x1_f - x) * (1.0 - (y1_f - y))).unsqueeze(1).cuda()          #tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
    wc = ((1.0 - (x1_f - x)) * (y1_f - y)).unsqueeze(1).cuda()        #tf.expand_dims(((1.0 - (x1_f - x)) * (y1_f - y)), 1)
    wd = ((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))).unsqueeze(1).cuda()  #tf.expand_dims(((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))), 1)

    output = (wa * pixel_a) + (wb * pixel_b) + (wc * pixel_c) + (wd * pixel_d)  #tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
    output = output.view(batch_size, channels, height, width)  #tf.reshape(output, tf.stack([batch_size, height, width, channels]))
    mask = mask.view(batch_size, 1, height, width)  #tf.reshape(mask, tf.stack([batch_size, height, width, 1]))

    return output, mask


def bilinear_sampler_1d(input_images, x_offset, wrap_mode='edge'):
    '''
        Example of use:
            tgt_img_l - pytorch cuda tensor of shape (B, C, H, W)
            disp - pytorch cuda tensor of shape (B, 1, H, W)
            output = bilinear_sampler_1d(tgt_img_l, -disp)
    '''
    def _repeat(x, n_repeats):
        rep = x.unsqueeze(1).repeat(1, n_repeats)
        return rep.view(-1)

    def _interpolate(img, x, y):
        if _wrap_mode == 'border':
            _edge_size = 1
            img = F.pad(img, (0, 1, 1, 0), 'constant', 0)
            x = x + _edge_size
            y = y + _edge_size
        elif _wrap_mode == 'edge':
            _edge_size = 0
        else:
            return None

        x = torch.clamp(x, 0.0, _width_f - 1 + 2 * _edge_size)

        x0_f = torch.floor(x)
        y0_f = torch.floor(y)
        x1_f = x0_f + 1

        x0 = x0_f.type(torch.FloatTensor)
        y0 = y0_f.type(torch.FloatTensor)

        min_val = _width_f - 1 + 2 * _edge_size
        scalar = torch.FloatTensor([min_val]).cuda()

        x1 = torch.min(x1_f, scalar)
        x1 = x1.type(torch.FloatTensor)
        dim2 = _width + 2 * _edge_size
        dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)  #_width * _height
        base = _repeat(torch.arange(_num_batch) * dim1, _height * _width).type(torch.FloatTensor)

        base_y0 = base + y0 * dim2
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1
        idx_l = Variable(idx_l.type(torch.cuda.LongTensor))
        idx_r = Variable(idx_r.type(torch.cuda.LongTensor))

        im_flat = img.contiguous().view(-1, _num_channels)

        pix_l = torch.gather(im_flat, 0, idx_l.repeat(_num_channels).view(-1, _num_channels))  #torch.gather(im_flat, 0, idx_l).unsqueeze(1)
        pix_r = torch.gather(im_flat, 0, idx_r.repeat(_num_channels).view(-1, _num_channels))  #torch.gather(im_flat, 0, idx_r).unsqueeze(1)

        weight_l = Variable((x1_f - x).unsqueeze(1))
        weight_r = Variable((x - x0_f).unsqueeze(1))

        sampled_img = weight_l * pix_l + weight_r * pix_r

        return sampled_img


    def _transform(input_images, x_offset):
        '''input_image - pytorch cuda tensor of shape (B, C, H, W)
            x_offset (that is disparity) - pytorch cuda tensor of shape (B, 1, H, W)'''
        a = torch.linspace(0.0, _width_f - 1.0, _width).cuda()
        b = torch.linspace(0.0, _height_f - 1.0, _height).cuda()

        x_t = a.repeat(_height)
        y_t = b.repeat(_width, 1).t().contiguous().view(-1)

        x_t_flat = x_t.repeat(_num_batch, 1)
        y_t_flat = y_t.repeat(_num_batch, 1)

        x_t_flat = x_t_flat.view(-1)
        y_t_flat = y_t_flat.view(-1)

        x_t_flat = x_t_flat + x_offset.data.contiguous().view(-1) * _width_f

        input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

        output = input_transformed.view(_num_batch, _num_channels, _height, _width)

        return output


    _num_batch = input_images.size(0)
    _num_channels = input_images.size(1)
    _height = input_images.size(2)
    _width = input_images.size(3)

    _height_f = float(_height)
    _width_f = float(_width)

    _wrap_mode = wrap_mode

    output = _transform(input_images, x_offset)

    return output


