import numpy as np
import transforms3d.euler as euler                         # pip install transforms3d
import math
import functools

from utils.bilinear_sampler import bilinear_sampler_1d

import torch
from torch.autograd import Variable

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w)).type_as(depth)   # [1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w)).type_as(depth)   # [1, H, W]
    ones = Variable(torch.ones(1,h,w)).type_as(depth)                                   # [1, H, W]

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)                         # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)                                                                          # [1, 3, H, W]
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).contiguous().view(b, 3, -1)      # [B, 3, H*W]
    # batch multiplication  [B,3,3]x[B,3,H*W] = [B, 3, H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)                          # [B, 3, H, W]
    return cam_coords * depth.unsqueeze(1)                                                          # [B, 3, H, W]


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 3, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 3]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)             # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)         # [B, 3, H*W]
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr                     # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1        # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1                            # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combination of im and gray
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)     # [B, H*W, 2]
    return pixel_coords.view(b,h,w,2)                       # [B, H, W, 2]


def inverse_warp(img, depth, transformation, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode='zeros', detach=True):
    """
    Inverse warp a source image to the target image plane.

    Args (all are pytorch tensors):
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        transformation: 6DoF transformation parameters (either pose from target to source
            or left to right)-- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(transformation, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')
    assert(intrinsics_inv.size() == intrinsics.size())
    batch_size, _, img_height, img_width = img.size()
    transformation_mat = pose_vec2mat(transformation, rotation_mode, detach=detach)                        # [B,3,4]

    # transform coordinates in the pixel frame to the camera frame
    cam_coords = pixel2cam(depth, intrinsics_inv)                       # [B,3,H,W]

    # get projection matrix for tgt camera frame to source pixel frame (batch multiplication)
    proj_cam_to_src_pixel = intrinsics.bmm(transformation_mat)                    # [B,3,3]x[B,3,4] = [B, 3, 4]

    # transform coordinates in the camera frame to the pixel frame
    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode)  # [B,H,W,2]

    # biliniar sampler
    projected_img = torch.nn.functional.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)

    return projected_img                                                # [B,3,H,W]


def generate_image_left(img_r, disp_l):
    return bilinear_sampler_1d(img_r, -disp_l)


def generate_image_right(img_l, disp_r):
    return bilinear_sampler_1d(img_l, disp_r)


######################### auxiliary functions for pytorch tensor #######################

def euler2mat(angle, detach=True):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle (pytorch tensor): rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    if detach:
        zeros = z.detach()*0
        ones = zeros.detach()+1
    else:
        zeros = z * 0
        ones = zeros + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).view(B, 3, 3)

    rotMat = xmat.bmm(ymat).bmm(zmat)
    return rotMat


def quat2mat(quat, detach=True):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat (pytorch tensor): first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    if detach:
        norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    else:
        norm_quat = torch.cat([quat[:,:1]*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler', detach=True):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec (pytorch tensor): 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4] (pytorch tensor)
    """
    translation = vec[:, :3].unsqueeze(-1)                          # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot, detach=detach)                                    # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot, detach=detach)                                     # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)        # [B, 3, 4]
    return transform_mat



######################### auxiliary functions for numpy arrays #######################

def quat2euler_arr(q):
    '''
        Convert euler angles to quaternion.
         - q is a np array of shape (N, 4).
    '''
    assert q.shape[1] == 4
    N = q.shape[0]
    azimut = np.zeros(N)
    roll = np.zeros(N)
    pitch = np.zeros(N)
    for i in range(N):
        roll[i], azimut[i], pitch[i] = euler.quat2euler(q[i, :], 'szyx')
    rz = azimut
    ry = roll
    rx = pitch
    return ry, rz, rx


def isRotationMatrix(R):
    '''
        Checks if a matrix is a valid rotation matrix.
        Input: a 3x3 np array.
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    '''
        Calculates rotation matrix, which should be a 3x3 np array,
        to euler angles.
    '''
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0
    return np.array([rx, ry, rz])


def euler2mat_np(z=0, y=0, x=0, isRadian=True):
    ''' Return matrix for rotations around z, y and x axes
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    M : array shape (3,3)
         Rotation matrix giving same rotation as for given angles
    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True
    The output rotation matrix is equal to the composition of the
    individual rotations
    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True
    You can specify rotations by named arguments
    >>> np.all(M3 == euler2mat(x=xrot))
    True
    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.
    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)
    Rotations are counter-clockwise.
    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True
    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''

    if not isRadian:
        z = ((np.pi)/180.) * z
        y = ((np.pi)/180.) * y
        x = ((np.pi)/180.) * x
    assert z>=(-np.pi) and z < np.pi, 'Inapprorpriate z: %f' % z
    assert y>=(-np.pi) and y < np.pi, 'Inapprorpriate y: %f' % y
    assert x>=(-np.pi) and x < np.pi, 'Inapprorpriate x: %f' % x

    Ms = []
    if z:
            cosz = math.cos(z)
            sinz = math.sin(z)
            Ms.append(np.array(
                            [[cosz, -sinz, 0],
                             [sinz, cosz, 0],
                             [0, 0, 1]]))
    if y:
            cosy = math.cos(y)
            siny = math.sin(y)
            Ms.append(np.array(
                            [[cosy, 0, siny],
                             [0, 1, 0],
                             [-siny, 0, cosy]]))
    if x:
            cosx = math.cos(x)
            sinx = math.sin(x)
            Ms.append(np.array(
                            [[1, 0, 0],
                             [0, cosx, -sinx],
                             [0, sinx, cosx]]))
    if Ms:
            return functools.reduce(np.dot, Ms[::-1])
    return np.eye(3)


def pose_vec_to_mat(vec):
    ''' vec is a 6dof np array of shape (6)'''
    tx = vec[0]
    ty = vec[1]
    tz = vec[2]
    trans = np.array([tx, ty, tz]).reshape((3,1))
    rot = euler2mat_np(vec[5], vec[4], vec[3])
    Tmat = np.concatenate((rot, trans), axis=1)
    hfiller = np.array([0, 0, 0, 1]).reshape((1,4))
    Tmat = np.concatenate((Tmat, hfiller), axis=0)
    return Tmat


def convert_and_change_coordinate_system(poses, new_coord_index=0):
    '''transform to 0-th-frame coordinate system
        poses is np array (snippet) of shape [seq_lenth, 6]'''
    coord_pose = pose_vec_to_mat(poses[new_coord_index])   # M_0  (4, 4)

    out = []
    for pose_vec in poses:
        pose = pose_vec_to_mat(pose_vec)                   # (4, 4)
        pose = np.dot(coord_pose, np.linalg.inv(pose))     # M_0 * M_i^-1
        out.append(pose)

    return out


def merge_sequences_poses(ps_arr):
    '''
        (from Zhou_tf github issue "Testing Pose_net Issue")
        ps - sequence of 'seq_len' poses of shape [seq_len, 6] (predicted pose vector (np array) from network output,
            e.g. seq_len=5)
        ps_arr (list) - list of 'seq_len' pose sequences with a single overlapping pose (last element)
        Result - array of pose transformations relative to the 0th frame:
            [I, T_01, T_02, T_03, T_04, T_05, T_06, T_07, T_08, ...]'''
    ps_arr = [convert_and_change_coordinate_system(ps) for ps in ps_arr]

    poses_global = []
    ps_prev_last = None
    for ps in ps_arr:
        # ps is of shape (4, 6)
        if ps_prev_last is None:
            # first group - do nothing
            ps_ = ps
        else:
            # use overlapping pose to translate current ps to global coordinate system
            ps_ = []
            for p in ps:
                p_ = np.dot(ps_prev_last, p)
                ps_.append(p_)

        ps_prev_last = ps_[-1]               # (4, 4)

        # skip the last overlapping pose
        for pose_global in ps_[:-1]:
            poses_global.append(pose_global)

    # get interesting values
    poses_stacked = np.stack(poses_global)
    txs = poses_stacked[:, 0, 3]
    tys = poses_stacked[:, 1, 3]
    tzs = poses_stacked[:, 2, 3]

    return poses_stacked, txs, tys, tzs # example - outputing just the position (x,y,z)























