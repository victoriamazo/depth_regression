import numpy as np

from utils.inverse_warp import pose_vec2mat

from torch.autograd import Variable


def compute_ATE_RE(pred_pose, gt_poses, rotation_mode='euler'):
    '''Computes ATE and RE for minisequence of n frames (ususally 3 or 5)'''
    # from utils.auxiliary import read_scene_data_KITTY
    # gt_pose, sample_indices = read_scene_data_KITTY('/media/victoria/d/data/KITTI_odom/data_odometry_color/sequences', ['09'], seq_length=3, step=1)
    # # gt_poses[0], sample_indices[0][i]
    # gt_poses1 = np.stack(gt_pose[0][j] for j in sample_indices[0][0])
    gt_poses = gt_poses.data.numpy().astype(np.float32)
    first_pose = gt_poses[0]
    gt_poses[:, :, -1] -= first_pose[:, -1]
    gt_poses = np.linalg.inv(first_pose[:, :3]) @ gt_poses

    inv_transform_matrices = pose_vec2mat(Variable(pred_pose), rotation_mode=rotation_mode).data.numpy().astype(np.float32)

    rot_matrices = np.linalg.inv(inv_transform_matrices[:, :, :3])
    tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]

    transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

    first_inv_transform = inv_transform_matrices[0]
    final_poses = first_inv_transform[:, :3] @ transform_matrices
    final_poses[:, :, -1:] += first_inv_transform[:, -1:]

    ATE, RE = compute_pose_error(gt_poses, final_poses)

    return ATE, RE


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return ATE/snippet_length, RE/snippet_length


