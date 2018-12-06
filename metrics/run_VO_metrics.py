import numpy as np
import os

from utils.auxiliary import read_KITTY_poses, save_mat_pose_to_file, ensure_dir, convert_pdf2jpg
from utils.inverse_warp import merge_sequences_poses
from metrics.terr_rerr_metrics import kittiEvalOdom
from metrics.APE_RPE import APE_RPE_metrics


def run_VO_metrics(data_dir, train_dir, n_iter, data_loader, ps_arr, ATE_RE_errors=None, stereo=False, alignment=False):
    '''Computes the following metrics for KITTY odometry:
         - Absolute Pose Error (APE, m) (with evo package)
         - Relative Pose Error (RPE, m) (with evo package)
         - Average Translational Drift Error (t_err, %)
         - Average Rotational Drift Error (r_err, deg/100m)
         - Absolute Trajectory Error (ATE, m) (for seq_length > 2 only)
         - Rotational Error (RE, deg) (for seq_length > 2 only)
    '''
    assert ps_arr is not None
    test_sequence_path = os.path.join(data_dir, 'test.txt')
    sequence = [folder[:-1] for folder in open(test_sequence_path)][0]
    gt_dir = os.path.join('/'.join(data_dir.split('/')[:-1]), 'poses')
    pose_gt_path = os.path.join(gt_dir, '{}.txt'.format(sequence))
    results_dir_lst, ape_stat_lst, rpe_stat_lst, t_err_lst, r_err_lst, ATE_lst, RE_lst = [], [], [], [], [], [], []
    case_list = ['scaling', 'no_scaling']
    # get visualization always for scales and not scaled trajectories, but report in results.csv always non-scaled
    for case in case_list:
        print('with ', case)
        results_dir = os.path.join(train_dir, 'results_{}_{}'.format(n_iter, case))
        ensure_dir(results_dir)
        results_dir_lst.append(results_dir)

        # merge list of 'seq_lenth'-frame snippets into global poses
        pred_poses_abs, txs, tys, tzs = merge_sequences_poses(ps_arr)

        gt_poses_abs = read_KITTY_poses(pose_gt_path)
        num_samples = len(gt_poses_abs)
        if len(pred_poses_abs) < num_samples:
            idx = len(pred_poses_abs) - (len(gt_poses_abs) - len(pred_poses_abs))
            pred_poses_abs = np.vstack((pred_poses_abs[:, :, :], pred_poses_abs[idx:, :, :]))
        if case == 'scaling':
            # scale the predicted pose
            for i in range(num_samples):
                scale_f = np.sum(gt_poses_abs[i][:, -1] * pred_poses_abs[i, :, -1]) / np.sum(pred_poses_abs[i, :, -1] ** 2)
                pred_poses_abs[i, :, -1] *= scale_f
            # print('scaling the pose')
        pred_poses_mat = pred_poses_abs[:num_samples, :3, :4]

        # save predicted pose
        save_mat_pose_to_file(pred_poses_mat, results_dir, sequence)

        # KITTY odom eval and APE, RPE (load pred and GT global matrix poses from files)
        line = ''
        if data_loader == 'KITTY_odom':
            # APE, RPE
            pred_file = os.path.join(results_dir, '{}.txt'.format(sequence))
            ape_stat, rpe_stat = APE_RPE_metrics(pose_gt_path, pred_file, use_aligned_trajectories=alignment,
                                                 mode='kitti')
            ape_stat_lst.append(ape_stat)
            rpe_stat_lst.append(rpe_stat)
            line += '\nAPE: {:.4f}, RPE: {:.4f}'.format(ape_stat, rpe_stat)

            # t_err, r_err
            odom_result_dir = '/'.join(pred_file.split('/')[:-1])
            odom_eval = kittiEvalOdom(gt_dir)
            odom_eval.eval_seqs = [
                int(sequence)]  # 1,2,4,5,6,7,8,9,10] # Seq 03 is missing since the dataset is not available in KITTY homepage.
            t_err, r_err = odom_eval.eval(odom_result_dir)
            t_err_lst.append(t_err)
            r_err_lst.append(r_err)
            line += '\nAverage translational RMSE (%): {:.4f}, Average rotational error (deg/100m): {:.4f}'.\
                format(t_err, r_err)

        # ATE and RE
        if ATE_RE_errors is not None and np.sum(ATE_RE_errors) > 0:
            ATE, RE = ATE_RE_errors.mean(0)
            ATE_lst.append(ATE)
            RE_lst.append(RE)
            std_errors = ATE_RE_errors.std(0)
            line += "\nATE: {:.4f}, RE: {:.4f}, ATE std: {:.4f}, RE std: {:.4f}\n".format(ATE, RE, *std_errors)

        print(line)
        save_path = os.path.join('/'.join(pred_file.split('/')[:-1]), '{}_results.txt'.format(sequence))
        file = open(save_path, 'w')
        file.write(line)
        file.close()

        # save trajectories to one folder
        convert_pdf2jpg(train_dir, n_iter, suffix=case.split('_')[0], sequence=sequence)

    # if stereo:
    metric_names = ['APE'] + ['RPE'] + ['t_err'] + ['r_err'] + ['ATE'] + ['RE']
    metric_values = [ape_stat_lst[1], rpe_stat_lst[1], t_err_lst[1], r_err_lst[1], ATE_lst[1], RE_lst[1]]
    print('non-scaled pose used for calculating metrics')
    # else:
    #     metric_names = ['APE'] + ['RPE'] + ['t_err'] + ['r_err'] + ['ATE'] + ['RE']
    #     metric_values = [ape_stat_lst[0], rpe_stat_lst[0], t_err_lst[0], r_err_lst[0], ATE_lst[0], RE_lst[0]]
    #     print('scaled pose used for calculating metrics')

    return results_dir_lst, metric_names, metric_values























