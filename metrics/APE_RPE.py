import os
from evo.core import metrics
from evo.tools import log
log.configure_logging(debug=False, silent=False)
from evo.tools import plot
import matplotlib.pyplot as plt
from evo.tools.settings import SETTINGS
SETTINGS.plot_usetex = False
from evo.tools import file_interface
from evo.core import trajectory


def APE_RPE_metrics(gt_file, est_file, use_aligned_trajectories, mode='kitti', save_dir='', show=False):
    '''
        Returns APE and RPE (absolute and relative pose errors) and saves plots
        of errors and trajectories.
        Input:
        - GT pose in matrix format (txt file)
        - Predicted pose in matrix format (txt file)
    '''
    max_diff = 0.01
    offset_2 = 0.0
    file_name = (est_file.split('/')[-1]).split('.')[0]
    if save_dir == '':
        save_dir = '/'.join(est_file.split('/')[:-1])

    if mode == 'kitti':
        traj_ref = file_interface.read_kitti_poses_file(gt_file)
        traj_est = file_interface.read_kitti_poses_file(est_file)
    elif mode == 'tum':
        traj_ref, traj_est = file_interface.load_assoc_tum_trajectories(
            gt_file,
            est_file,
            max_diff,
            offset_2,
        )

    # Umeyama's method for trajectory alignment
    traj_est_aligned = trajectory.align_trajectory(traj_est, traj_ref, correct_scale=False, correct_only_scale=False)
    fig = plt.figure()
    traj_by_label = {
        "estimate (not aligned)": traj_est,
        "estimate (aligned)": traj_est_aligned,
        "reference": traj_ref
    }
    plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    save_path = os.path.join(save_dir, file_name + '_trajectories.png')
    plt.savefig(save_path)
    if show:
        plt.show()

    #################################### APE ######################################
    # The absolute pose error (APE) is a metric for investigating the global consistency of a SLAM trajectory
    # APE is based on the absolute pose difference between two poses
    pose_relation = metrics.PoseRelation.translation_part
    if use_aligned_trajectories:
        data = (traj_ref, traj_est_aligned)
    else:
        data = (traj_ref, traj_est)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    # print('\nAPE rsme = ', ape_stat)
    ape_stats = ape_metric.get_all_statistics()
    # pprint.pprint(ape_stats)

    # Plot the APE values and statistics:
    fig = plt.figure()
    if mode == 'kitti':
        plot.error_array(fig, ape_metric.error, statistics=ape_stats,
                         name="APE", title="APE w.r.t. " + ape_metric.pose_relation.value, xlabel="$t$ (s)")
    elif mode == 'tum':
        seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est.timestamps]
        plot.error_array(fig, ape_metric.error, x_array=seconds_from_start, statistics=ape_stats,
                         name="APE", title="APE w.r.t. " + ape_metric.pose_relation.value, xlabel="$t$ (s)")
    save_path = os.path.join(save_dir, file_name + '_APE_stat.png')
    plt.savefig(save_path)
    if show:
        plt.show()

    # Plot the trajectory with colormapping of the APE:
    plot_mode = plot.PlotMode.xz
    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    plot.traj(ax, plot_mode, traj_ref, '--', "gray", "reference")
    plot.traj_colormap(ax, traj_est_aligned if use_aligned_trajectories else traj_est, ape_metric.error,
                       plot_mode, min_map=ape_stats["min"], max_map=ape_stats["max"])
    ax.legend()
    save_path = os.path.join(save_dir, file_name + '_APE_traj_xz.png')
    plt.savefig(save_path)
    if show:
        plt.show()

    ####################################### RPE #########################################3
    # RPE compares the relative poses along the estimated and the reference trajectory.
    # This is based on the delta pose difference
    pose_relation = metrics.PoseRelation.rotation_angle_deg
    # normal mode
    delta = 1
    delta_unit = metrics.Unit.frames
    # all pairs mode
    all_pairs = False  # activate
    data = (traj_ref, traj_est)
    rpe_metric = metrics.RPE(pose_relation, delta, delta_unit, all_pairs)
    rpe_metric.process_data(data)
    rpe_stat = rpe_metric.get_statistic(metrics.StatisticsType.rmse)
    # print('\nRPE rsme = ', rpe_stat)
    rpe_stats = rpe_metric.get_all_statistics()
    # pprint.pprint(rpe_stats)

    # Plot the RPE values and statistics
    # important: restrict data to delta ids for plot
    import copy
    traj_ref_plot = copy.deepcopy(traj_ref)
    if use_aligned_trajectories:
        traj_est = traj_est_aligned
    else:
        traj_est = traj_est
    traj_est_plot = copy.deepcopy(traj_est)
    traj_ref_plot.reduce_to_ids(rpe_metric.delta_ids)
    traj_est_plot.reduce_to_ids(rpe_metric.delta_ids)
    fig = plt.figure()
    if mode == 'kitti':
        plot.error_array(fig, rpe_metric.error, statistics=rpe_stats,
                         name="RPE", title="RPE w.r.t. " + rpe_metric.pose_relation.value, xlabel="$t$ (s)")
    elif mode == 'tum':
        seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est.timestamps[1:]]
        plot.error_array(fig, rpe_metric.error, x_array=seconds_from_start, statistics=rpe_stats,
                         name="RPE", title="RPE w.r.t. " + rpe_metric.pose_relation.value, xlabel="$t$ (s)")
    save_path = os.path.join(save_dir, file_name + '_RPE_stat.png')
    plt.savefig(save_path)
    if show:
        plt.show()

    # Plot the trajectory with colormapping of the RPE
    plot_mode = plot.PlotMode.xz
    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    plot.traj(ax, plot_mode, traj_ref_plot, '--', "gray", "reference")
    plot.traj_colormap(ax, traj_est_plot, rpe_metric.error, plot_mode, min_map=rpe_stats["min"],
                       max_map=rpe_stats["max"])
    ax.legend()
    save_path = os.path.join(save_dir, file_name + '_RPE_traj_xz.png')
    plt.savefig(save_path)
    if show:
        plt.show()

    return ape_stat, rpe_stat




if __name__ == '__main__':
    ############ TUM APE, RPE #####################
    # Load two trajectories files in TUM format associated via matching timestamps
    # mode = 'tum'
    # gt_file = "/home/victoria/evo/test/data/freiburg1_xyz-groundtruth.txt"
    # est_file = "/home/victoria/evo/test/data/freiburg1_xyz-rgbdslam_drift.txt"

    ############ KITTY APE, RPE #####################
    # load two trajectories in the kitti format
    mode = 'kitti'
    # gt_file = '/home/victoria/evo/test/data/KITTI_00_gt.txt'
    # est_file = '/home/victoria/evo/test/data/KITTI_00_ORB.txt'
    sequence = '10'
    gt_file = '/media/victoria/d/data/KITTI_odom/data_odometry_color/poses/{}.txt'.format(sequence)
    est_file = '/media/victoria/d/models/PoseEst/May16_23-34-02_KITTY_odom_zhou_h128_w416_l0.0002_s1/results_1000000/{}.txt'.format(sequence)
    use_aligned_trajectories = True

    ape_stat, rpe_stat = APE_RPE_metrics(gt_file, est_file, use_aligned_trajectories, mode=mode)

    save_path = os.path.join('/'.join(est_file.split('/')[:-1]), '{}_results.txt'.format(sequence))
    line = '\nAPE: {} \n RPE: {}'.format(ape_stat, rpe_stat)
    print(line)
    file = open(save_path, 'a')
    file.write(line)
    file.close()

    ########### KITTY odom eval ###################
    from metrics.terr_rerr_metrics import kittiEvalOdom
    gt_dir = '/'.join(gt_file.split('/')[:-1])
    odom_result_dir = '/'.join(est_file.split('/')[:-1])
    odom_eval = kittiEvalOdom(gt_dir)
    odom_eval.eval_seqs = [int(sequence)]     #,1,2,4,5,6,7,8,9,10] # Seq 03 is missing since the dataset is not available in KITTY homepage.
    t_err, r_err = odom_eval.eval(odom_result_dir)

    save_path = os.path.join(odom_result_dir, '{}_results.txt'.format(sequence))
    line = 'Average translational RMSE (%): {} \nAverage rotational error (deg/100m): {}'.format(t_err, r_err)
    print(line)
    file = open(save_path, 'a')
    file.write(line)
    file.close()




















