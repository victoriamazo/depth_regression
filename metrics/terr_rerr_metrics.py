import numpy as np
from matplotlib import pyplot as plt
import os, os.path

from utils.auxiliary import read_KITTY_poses


class kittiEvalOdom():
    '''
        (From pytorch implementation of Zhou et al. "Unsupervised Learning of Depth and Ego-Motion from Video")
        
        t_err is averaged RMSE over position drifts (differences in position between the first and the
    last frame in segments [100, 200, 300, 400, 500, 600, 700, 800] meters) divided by
    segment length (100 m) per sequence (in %).
        r_err is averaged RMSE over rotational drifts (differences in orientarion between the first and the
    last frame in segments [100, 200, 300, 400, 500, 600, 700, 800] meters) divided by
    segment length (100 m) per sequence (in degrees per 100 m)
        Input:
            - Kitty odometry GT in the following format: txt file for each sequence
            with every line is a pose in the matrix format (first 12 element of the 4x4
            transforamtion matrix, the other 3 are '0, 0 ,1'
            - Predicted poses for each sequence in the same format as GT
        Returns:
            - t_err - average translational RMSE (in %)
            - r_err - average rotational error (in deg/100m)
    '''
    # ----------------------------------------------------------------------
    # poses: [N,4,4]
    # pose: [4,4]
    # ----------------------------------------------------------------------
    def __init__(self, gt_dir):
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)
        self.gt_dir = gt_dir


    def trajectoryDistances(self, poses):
        # ----------------------------------------------------------------------
        # poses: dictionary: [frame_idx: pose]
        # ----------------------------------------------------------------------
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx) - 1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i + 1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i] + np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        return dist


    def rotationError(self, pose_error):
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        return np.arccos(max(min(d, 1.0), -1.0))


    def translationError(self, pose_error):
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


    def lastFrameFromSegmentLength(self, dist, first_frame, len_):
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + len_):
                return i
        return -1


    def calcSequenceErrors(self, poses_gt, poses_result):
        err = []
        dist = self.trajectoryDistances(poses_gt)
        self.step_size = 10

        for first_frame in range(9, len(poses_gt), self.step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]           #[100, 200, 300, 400, 500, 600, 700, 800]
                last_frame = self.lastFrameFromSegmentLength(dist, first_frame, len_)

                # ----------------------------------------------------------------------
                # Continue if sequence not long enough
                # ----------------------------------------------------------------------
                if last_frame == -1 or not (last_frame in poses_result.keys()) or not (
                    first_frame in poses_result.keys()):
                    continue

                # ----------------------------------------------------------------------
                # compute rotational and translational errors
                # ----------------------------------------------------------------------
                pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
                pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

                r_err = self.rotationError(pose_error)
                t_err = self.translationError(pose_error)

                # ----------------------------------------------------------------------
                # compute speed
                # ----------------------------------------------------------------------
                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1 * num_frames)

                err.append([first_frame, r_err / len_, t_err / len_, len_, speed])
        return err


    def saveSequenceErrors(self, err, file_name):
        fp = open(file_name, 'w')
        for i in err:
            line_to_write = " ".join([str(j) for j in i])
            fp.writelines(line_to_write + "\n")
        fp.close()


    def computeOverallErr(self, seq_err):
        t_err = 0
        r_err = 0

        seq_len = len(seq_err)

        for item in seq_err:
            r_err += item[1]
            t_err += item[2]
        ave_t_err = t_err / seq_len
        ave_r_err = r_err / seq_len
        return ave_t_err, ave_r_err


    def plotPath(self, seq, poses_gt, poses_result):
        plot_keys = ["Ground Truth", "Ours"]
        fontsize_ = 20
        plot_num = -1

        poses_dict = {}
        poses_dict["Ground Truth"] = poses_gt
        poses_dict["Ours"] = poses_result

        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')

        for key in plot_keys:
            pos_xz = []
            # for pose in poses_dict[key]:
            for frame_idx in sorted(poses_dict[key].keys()):
                pose = poses_dict[key][frame_idx]
                pos_xz.append([pose[0, 3], pose[2, 3]])
            pos_xz = np.asarray(pos_xz)
            plt.plot(pos_xz[:, 0], pos_xz[:, 1], label=key)

        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xticks(fontsize=fontsize_)
        plt.yticks(fontsize=fontsize_)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        fig.set_size_inches(10, 10)
        png_title = "sequence_{:02}".format(seq)
        plt.savefig(self.plot_path_dir + "/" + png_title + ".pdf", bbox_inches='tight', pad_inches=0)
    # plt.show()


    def plotError(self, avg_segment_errs):
        # ----------------------------------------------------------------------
        # avg_segment_errs: dict [100: err, 200: err...]
        # ----------------------------------------------------------------------
        plot_y = []
        plot_x = []
        for len_ in self.lengths:
            plot_x.append(len_)
            plot_y.append(avg_segment_errs[len_][0])
        fig = plt.figure()
        plt.plot(plot_x, plot_y)
        plt.show()


    def computeSegmentErr(self, seq_errs):
        # ----------------------------------------------------------------------
        # This function calculates average errors for different segment.
        # ----------------------------------------------------------------------

        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []
        # ----------------------------------------------------------------------
        # Get errors
        # ----------------------------------------------------------------------
        for err in seq_errs:
            len_ = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])
        # ----------------------------------------------------------------------
        # Compute average
        # ----------------------------------------------------------------------
        for len_ in self.lengths:
            if segment_errs[len_] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs


    def eval(self, result_dir):
        error_dir = result_dir + "/errors"
        self.plot_path_dir = result_dir + "/plot_path"
        if not os.path.exists(error_dir):
            os.makedirs(error_dir)
        if not os.path.exists(self.plot_path_dir):
            os.makedirs(self.plot_path_dir)

        total_err = []

        ave_t_errs = []
        ave_r_errs = []
        t_err = []
        r_err = []
        for i in self.eval_seqs:
            self.cur_seq = '{:02}'.format(i)
            file_name = '{:02}.txt'.format(i)

            poses_result = read_KITTY_poses(result_dir + "/" + file_name)
            poses_gt = read_KITTY_poses(self.gt_dir + "/" + file_name)
            self.result_file_name = result_dir + file_name

            # ----------------------------------------------------------------------
            # compute sequence errors
            # ----------------------------------------------------------------------
            seq_err = self.calcSequenceErrors(poses_gt, poses_result)
            self.saveSequenceErrors(seq_err, error_dir + "/" + file_name)

            # ----------------------------------------------------------------------
            # Compute segment errors
            # ----------------------------------------------------------------------
            avg_segment_errs = self.computeSegmentErr(seq_err)

            # ----------------------------------------------------------------------
            # compute overall error
            # ----------------------------------------------------------------------
            ave_t_err, ave_r_err = self.computeOverallErr(seq_err)
            t_err.append(ave_t_err * 100)
            r_err.append(ave_r_err / np.pi * 180 * 100)
            # print("Sequence: " + str(i))
            # print("Average translational RMSE (%): ", t_err[0])
            # print("Average rotational error (deg/100m): ", r_err[0])
            ave_t_errs.append(ave_t_err)
            ave_r_errs.append(ave_r_err)

            # ----------------------------------------------------------------------
            # Ploting (To-do)
            # (1) plot trajectory
            # (2) plot per segment error
            # ----------------------------------------------------------------------
            self.plotPath(i, poses_gt, poses_result)
        # self.plotError(avg_segment_errs)

        # print("-------------------- For Copying ------------------------------")
        # for i in range(len(ave_t_errs)):
        #     print("{0:.2f}".format(ave_t_errs[i] * 100))
        #     print("{0:.2f}".format(ave_r_errs[i] / np.pi * 180 * 100))

        return t_err[0], r_err[0]






if __name__ == '__main__':
    gt_dir = '/media/victoria/d/data/KITTI_odom/pose_gt'
    odom_result_dir = '/media/victoria/d/models/PoseEst/May16_23-34-02_KITTY_odom_zhou_h128_w416_l0.0002_s1/results_1000000' #'/media/victoria/d/data/KITTI_odom/results'
    odom_eval = kittiEvalOdom(gt_dir)
    sequence = '00'
    odom_eval.eval_seqs = [int(sequence)]     #,1,2,4,5,6,7,8,9,10] # Seq 03 is missing since the dataset is not available in KITTY homepage.
    t_err, r_err = odom_eval.eval(odom_result_dir)

    save_path = os.path.join(odom_result_dir, '{}_results.txt'.format(sequence))
    line = 'Average translational RMSE (%): {} \nAverage rotational error (deg/100m): {}'.format(t_err, r_err)
    print(line)
    file = open(save_path, 'a')
    file.write(line)
    file.close()
