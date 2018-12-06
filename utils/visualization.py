import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
import PIL
from PIL import Image


def results_visualization(root_dir):
    '''
        Visualization of all run's results, which are recorded in the results table,
        in all subdir in root_dir.
    '''

    dirlist1 = os.listdir(root_dir)
    dirlist_tot = []
    for dirname in dirlist1:
        if len(dirname.split('.')) < 2:
            dirlist_tot.append(dirname)

    if len(dirlist_tot) > 0:
        num_cols = 0
        num_rows_max = 5
        dirlist = []
        for dirr in dirlist_tot:
            results_table_path = os.path.join(os.path.join(root_dir, dirr), 'results.csv')
            if os.path.isfile(results_table_path):
                results_table = pd.read_csv(results_table_path, index_col=0)
                if len(results_table.index) > 0:
                    dirlist.append(dirr)
                    num_cols = len(results_table.columns) - 1

        if len(dirlist) > 0 and num_cols > 0:
            num_rows = num_rows_max
            num_loops = int(np.ceil(len(dirlist) / float(num_rows_max)))

            for loop_num in range(num_loops):
                print('starting making results visualization {}'.format(loop_num+1))
                idx_min = loop_num*num_rows_max
                idx_max = min(loop_num*num_rows_max + num_rows_max, len(dirlist))

                f, axarr = plt.subplots(num_rows, num_cols)
                f.set_size_inches((19, 23), forward=False)
                plt.suptitle('Results visualization', fontsize=16)
                for i, exp in enumerate(dirlist[idx_min:idx_max]):
                    results_table_path = os.path.join(os.path.join(root_dir, exp), 'results.csv')
                    if os.path.isfile(results_table_path):
                        results_table = pd.read_csv(results_table_path, index_col=0)
                        if len(results_table.index) > 0:
                            col_list = list(results_table)
                            assert 'iter' in col_list
                            x = np.array(results_table['iter'])
                            j = 0
                            for col in col_list:
                                if j == 0:
                                    axarr[i, 0].set_title('Experiment {}'.format(exp), fontsize=14)
                                if col != 'iter':
                                    y = np.array(results_table[col])
                                    if col == 'train_loss':
                                        y_best = np.min(y)
                                    else:
                                        y_best = np.max(y)
                                    axarr[i, j].plot(x, y)
                                    axarr[i, j].set_ylabel(col)
                                    axarr[i, j].text(1, 0.5, 'best {}: {:.4f}'.format(col, y_best),
                                                     verticalalignment='bottom', horizontalalignment='right',
                                                     transform=axarr[i, j].transAxes, color='green', fontsize=12)
                                    j += 1

                # Fine-tune figure; make subplots farther from each other.
                f.subplots_adjust(hspace=0.3)

                # save the figure to file
                path = os.path.join(root_dir, 'results_{}.png'.format(loop_num+1))
                f.savefig(path, dpi=500, bbox_inches='tight')
                print('visualization is saved to {}'.format(root_dir))
                # plt.show()
        else:
            print('no valid results table found in the dir {}'.format(root_dir))


def w_params_visualization(train_dir, num_workers):
    '''
        Visualization of advancement of all workers in PBT training
    '''
    history_path = os.path.join(train_dir, 'history.csv')
    if os.path.isfile(history_path):
        history_table = pd.read_csv(history_path, index_col=0)
        num_rows, num_cols = history_table.shape
        assert num_rows > 0
        num_rows_per_worker = int(num_rows / float(num_workers))

        # get param names
        num_params = (num_cols - 4) / 2
        param_list = []
        for col in list(history_table):
            if col != 'worker' and col != 'iter' and col != 'test_acc' and col != 'copied_from_w' and not col.startswith('mutation_'):
                param_list.append(col)
        print('num_cols = {}, num_params = {}'.format(num_cols, num_params), 'param_list = ', param_list)

        # every worker array will have two params (x, y and accuracy), if there are more params, they will be added
        ## to x or y for 2D representation purposes
        num_params_repr = 2
        worker_arrays = np.zeros((num_workers, num_rows_per_worker, num_params_repr+1))

        w_row_idx = -1
        for row_idx in range(num_rows):
            w = int(history_table.iloc[row_idx]['worker'])
            if w == 0:
                w_row_idx += 1
            for param_idx in range(num_params):
                worker_arrays[w][w_row_idx][param_idx%num_params_repr] += history_table.iloc[row_idx][param_list[param_idx]]
                worker_arrays[w][w_row_idx][2] = history_table.iloc[row_idx]['test_acc']
                # if w == 0:
                    # print 'w = {}, param = {}, row_idx = {}, val = {}'.format(w, param_list[param_idx], row_idx, history_table.iloc[row_idx][param_list[param_idx]])
                    # print 'w_row_idx = {}, worker_arrays[w][w_row_idx][param_idx%num_params_repr] = {}'.format(w_row_idx, worker_arrays[w][w_row_idx][param_idx%num_params_repr])

        # print 'worker_arrays[0] = \n', worker_arrays[0]
        # print 'worker_arrays[0][:, 0] = ', worker_arrays[0][:, 0]
        # print 'worker_arrays[0][:, 1] = ', worker_arrays[0][:, 1]
        for w in range(num_workers):
            plt.scatter(worker_arrays[w][w_row_idx:w_row_idx+1, 0], worker_arrays[w][w_row_idx:w_row_idx+1, 1], marker='d', s=250, c='r')
            plt.plot(worker_arrays[w][:, 0], worker_arrays[w][:, 1], '--', color='black', linewidth=1)
            plt.scatter(worker_arrays[w][:, 0], worker_arrays[w][:, 1], c=worker_arrays[w][:, 2], marker='s', s=150)
        plt.colorbar()

        # save the figure to file
        path = os.path.join(train_dir, 'results.png')
        plt.savefig(path, dpi=500, bbox_inches='tight')
        print('visualization is saved to {}'.format(train_dir))
        # plt.show()
    else:
        print('no history table found')


def save_concat_imgs(images, img_names, save_path):
    '''
        Saves concatenated images to file
        Input:
        - images - list of np arrays of images (of length N each), which names are decribed in img_names
        - img_names - list if image names ('trg', 'ref', 'disp', 'depth', 'ref_inv_warp', 'ref_warped', 'trg-ref_warped')
        - filenames - list of N filenames
    '''
    imgs = [Image.fromarray(np.array(img_np*255).astype('uint8')) for img_np in images]
    # pick the image which is the smallest, and resize the others to match it
    min_shape = sorted([(np.sum(img.size), img.size) for img in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(img.resize(min_shape)) for img in imgs))

    # save combined image
    imgs_concat = PIL.Image.fromarray(imgs_comb)
    imgs_concat.save(save_path)


def plot3D(x, y, z, n_iter, line_label='', is_locs=True, x2=None, y2=None, z2=None, line_label2='',
           show=False, save_dir=None):
    '''
        3D plot of trajectories (locations or euler angles).
    '''
    label1, label2, label3 = 'azimut', 'roll', 'pitch'
    label = 'angs3D'
    if is_locs:
        label1, label2, label3 = 'x', 'z', 'y'
        label = 'locs3D'
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.text2D(0.05, 0.95, "3D trajectories", transform=ax.transAxes)
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_zlabel(label3)
    ax.plot(x, z, y, label='{}'.format(line_label))
    if line_label2 != '':
        ax.plot(x2, z2, y2, label='{}'.format(line_label2))
    ax.legend()
    if show:
        plt.show()
    if save_dir != None:
        fig.savefig(os.path.join(save_dir, '{}_{}.png'.format(label, n_iter)))


def visualization_trajectories(pred_poses_tgt, gt_poses_tgt, visualization_paths_test_dir, n_iter):
    # 2D trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches((10, 4), forward=False)
    plt.suptitle('Trajectory visualization', fontsize=16)
    ax1.plot(pred_poses_tgt[:, 0], pred_poses_tgt[:, 2], label='pred')
    ax1.plot(gt_poses_tgt[:, 0], gt_poses_tgt[:, 2], label='GT')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('z (m)')
    ax1.legend()
    ax2.plot(pred_poses_tgt[:, 0], pred_poses_tgt[:, 1], label='pred')
    ax2.plot(gt_poses_tgt[:, 0], gt_poses_tgt[:, 1], label='GT')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.legend()
    path = os.path.join(visualization_paths_test_dir, 'trajectories_{}.png'.format(n_iter))
    fig.savefig(path, dpi=500, bbox_inches='tight')
    print('Trajectories visualization is saved to {}'.format(visualization_paths_test_dir))
    # plt.show()

    # 3D trajectories
    plot3D(pred_poses_tgt[:, 0], pred_poses_tgt[:, 1], pred_poses_tgt[:, 2], n_iter, line_label='pred', is_locs=True,
           x2=gt_poses_tgt[:, 0], y2=gt_poses_tgt[:, 1], z2=gt_poses_tgt[:, 2], line_label2='GT', show=False,
           save_dir=visualization_paths_test_dir)
    plot3D(pred_poses_tgt[:, 3], pred_poses_tgt[:, 4], pred_poses_tgt[:, 5], n_iter, line_label='pred', is_locs=False,
           x2=gt_poses_tgt[:, 3], y2=gt_poses_tgt[:, 4], z2=gt_poses_tgt[:, 5], line_label2='GT', show=False,
           save_dir=visualization_paths_test_dir)







if __name__ == '__main__':
    root_dir = '/media/victoria/d/models/mnist'
    results_visualization(root_dir)



























