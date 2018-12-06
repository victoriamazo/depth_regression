import numpy as np
from scipy.ndimage.interpolation import zoom
import os
from path import Path

from utils.auxiliary import generate_mask


def compute_depth_metrics(depth_metrics, train_dir, n_iter):
    metric_values = depth_metrics.mean(1)
    metric_names = ['abs_rel', 'sq_rel', 'rmse', 'log_rmse', 'a1', 'a2', 'a3']

    line = "Depth results with scale factor determined by GT/prediction ratio (like the original paper) : \n"
    line += "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}\n".format(*metric_names)
    line += "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}\n".format(*metric_values)
    print(line)

    # write results to file
    train_dir = Path(train_dir)
    results_dir = train_dir/'depth_results'.format(n_iter)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    save_path = results_dir / '{}.txt'.format(n_iter)
    f = open(save_path, 'w')
    f.write(line)
    f.close()

    return metric_names, metric_values


def compute_depth_metrics_i(pred_depth, gt_depth, min_depth=1e-3, max_depth=80):
    '''Computes depth metrics for frame i:
         - a1
         - a2
         - a3
         - RMSE
         - RMSE_log
         - Abs_rel
         - Sq_rel
        '''
    pred_depth_zoomed = zoom(pred_depth, (gt_depth.shape[0] / pred_depth.shape[0],
                                          gt_depth.shape[1] / pred_depth.shape[1])).clip(min_depth, max_depth)
    mask = generate_mask(gt_depth)
    pred_depth_zoomed = pred_depth_zoomed[mask]
    gt_depth = gt_depth[mask]

    # scale factor determined by GT/prediction ratio (like the original paper)
    scale_factor = np.median(gt_depth) / np.median(pred_depth_zoomed)
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(gt_depth, pred_depth_zoomed * scale_factor)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


