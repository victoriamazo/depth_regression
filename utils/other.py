import os
from wand.image import Image
import shutil
from path import Path
from pdf2jpg import pdf2jpg

def copy_files():
    root_dir = '/media/victoria/d/mnist/training'
    dir_for_copied_files = '/media/victoria/d/mnist/train'
    file_suffix = 'png'

    if not os.path.isdir(dir_for_copied_files):
        os.makedirs(dir_for_copied_files)

    count = 0
    for subdir in os.listdir(root_dir):
        if subdir != '.DS_Store':
            print('subdir {}'.format(subdir))
            for filename in os.listdir(os.path.join(root_dir, subdir)):
                print('filename = ', filename)
                if filename.split('.')[-1] == 'png':
                    file_path = os.path.join(os.path.join(root_dir, subdir), filename)
                    copied_file_path = os.path.join(dir_for_copied_files, '{}_{}.{}'.format(filename.split('.')[0],
                                                                                            subdir, file_suffix))
                    shutil.copy(file_path, copied_file_path)
                    count += 1
    print('copied to {} {} files'.format(dir_for_copied_files, count))


def make_file_list(root_dir):
    '''File list of all KITTY_raw image_02 files, which are not in test_files_eigen.'''
    root_dir_path = Path(root_dir)

    with open(root_dir_path/'test_files_eigen.txt', 'r') as f:
        test_files = list(f.read().splitlines())

    train_files = []
    for date in os.listdir(root_dir):
        if os.path.isdir(root_dir_path/date):
            for scene in os.listdir(root_dir_path/date):
                if os.path.isdir(root_dir_path/date/scene):
                    for cam_id in os.listdir(root_dir_path/date/scene):
                        if cam_id == 'image_02':
                            img_dir = date + '/' + scene + '/' + cam_id + '/' + 'data'
                            file_list = os.listdir(root_dir_path/img_dir)
                            file_list_sort = sorted(file_list)
                            for file in file_list_sort:
                                file_path_rel = img_dir + '/' + file
                                if file[-3:] == 'png' and file_path_rel not in test_files:
                                    train_files.append(file_path_rel)

    train_files_path = root_dir_path/'train_files_eigen.txt'
    with open(train_files_path, 'w') as f:
        for item in train_files:
            f.write("%s\n" % item)

    print('number of train files: {}, test files: {}'.format(len(train_files), len(test_files)))




def convert_pdf2jpg(root_dir, suffix='no', sequence='09'):
    '''Converts and saves pdf paths to jmg for all paths in results
    to quickly see the non-scaled paths.
    Suffix can be:
      - 'no' for non-scaled paths
      - 'scaling' for scaled paths
    '''
    if suffix == 'no':
        save_dir = os.path.join(root_dir, 'non-scaled_paths')
    else:
        save_dir = os.path.join(root_dir, 'scaled_paths')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for subdir in os.listdir(root_dir):
        if subdir.startswith('results_'):
            subdir_path = os.path.join(root_dir, subdir)
            n_iter = subdir.split('_')[1]
            subdir_suffix = ''
            if len(subdir.split('_')) > 2:
                subdir_suffix = subdir.split('_')[2]
            if subdir_suffix == suffix:
                pdf_path = os.path.join(subdir_path, 'plot_path/sequence_{}.pdf'.format(sequence))
                jpg_path = os.path.join(save_dir, '{}.jpg'.format(n_iter))
                with Image(filename=pdf_path, resolution=200) as img:
                    # keep good quality
                    img.compression_quality = 80
                    # save it to tmp name
                    img.save(filename=jpg_path)




if __name__ == '__main__':
    root_dir = '/media/victoria/d/models/PoseEst/1809Sep09_14-53-11_KITTY_odom_zhan_l0.0002_b4_l12-0.5,LR-0.25,RL-0.25,S-0.1,E-0.2_e1_s1'
    convert_pdf2jpg(root_dir, suffix='no')
    convert_pdf2jpg(root_dir, suffix='scaling')

    # make_file_list('/media/victoria/d/data/KITTI_raw')