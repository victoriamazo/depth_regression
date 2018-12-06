import os
import imp

import torch.nn as nn


class Model(object):
    def __init__(self):
        pass


    @classmethod
    def model_builder(cls, target_class, FLAGS, parent_class=nn.Module):
        path = os.path.dirname(os.path.realpath(__file__))
        for filename in os.listdir(path):
            prefix, suffix = filename.split('.')[0], filename.split('.')[-1]
            if suffix == 'py' and prefix != '__init__' and prefix != 'layers' and prefix != 'model_builder':
                path_to_module = os.path.join(path, filename)
                module_dir, module_file = os.path.split(path_to_module)
                module_name, module_ext = os.path.splitext(module_file)
                module_obj = imp.load_source(module_name, path_to_module)

                for name in dir(module_obj):
                    if str(name) == str(target_class):
                        o = getattr(module_obj, name)
                        try:
                            if issubclass(o, parent_class):
                                return o(FLAGS)
                        except TypeError:
                            pass








if __name__ == "__main__":
    from dataloaders.dataloader_builder import DataLoader
    import torch.utils.data as data
    from PIL import Image


    # ########################  KITTY  #############################
    # # data loader test
    # class FLAGS():
    #     def __init__(self):
    #         self.batch_size = 10
    #         self.data_dir = '/media/victoria/d/data/KITTI_odom/data_odometry_color/sequences'
    #         self.series_list = ['00']
    #         self.height = 416
    #         self.width = 128
    #         self.n_hiddens = '256, 256'
    #         self.num_classes = 10
    #
    # FLAGS = FLAGS()
    # mode = 'train'
    # target_class = 'KITTY_loader'
    # gen, num_samples = DataLoader.dataloader_builder(target_class, FLAGS, mode)
    #
    # # show images
    # idx = 1
    # for batch_idx, (img1_trans, img2_trans, filename) in enumerate(gen):
    #     print(batch_idx)
    #     if batch_idx % 20 == 0:
    #         print('img1_trans = {}, img2_trans = {}, filename = {}'.format(img1_trans.shape, img2_trans.shape, filename))
    #         # im1 = Image.fromarray(img1_trans[idx])
    #         # im1_name = 'img1_trans_{}.png'.format(filename[idx])
    #         # im1.save('debug/{}'.format(im1_name))
    #         # os.system('xdg-open debug/{}'.format(im1_name))
    #         # im2 = Image.fromarray(img2_trans[idx])
    #         # im2_name = 'img2_trans_{}.png'.format(filename[idx])
    #         # im2.save('debug/{}'.format(im2_name))
    #         # os.system('xdg-open debug/{}'.format(im2_name))
    #     if batch_idx == 10:
    #         break
    #
    # # model test
    # target_class = 'mpl'
    # model = Model.model_builder(target_class, FLAGS)
    # print("model = ", model)




    # ########################  EuRoc_MAV  #############################
    # from dataloaders.dataloader_builder import DataLoader
    #
    # # data loader test
    # class FLAGS():
    #     def __init__(self):
    #         self.batch_size = 7
    #         self.data_dir = '/media/victoria/d/data/EuRoC_MAV'
    #         self.data_loader = 'EuRoc'
    #         self.height = 240
    #         self.width = 376
    #         self.seed = 127
    #         self.stereo = 1
    #         self.seq_length = 2
    #         self.euler_angles = False
    #         self.with_gt = 1
    #         self.hflip = 1
    #         self.rand_crop = 1
    # FLAGS = FLAGS()
    # mode = 'train'
    # dataloader, num_samples = DataLoader.dataloader_builder(FLAGS.data_loader, FLAGS, mode, parent_class=data.Dataset)
    #
    # epoch_size = len(dataloader)
    # print('epoch_size = ', epoch_size)
    #
    # for i, var_list in enumerate(dataloader):
    #     tgt_img_l, ref_imgs_l, tgt_img_r, ref_imgs_r, intrinsics_l, intrinsics_r, intrinsics_inv_l,\
    #     intrinsics_inv_r, trg_pose, ref_pose, filename_tgt, filename_ref = var_list
    #     print('\n', i, 'tgt_img_l = ', tgt_img_l.shape)
    #     print('ref_imgs_l = ', ref_imgs_l.shape)
    #     print('tgt_img_r = ', tgt_img_r.shape)
    #     print('ref_imgs_r = ', ref_imgs_r.shape)
    #     print('intrinsics_l = ', intrinsics_l)
    #     print('intrinsics_r = ', intrinsics_r)
    #     print('filename_tgt = ', filename_tgt)
    #     if trg_pose is not None:
    #         print('trg_pose = ', trg_pose)
    #     print('filename_ref = ', filename_ref)
    #     if ref_pose is not None:
    #         print('ref_pose = ', ref_pose)



    # ########################  KITTY_formatted  #############################
    # from dataloaders.dataloader_builder import DataLoader
    #
    # # data loader test
    # class FLAGS():
    #     def __init__(self):
    #         self.batch_size = 1
    #         self.data_dir = '/media/victoria/d/data/KITTI_formatted'
    #         self.data_loader = 'KITTY_formatted'
    #         self.height = 128
    #         self.width = 416
    #         self.seq_length = 2
    #         self.seed = 127
    #         self.stereo = 1
    #         self.euler_angles = True
    #         self.with_gt = 1
    #         self.hflip = 1
    #         self.rand_crop = 1
    # FLAGS = FLAGS()
    # mode = 'train'
    # dataloader, num_samples = DataLoader.dataloader_builder(FLAGS.data_loader, FLAGS, mode, parent_class=data.Dataset)
    #
    # epoch_size = len(dataloader)
    # print('epoch_size = ', epoch_size)
    #
    # for i, (tgt_img_l, ref_img_l, intrinsics, intrinsics_inv, trg_pose, ref_pose, filename_tgt, filename_ref) \
    #         in enumerate(dataloader):
    #     print('\n', i, 'tgt_img_l = ', tgt_img_l.shape)
    #     print('ref_img_l = ', ref_img_l.shape)
    #     print('intrinsics_l = ', intrinsics)
    #     print('filename_tgt = ', filename_tgt)
    #     print('filename_ref = ', filename_ref)



    ########################  KITTY_odom  #############################
    class FLAGS():
        def __init__(self):
            self.batch_size = 2
            self.data_dir = '/media/victoria/d/data/KITTI_odom/data_odometry_color/sequences'
            self.height = 128
            self.width = 416
            self.seed = 127
            self.stereo = 0
            self.seq_length = 5
            self.euler_angles = True
            self.with_gt = 0
            self.hflip = 1
            self.rand_crop = 1
            self.shuffle = False
            self.data_loader = 'KITTY_odom'


    FLAGS = FLAGS()

    import multiprocessing
    from dataloaders.KITTY_odom import KITTY_odom

    mode = 'train'
    train_set = KITTY_odom(FLAGS, mode)
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('stereo: ', bool(FLAGS.stereo), ", seq_length = ", FLAGS.seq_length, 'with_gt:', bool(FLAGS.with_gt))

    train_loader = data.DataLoader(train_set, batch_size=FLAGS.batch_size, shuffle=FLAGS.shuffle,
                                   num_workers=multiprocessing.cpu_count() - 2, pin_memory=True)
    epoch_size = len(train_loader)
    print('epoch_size = ', epoch_size)

    if not FLAGS.stereo and FLAGS.seq_length == 1:
        for i, var_list in enumerate(train_loader):
            tgt_img_l, intrinsics_l, intrinsics_inv, trg_pose, filename_tgt = var_list
            print('\n', i, 'tgt_img_l = ', tgt_img_l.shape)  # tgt_img is a PyTorch tensor
            print('intrinsics_l = ', intrinsics_l)
            print('filename_tgt = ', filename_tgt)
            if trg_pose is not None:
                print('trg_pose = ', trg_pose)
    elif not FLAGS.stereo and FLAGS.seq_length > 1:
        for i, var_list in enumerate(train_loader):
            tgt_img_l, ref_imgs_l, intrinsics_l, intrinsics_inv, trg_pose, ref_poses, filename_tgt, filenames_ref = var_list
            print('\n', i, 'tgt_img_l = ', tgt_img_l.shape)
            print('ref_imgs_l = ', len(ref_imgs_l), ref_imgs_l[0].shape)
            print('intrinsics_l = ', intrinsics_l)
            print('filename_tgt = ', filename_tgt)
            if trg_pose is not None:
                print('trg_pose = ', trg_pose)
            print('filenames_ref = ', filenames_ref)
            if len(ref_poses) > 0:
                print('ref_poses = ', ref_poses)
    elif FLAGS.stereo and FLAGS.seq_length == 1:
        for i, var_list in enumerate(train_loader):
            tgt_img_l, tgt_img_r, intrinsics_l, intrinsics_r, intrinsics_inv_l, intrinsics_inv_r, trg_pose, \
            filename_tgt, baseline = var_list
            print('\n', i, 'tgt_img_l = ', tgt_img_l.shape)
            print('tgt_img_r = ', tgt_img_r.shape)
            print('intrinsics_l = ', intrinsics_l)
            print('intrinsics_r = ', intrinsics_r)
            print('baseline = ', baseline)
            print('filename_tgt = ', filename_tgt)
            if trg_pose is not None:
                print('trg_pose = ', trg_pose)
    elif FLAGS.stereo and FLAGS.seq_length > 1:
        for i, var_list in enumerate(train_loader):
            tgt_img_l, ref_imgs_l, tgt_img_r, ref_imgs_r, intrinsics_l, intrinsics_r, intrinsics_inv_l, \
            intrinsics_inv_r, trg_pose, ref_poses, filename_tgt, filenames_ref, baseline = var_list
            print('\n', i, 'tgt_img_l = ', tgt_img_l.shape)
            print('ref_imgs_l = ', len(ref_imgs_l), ref_imgs_l[0].shape)
            print('tgt_img_r = ', tgt_img_r.shape)
            print('ref_imgs_r = ', len(ref_imgs_r), ref_imgs_r[0].shape)
            print('intrinsics_l = ', intrinsics_l)
            print('intrinsics_r = ', intrinsics_r)
            print('baseline = ', baseline)
            print('filename_tgt = ', filename_tgt)
            if trg_pose is not None:
                print('trg_pose = ', trg_pose)
            print('filenames_ref = ', filenames_ref)
            if len(ref_poses) > 0:
                print('ref_poses = ', ref_poses)











