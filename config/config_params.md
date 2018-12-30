### Configuration parameters 

- _data_dir_ - full path to the root of a data directory
- _train_dir_ - full path to a training directory (is created if does not exist) 
- _model_ - name of a network ("DispNetS" or "ResNet")
- _num_layers_ - number of layers in ResNet (can be 18, 34, 50, 101 or 152) 
- _xy_cut_ - depth map and image cropping coordinates [x_min, x_max, y_min, y_max]
- _max_depth_ - maximal depth in meters (usually 50 or 80)
- _stereo_ - 0: mono input, 1: stereo input
- _loss_weights_ - weights of losses for training, e.g."DS-1,S-0.1" means depth supervised wight is 1 and
smoothing loss weight is 0.1
- _concat_LR_ - (only in stereo mode) 0: input is a left image, 1 - input is a concatenated left and right 
images 
- _disp_norm_ - 0: no disparity normalization, 1: disparity is normalized by a disparity mean
- _upscaling_ - 0: image is be downscaled, 1: predicted disparities are upscaled to the orignal image size 
- _nonlinearity_ - default is ReLU, other options: "elu", "lrelu" (in DispNetS)
- _version_ - version of a train ("train_unsup") and test ("test_depth") scripts
- _gpus_ - either one or several GPUs for running train/test, e.g. "0,1"
- _lr_ - learning rate
- _decreasing_lr_epochs_ - list of epochs to decrease learning rate by 2, e.g. "15,30,45"
- _num_epochs_ - number of epochs for training 
- _num_iters_for_ckpt_ - number of iterations to save a checkpoint
- _load_ckpt_ - full path to a saved model to resume training / run a test
- _rand_crop_ - random crop of an image and GT depth (not in test)
- _hflip_ - horizontal flip of an image and GT depth 
- _exp_description_ - which parameters to include in the name of a created training directory, e.g. 
"lr,loss_weights"
- _stereo_test_ - 0: mono input, 1: stereo input
- _best_criteria_depth_ - criteria for saving best model: "rmse", "abs_rel", "sq_rel", "log_rmse", 
"a1", "a2" or "a3"
- _sleep_time_sec_ - waiting time for running a test (in sec) in a train/test paralellel mode 


