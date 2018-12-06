# Config file structure

###### general
* **data_dir**: (required) root directory, whose subdirectories are *'train', 'test', ('val')*
* **train_dir**: (required) directory to save checkpoints and tensorboard visualization
* **data_loader**: (required) e.g., *"mnist_loader"*
* **model**: e.g., *"mpl"*
* **metric**: e.g., *"accuracy_categ"*
* **height**: height (in pixels) of input image
* **width**: width (in pixels) of input image
* **num_classes**: 
* **n_hiddens**: e.g., *"256,256"*
* **batch_norm**: *0* is false, *1* is true
* **initialization**: e.g., *"xavier"*
* **num_channels**: *1* for B&W, *3* for a colored image
* **seed**: random seed (int)
* **exp_description**: unique description of the experiment, e.g. *"data_loader,height,width"*

###### train
* **version**: (required) e.g., *"train_MPL"*
* **batch_size**: (required) 
* **num_epochs**: (required)
* **num_iters_for_ckpt**: (required)
* **lr**: (required) learning rate
* **metric**: e.g., *"accuracy"*
* **gpus**: "1",
* **loss**: "crossentropy_loss"
* **decreasing_lr_epochs**: "80,90",
* **load_ckpt**: for weight initialization with pretrained weights. 
Should be given full path (including ckpt full name) or ""
* **keep_prob**: 0.8,
* **weight_decay**: 0.0001,
* **crop_min_h**: 1.0,
* **crop_min_w**: 1.0,
* **hflip**: 0

###### test
* **version**: "test_MPL",
* **load_ckpt**: Should be given full path (including ckpt full name) or "". *train_dir* will be updated correspondingly 
* **batch_size**: 1,
* **gpus**: "0",
* **sleep_time_sec**: 30

###### val
* **version**: *"test_MPL"*
* **load_ckpt**: Should be given full path (including ckpt full name) or "". *train_dir* will be updated correspondingly
* **val**: "_val",
* **batch_size**: 1,
* **gpus**: "0"

