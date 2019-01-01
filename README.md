# README

## Depth Estimation
Monocular/stereo depth estimation with regression. Trained and evaluated on the raw KITTY dataset.
For training, dense depth maps are generated with a [Sparse-to-Dense Network](https://arxiv.org/abs/1709.07492)
([PyTorch implementation](https://github.com/yxgeee/DepthComplete)). Evaluation is on sparse 
depth maps (Eigen split). The following models are available: DispNetS, ResNet autoencoder with 18, 34, 
50, 101 and 152 layers.

Loss includes:
- regression loss (reversed Huber loss)
- smoothing loss
- occlusion loss
- disparity consistency loss (in stereo mode) 

The network was inspired by the following papers:
1. [Laina et al. "Deeper Depth Prediction with Fully Convolutional Residual Networks"](https://arxiv.org/abs/1606.00373) (2016)
2. [Godard et al. "Unsupervised Monocular Depth Estimation with Left-Right Consistency"](https://arxiv.org/abs/1609.03677) (2016)
3. [Kuznietsov et al. "Semi-Supervised Deep Learning for Monocular Depth Map Prediction"](https://arxiv.org/abs/1702.02706) (2017)
4. [Radwan et al. "VLocNet++ Deep Multitask Learning for Semantic Visual Localization and Odometry"](https://arxiv.org/abs/1804.08366) (2018)
5. [Godard et al. "Digging Into Self-Supervised Monocular Depth Estimation"](https://arxiv.org/abs/1806.01260) (2018)
6. [Yang et al. "Deep Virtual Stereo Odometry Leveraging Deep Depth Prediction for Monocular Direct Sparse Odometry"](https://arxiv.org/abs/1807.02570) (2018)

## Dependencies
- Python 3
- PyTorch 1.0

Training and testing were performed on Ununtu 16.04, Cuda 8.0 and 1080Ti GPU.


## Usage

### Downloads
- Clone the code
```
git clone https://github.com/victoriamazo/depth_regression.git
```
- Download [KITTY raw dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php)
- Download a pretrained DispNetS model (mono) [here](https://drive.google.com/open?id=1fgBdfvdG7--c73KV-BwaAxaHNRQjxP5L)
- Download a pretrained ResNet34 model (mono) [here](https://drive.google.com/open?id=1586WoEmKay-EGo1aX5Qs2C0h7ws6PEPS)

### Training/testing
All configuration parameters are 
explained in "config/config_params.md".

- Training and testing as parallel threads
```
python3 main.py config/conv.json 
```
- testing 
```
python3 main.py config/conv.json -m test
```
- training
```
python3 main.py config/conv.json -m train
```

### Results

The following results are from evaluation on the raw KITTY dataset (Eigen split):

|    Method                 | Abs Rel  |   Sq Rel    |  RMSE    |  RMSE(log)    |  &delta;1    |   &delta;2  |  &delta;3    |
| :-----------------------: | :----: | :------: | :------: | :------: | :----------: | :----------- | :-------|
| DispNetS (mono)           |0.1850     | 0.6659      |  2.8280  | 0.2193        | 0.7064 | 0.9566 | 0.9909 |
| ResNet34 (mono)             |0.2665     | 1.1139      |  3.6226  | 0.3720        | 0.4883 | 0.8079 | 0.9120 |


Qualitative results:

<pre>          Input                          Ground Truth                            DispnetS      </pre>
![alt-text-1](https://github.com/victoriamazo/depth_regression/blob/master/images/DispNetS/img_comb_143000_0000000000.jpg "DispNetS") 
![alt-text-1](https://github.com/victoriamazo/depth_regression/blob/master/images/DispNetS/img_comb_143000_0000000005.jpg "DispNetS") 
![alt-text-1](https://github.com/victoriamazo/depth_regression/blob/master/images/DispNetS/img_comb_143000_0000000006.jpg "DispNetS") 
![alt-text-1](https://github.com/victoriamazo/depth_regression/blob/master/images/DispNetS/img_comb_143000_0000000009.jpg "DispNetS") 
![alt-text-1](https://github.com/victoriamazo/depth_regression/blob/master/images/DispNetS/img_comb_143000_0000000010.jpg "DispNetS") 
![alt-text-1](https://github.com/victoriamazo/depth_regression/blob/master/images/DispNetS/img_comb_143000_0000000011.jpg "DispNetS") 
![alt-text-1](https://github.com/victoriamazo/depth_regression/blob/master/images/DispNetS/img_comb_143000_0000003992.jpg "DispNetS") 


<pre>          Input                          Ground Truth                            DispnetS      </pre>
![alt-text-1](https://github.com/victoriamazo/depth_regression/blob/master/images/ResNet34/img_comb_262000_0000000000.jpg "ResNet34") 
![alt-text-1](https://github.com/victoriamazo/depth_regression/blob/master/images/ResNet34/img_comb_262000_0000000005.jpg "ResNet34") 
![alt-text-1](https://github.com/victoriamazo/depth_regression/blob/master/images/ResNet34/img_comb_262000_0000000006.jpg "ResNet34") 
![alt-text-1](https://github.com/victoriamazo/depth_regression/blob/master/images/ResNet34/img_comb_262000_0000000009.jpg "ResNet34") 
![alt-text-1](https://github.com/victoriamazo/depth_regression/blob/master/images/ResNet34/img_comb_262000_0000000010.jpg "ResNet34") 
![alt-text-1](https://github.com/victoriamazo/depth_regression/blob/master/images/ResNet34/img_comb_262000_0000000011.jpg "ResNet34") 
![alt-text-1](https://github.com/victoriamazo/depth_regression/blob/master/images/ResNet34/img_comb_262000_0000003992.jpg "ResNet34") 
