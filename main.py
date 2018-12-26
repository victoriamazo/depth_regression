'''
Main allows to run training and test as separate processes or train, test, val serially.
Configuration should be specified in json file, given as an argument.
Possible running multiple tests on ckpts given in a list or in a range.

Required to update bashrc (once):
            $ nano ~/.bashrc
            $ export CUDA_DEVICE_ORDER=PCI_BUS_ID
            $cd ~  $source ~/.bashrc

Examples:
* train and test as separate processes:
    config/poseest/KITTY_odom_zhan.json
* serially:
    config/poseest/KITTY_odom_zhan.json -m train
    config/poseest/KITTY_odom_zhan.json -m test
    config/poseest/KITTY_odom_zhan.json -m traintest

By adding '--debug', no tensorboard will start
'''


import argparse
import glob
import json
import multiprocessing
import os
import time
from argparse import Namespace
from time import sleep

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = '1'
from trains.train_builder import Train
from tests.test_builder import Test


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='DIR', help="(required) Path to config file")
    parser.add_argument('-m', '--mode', type=str, default='',
                        help="(optional) 'train', 'test', 'val', 'traintest' (in serial case only)")
    parser.add_argument('--debug', action='store_true', help='Deactivation of tensorboard and csv writers for debugging')
    args = parser.parse_args()
    return args


def input_params(args=None):
    # get args, check them and write them to json file
    if args is None:
        args = getArgs()
    config_filename = args.config
    mode = args.mode

    # read train/test/val configuration dictionaries
    present_dir = os.getcwd()
    config_path = os.path.join(present_dir, config_filename)
    with open(config_path, 'rt') as r:
        print('config_path = ', config_path)
        config = json.load(r)
    config_general = config['general']
    debug = False
    if args.debug:
        debug = True
    config_train = config['train']
    config_train.update(config_general)
    config_test = config['test']
    config_test.update(config_general)
    config_val = {}
    if 'val' in config:
        config_val = config['val']
        config_val.update(config_general)
    root_dir = config_general['train_dir']
    config_train_dir = {}

    # create name for current training dir (incl. time and description) if training
    if mode == '' or mode == 'train' or mode == 'traintest':
        exp_time = str(time.strftime('%y%m%b%d_%H-%M-%S', time.localtime(time.time())))
        if 'exp_description' in config_train:
            config_name = (config_filename.split('/')[-1]).split('.')[0]
            exp_time += '_' + config_name
            for item in config_train['exp_description'].split(','):
                exp_time += '_' + str(item)[0] + str(config_train[str(item)])
        # if resume training, train_dir and all parameters should stay the same for train and test
        if 'load_ckpt' in config_train and config_train['load_ckpt'] != '':
            config_train_dir['load_ckpt'] = config_train['load_ckpt']
            config_general['train_dir'] = "/".join((config_train['load_ckpt']).split("/")[:-2])
            config_train_path = os.path.join(config_general['train_dir'], 'args_train.json')
            if os.path.isfile(config_train_path):
                with open(config_train_path, 'rt') as r:
                    config_train = json.load(r)
                print('loaded train config from {}'.format(config_train_path))
            config_test_path = os.path.join(config_general['train_dir'], 'args_test.json')
            if os.path.isfile(config_test_path):
                with open(config_test_path, 'rt') as r:
                    config_test = json.load(r)
                print('loaded test config from {}'.format(config_test_path))
        else:
            config_general['train_dir'] = os.path.join(config_general['train_dir'], exp_time)
        config_train_dir['train_dir'] = config_general['train_dir']
        config_train.update(config_train_dir)
        config_test.update(config_train_dir)
        if 'val' in config:
            config_val.update(config_train_dir)
        # if train/test in parallel, ignore loading certain ckpt in test (it will load always the last ckpt)
        config_test['load_ckpt'] = ''
    # if mode is 'test' and no 'load_ckpt' choose latest train_dir and load config
    elif 'load_ckpt' in config_test and config_test['load_ckpt'] != '':
        train_dir = "/".join((config_test['load_ckpt']).split("/")[:-2])
        load_ckpt = config_test['load_ckpt']
        config_test_path = os.path.join(train_dir, 'args_test.json')
        if os.path.isfile(config_test_path):
            with open(config_test_path, 'rt') as r:
                config_test = json.load(r)
            print('loaded test config from {}'.format(config_test_path))
        config_test['train_dir'] = train_dir
        config_test['load_ckpt'] = load_ckpt
    else:
        # load ckpt from the latest created train_dir
        root_dir = "/".join(config_general['train_dir'].split("/")[:-1])
        config_test['train_dir'] = sorted(glob.glob(os.path.join(root_dir, '*/')), key=os.path.getmtime)[-1]

    config_train['debug'] = debug
    config_test['debug'] = debug
    if 'val' in config:
        config_val['debug'] = debug

    return config_train, config_test, config_val, mode, root_dir


def set_cuda_visible_gpus(config, mode, gpu_list_train=None):
    if 'gpus' in config and len(config['gpus']) > 0:
        gpus_list = config['gpus']
        print('running {} on gpu(s) {}'.format(mode, gpus_list))
        gpus_list = list(map(int, gpus_list.split(',')))
        gpu_list = ','.join([str(i) for i in gpus_list])
        if gpu_list_train != None:
            for gpu in gpu_list:
                assert gpu not in gpu_list_train
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''  # run on cpu
        gpu_list = []
        print('running {} on cpu'.format(mode))
    return gpu_list


def main(args=None):
    # input parameters (dictionary)
    config_train, config_test, config_val, mode, root_dir = input_params(args=args)

    # run train/test as separate processes
    if mode == '':
        print('running train and test as multiprocesses')

        # start train process
        gpu_list_train = set_cuda_visible_gpus(config_train, 'train')
        train_process = multiprocessing.Process(target=Train.train_builder, args=(Namespace(**config_train), ), name='train')
        train_process.start()

        # wait sleep_time_sec minutes and start test process, run test every sleep_time_sec minutes
        if config_test['version'] != 'test_none':
            while train_process.is_alive():
                set_cuda_visible_gpus(config_test, 'test', gpu_list_train=gpu_list_train)
                sleep(int(config_test["sleep_time_sec"]))
                test_process = multiprocessing.Process(target=Test.test_builder, args=(Namespace(**config_test),), name='test')
                test_process.start()
                test_process.join()
        else:
            print('no test')

        train_process.join()
        # run test on the last checkpoint
        if config_test['version'] != 'test_none':
            test_process = multiprocessing.Process(target=Test.test_builder, args=(Namespace(**config_test),), name='test')
            test_process.start()
            test_process.join()

    # run train/test serially
    elif mode == 'test' or mode == 'train' or mode == 'traintest' or mode == 'val':
        print('running {} serially'.format(mode))

        # run train/test/val/eval once (cpu or gpu)
        if mode == 'train' or mode == 'traintest':
            set_cuda_visible_gpus(config_train, 'train')
            Train.train_builder(Namespace(**config_train), )
        if mode == 'test' or mode == 'traintest':
            if 'gpus' in config_test:
                set_cuda_visible_gpus(config_test, 'test')
            Test.test_builder(Namespace(**config_test), )
        elif mode == 'val':
            if 'gpus' in config_val:
                set_cuda_visible_gpus(config_val, 'val')
            print('validation')
            Test.test_builder(Namespace(**config_val), )
    else:
        'wrong arguments given'



if __name__ == '__main__':
    main()