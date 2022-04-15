#!/usr/bin/env python

# A sample training component that trains a simple scikit-learn decision tree model.
# This implementation works in File mode and makes no assumptions about the input file names.
# Input is specified as CSV with a data point in each row and the labels in the first column.

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import os
import pickle
import sys
import traceback

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append('./PaddleOCR')

import yaml
import paddle
import paddle.distributed as dist
from paddle.jit import to_static

paddle.seed(2)

from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
#from ppocr.utils.save_load import load_model
from ppocr.utils.save_load import init_model
import tools.program as program
from tools.program import merge_config,load_config,check_gpu
from ppocr.utils.logging import get_logger

# These are the paths to where SageMaker mounts interesting things in your container.

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
channel_name='training'
training_path = os.path.join(input_path, channel_name)

# The function to execute the training.
default_config = {'Global': {'debug': False, }}

def load_train_config(config_path, is_train=False):
    config = load_config(config_path)
    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    check_gpu(use_gpu)

    alg = config['Architecture']['algorithm']
    assert alg in [
        'EAST', 'DB', 'SAST', 'Rosetta', 'CRNN', 'STARNet', 'RARE', 'SRN', 'CLS'
    ]

    device = 'gpu:{}'.format(dist.ParallelEnv().dev_id) if use_gpu else 'cpu'
    device = paddle.set_device(device)

    config['Global']['distributed'] = dist.get_world_size() != 1
    if is_train:
        # save_config
        save_model_dir = config['Global']['save_model_dir']
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, 'config.yml'), 'w') as f:
            yaml.dump(
                dict(config), f, default_flow_style=False, sort_keys=False)
        log_file = '{}/train.py.log'.format(save_model_dir)
    else:
        log_file = None
    logger = get_logger(name='root', log_file=log_file)
    if config['Global']['use_visualdl']:
        from visualdl import LogWriter
        save_model_dir = config['Global']['save_model_dir']
        vdl_writer_path = '{}/vdl/'.format(save_model_dir)
        os.makedirs(vdl_writer_path, exist_ok=True)
        vdl_writer = LogWriter(logdir=vdl_writer_path)
    else:
        vdl_writer = None
    logger.info('train.py with paddle {} and device {}'.format(paddle.__version__,
                                                            device))
    return config, device, logger, vdl_writer


def export_single_model(model, arch_config, save_path, logger):
    if arch_config["algorithm"] == "SRN":
        max_text_length = arch_config["Head"]["max_text_length"]
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 1, 64, 256], dtype="float32"), [
                    paddle.static.InputSpec(
                        shape=[None, 256, 1],
                        dtype="int64"), paddle.static.InputSpec(
                            shape=[None, max_text_length, 1], dtype="int64"),
                    paddle.static.InputSpec(
                        shape=[None, 8, max_text_length, max_text_length],
                        dtype="int64"), paddle.static.InputSpec(
                            shape=[None, 8, max_text_length, max_text_length],
                            dtype="int64")
                ]
        ]
        model = to_static(model, input_spec=other_shape)
    elif arch_config["algorithm"] == "SAR":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 48, 160], dtype="float32"),
        ]
        model = to_static(model, input_spec=other_shape)
    else:
        infer_shape = [3, -1, -1]
        if arch_config["model_type"] == "rec":
            infer_shape = [3, 32, -1]  # for rec model, H must be 32
            if "Transform" in arch_config and arch_config[
                    "Transform"] is not None and arch_config["Transform"][
                        "name"] == "TPS":
                logger.info(
                    "When there is tps in the network, variable length input is not supported, and the input size needs to be the same as during training"
                )
                infer_shape[-1] = 100
            if arch_config["algorithm"] == "NRTR":
                infer_shape = [1, 32, 100]
        elif arch_config["model_type"] == "table":
            infer_shape = [3, 488, 488]
        model = to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None] + infer_shape, dtype="float32")
            ])

    paddle.jit.save(model, save_path)
    logger.info("inference model is saved to {}".format(save_path))
    return


def train():
    print('Starting the training.')
    try:
        # Read in any hyperparameters that the user passed with the training job
        with open(param_path, 'r') as tc:
            trainingParams = json.load(tc)
        #update parameters to int
        for key in trainingParams:
            try:
                trainingParams[key] = int(trainingParams[key])
            except:
                continue

        # Take the set of files and read them all into a single pandas dataframe
        input_files = [ os.path.join(training_path, file) for file in os.listdir(training_path) ]
        print ("<<<< input files: ", input_files)
        if len(input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(training_path, channel_name))

        # todo: change the data directory
        config, device, logger, vdl_writer = load_train_config('./rec_chinese_lite_train_v2.0_hyperparam.yml')
        # init dist environment
        if config['Global']['distributed']:
            dist.init_parallel_env()

        global_config = config['Global']
        #replace training params with your own configs
        global_config.update(trainingParams)

        # build dataloader
        train_dataloader = build_dataloader(config, 'Train', device, logger)
        if len(train_dataloader) == 0:
            logger.error(
                'No Images in train.py dataset, please check annotation file and path in the configuration file'
            )

        if config['Eval']:
            valid_dataloader = build_dataloader(config, 'Eval', device, logger)
        else:
            valid_dataloader = None

        # build post process
        post_process_class = build_post_process(config['PostProcess'],
                                                global_config)

        '''
        # build model
        # for rec algorithm
        if hasattr(post_process_class, 'character'):
            char_num = len(getattr(post_process_class, 'character'))
            config['Architecture']["Head"]['out_channels'] = char_num

        model = build_model(config['Architecture'])
        logger.info('train.py dataloader has {} iters, valid dataloader has {} iters'.
                    format(len(train_dataloader), len(valid_dataloader)))
        # build loss
        loss_class = build_loss(config['Loss'])

        # build optim
        optimizer, lr_scheduler = build_optimizer(
            config['Optimizer'],
            epochs=config['Global']['epoch_num'],
            step_each_epoch=len(train_dataloader),
            parameters=model.parameters())

        # build metric
        eval_class = build_metric(config['Metric'])
        # load pretrain model
        pre_best_model_dict = load_model(config, model, optimizer)
        logger.info('train dataloader has {} iters'.format(len(train_dataloader)))

        if valid_dataloader is not None:
            logger.info('valid dataloader has {} iters'.format(
                len(valid_dataloader)))

        use_amp = config["Global"].get("use_amp", False)
        if use_amp:
            AMP_RELATED_FLAGS_SETTING = {
                'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
                'FLAGS_max_inplace_grad_add': 8,
            }
            paddle.fluid.set_flags(AMP_RELATED_FLAGS_SETTING)
            scale_loss = config["Global"].get("scale_loss", 1.0)
            use_dynamic_loss_scaling = config["Global"].get(
                "use_dynamic_loss_scaling", False)
            scaler = paddle.amp.GradScaler(
                init_loss_scaling=scale_loss,
                use_dynamic_loss_scaling=use_dynamic_loss_scaling)
        else:
            scaler = None

        # start train.py
        config['profiler_options'] = None
    '''

            # build model
        # for rec algorithm
        if hasattr(post_process_class, 'character'):
            char_num = len(getattr(post_process_class, 'character'))
            config['Architecture']["Head"]['out_channels'] = char_num
        model = build_model(config['Architecture'])
        if config['Global']['distributed']:
            model = paddle.DataParallel(model)

        # build loss
        loss_class = build_loss(config['Loss'])

        # build optim
        optimizer, lr_scheduler = build_optimizer(
            config['Optimizer'],
            epochs=config['Global']['epoch_num'],
            step_each_epoch=len(train_dataloader),
            parameters=model.parameters())

        # build metric
        eval_class = build_metric(config['Metric'])
        # load pretrain model
        pre_best_model_dict = init_model(config, model, logger, optimizer)

        logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
        if valid_dataloader is not None:
            logger.info('valid dataloader has {} iters'.format(
                len(valid_dataloader)))

        program.train(config, train_dataloader, valid_dataloader, device, model,
                      loss_class, optimizer, lr_scheduler, post_process_class,
                      eval_class, pre_best_model_dict, logger, vdl_writer)
        print('Training complete.')

        #export model for inference
        save_path = os.path.join(model_path, "inference")
        arch_config = config["Architecture"]
        export_single_model(model, arch_config, save_path, logger)
        print('convert model complete.')

    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)

if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
