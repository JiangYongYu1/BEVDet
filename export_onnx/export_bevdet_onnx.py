# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from copy import deepcopy
import mmdet3d
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (load_checkpoint, wrap_fp16_model)
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmcv.parallel import collate, scatter
import _init_path
from mmxdeploy.core import RewriterContext
from mmxdeploy.codebase.bevdet import *
from thop import profile
from torchstat import stat

from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument("--save_onnx_path", type=str, required=True, help="onnx save dir")
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args = parser.parse_args()
    return args


def read_data(model, dataset, index):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    # test_pipeline = deepcopy(cfg.data.test.pipeline)
    # test_pipeline = Compose(test_pipeline)
    data = dataset[index]
    # data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
        data['img_inputs'] = data['img_inputs'][0].data
    return data


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    model.cfg = cfg
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.eval()

    data = read_data(model, dataset, 0)
    img_metas = data['img_metas'][0]
    points = data['points'][0]
    img_inputs = data['img_inputs'][0]

    trans_cfg = cfg.model.img_view_transformer
    img_view_transformer = mmdet3d.models.builder.build_neck(trans_cfg).to(
        device)

    with RewriterContext(cfg={}, backend='default'), torch.no_grad():
        volume = model.extract_pseudo_cloud(img_inputs[0], img_metas[0])
        rots, trans, intrins, post_rots, post_trans = img_inputs[1:]
        bev_feat = img_view_transformer.splat(volume, rots, trans, intrins,
                                              post_rots, post_trans)
        outs = model.detector(bev_feat)
        result = model.postprocess(outs, img_metas)
        print(result)

        out_onnx_f = Path(args.save_onnx_path) / 'bevdet_pseudo_cloud.onnx'
        model.forward = model.extract_pseudo_cloud
        torch.onnx.export(
            model,
            img_inputs[0],
            out_onnx_f,
            export_params=True,
            opset_version=12,
            input_names=['inputs'],
            output_names=['volume'],
            verbose=False)

        output_names = []
        for task_id in range(6):
            for out_name in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                output_names.append(f'{out_name}_{task_id}')
        out_onnx_f = Path(args.save_onnx_path) / 'bevdet_detector.onnx'
        model.forward = model.detector
        torch.onnx.export(
            model,
            bev_feat,
            out_onnx_f,
            export_params=True,
            opset_version=12,
            input_names=['bev_feat'],
            output_names=output_names,
            verbose=False)


if __name__ == '__main__':
    main()
