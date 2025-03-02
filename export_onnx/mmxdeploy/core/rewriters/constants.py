# Copyright (c) OpenMMLab. All rights reserved.
from enum import Enum


class AdvancedEnum(Enum):
    """Define an enumeration class."""

    @classmethod
    def get(cls, value):
        """Get the key through a value."""
        for k in cls:
            if k.value == value:
                return k

        raise KeyError(f'Cannot get key by value "{value}" of {cls}')


class Task(AdvancedEnum):
    """Define task enumerations."""
    TEXT_DETECTION = 'TextDetection'
    TEXT_RECOGNITION = 'TextRecognition'
    SEGMENTATION = 'Segmentation'
    SUPER_RESOLUTION = 'SuperResolution'
    CLASSIFICATION = 'Classification'
    OBJECT_DETECTION = 'ObjectDetection'
    INSTANCE_SEGMENTATION = 'InstanceSegmentation'
    VOXEL_DETECTION = 'VoxelDetection'
    POSE_DETECTION = 'PoseDetection'
    ROTATED_DETECTION = 'RotatedDetection'


class Codebase(AdvancedEnum):
    """Define codebase enumerations."""
    MMDET = 'mmdet'
    MMSEG = 'mmseg'
    MMCLS = 'mmcls'
    MMOCR = 'mmocr'
    MMEDIT = 'mmedit'
    MMDET3D = 'mmdet3d'
    MMPOSE = 'mmpose'
    MMROTATE = 'mmrotate'


class IR(AdvancedEnum):
    """Define intermediate representation enumerations."""
    ONNX = 'onnx'
    TORCHSCRIPT = 'torchscript'
    DEFAULT = 'default'


class Backend(AdvancedEnum):
    """Define backend enumerations."""
    PYTORCH = 'pytorch'
    TENSORRT = 'tensorrt'
    ONNXRUNTIME = 'onnxruntime'
    PPLNN = 'pplnn'
    NCNN = 'ncnn'
    OPENVINO = 'openvino'
    SDK = 'sdk'
    TORCHSCRIPT = 'torchscript'
    DEFAULT = 'default'