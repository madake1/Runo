"""统一检测模型构建。复用网络结构，训练流程由 Runo 自己实现。"""
from functools import partial

from torch import nn
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    ssdlite320_mobilenet_v3_large,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

from configs.defaults import MODEL_CHOICES

MODEL_NAMES = MODEL_CHOICES


def build_model(name, num_classes, pretrained=True):
    """num_classes 为前景类别数，内部额外增加背景类别。"""
    if num_classes < 1:
        raise ValueError("至少需要一个前景类别")
    if name == MODEL_NAMES[0]:
        model = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None,
            weights_backbone=None,
            num_classes=None if pretrained else num_classes+1,
            trainable_backbone_layers=6 if pretrained else None)
        if pretrained:
            # 保留 COCO 主干和框回归分支，只替换类别预测分支。
            in_channels = [block[0][0].in_channels for block in model.head.classification_head.module_list]
            anchors = model.anchor_generator.num_anchors_per_location()
            model.head.classification_head = SSDLiteClassificationHead(
                in_channels, anchors, num_classes+1,
                partial(nn.BatchNorm2d, eps=0.001, momentum=0.03))
    elif name in (MODEL_NAMES[1], MODEL_NAMES[2]):
        builder, weights, backbone_layers = {
            MODEL_NAMES[1]: (fasterrcnn_mobilenet_v3_large_320_fpn,
                             FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT, 6),
            MODEL_NAMES[2]: (fasterrcnn_resnet50_fpn_v2,
                             FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 5),
        }[name]
        model = builder(
            weights=weights if pretrained else None,
            weights_backbone=None,
            num_classes=None if pretrained else num_classes+1,
            trainable_backbone_layers=backbone_layers if pretrained else None)
        if pretrained:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
    else:
        raise ValueError(f"未知模型 {name}，可选 {MODEL_NAMES}")
    return model


def freeze_backbone(model, frozen):
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(not frozen)
