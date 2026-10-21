"""教学用检测 AP：全数据集排序、一对一匹配、101 点插值。

包含 AP50 和 IoU 0.50:0.95 的均值。不实现 COCO crowd/area 分组，
因此名称使用 map_50_95_simple，不宣称与官方 COCO evaluator 完全等价。
"""
import numpy as np


def iou(box, boxes):
    """计算一个框与多个框的交并比。"""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not len(boxes):
        return np.empty(0)
    low = np.maximum(box[:2], boxes[:, :2])
    high = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(high-low, 0).prod(axis=1)
    area = np.maximum(box[2:]-box[:2], 0).prod()
    other = np.maximum(boxes[:, 2:]-boxes[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area+other-intersection, 1e-12)


def class_ap(predictions, targets, category, threshold):
    """计算指定类别及 IoU 阈值下的平均精度。"""
    truth, detections = {}, []
    for image_id, (prediction, target) in enumerate(zip(predictions, targets)):
        truth[image_id] = target["boxes"][target["labels"] == category].numpy()
        # 限制每张图每类的 top 100；这里采用简化评估约定。
        indexes = (prediction["labels"] == category).nonzero().flatten()
        indexes = indexes[prediction["scores"][indexes].argsort(descending=True)][:100]
        for index in indexes.tolist():
            detections.append((float(prediction["scores"][index]), image_id, prediction["boxes"][index].numpy()))
    count = sum(len(boxes) for boxes in truth.values())
    if not count:
        return None
    detections.sort(key=lambda item: item[0], reverse=True)
    matched = {image_id: set() for image_id in truth}
    flags = []
    for _, image_id, box in detections:
        overlaps = iou(box, truth[image_id])
        candidates = [index for index in np.argsort(-overlaps) if index not in matched[image_id]]
        hit = bool(candidates and overlaps[candidates[0]] >= threshold)
        if hit:
            matched[image_id].add(candidates[0])
        flags.append(hit)
    if not flags:
        return 0.0
    tp = np.cumsum(flags)
    recall = tp / count
    precision = tp / np.arange(1, len(flags)+1)
    return float(np.mean([precision[recall >= r].max() if (recall >= r).any() else 0
                          for r in np.linspace(0, 1, 101)]))


def detection_metrics(predictions, targets, classes):
    """汇总所有类别的 AP50 与简化 AP50:95。"""
    results = {name: [] for name in classes}
    for threshold in np.linspace(0.5, 0.95, 10):
        for category, name in enumerate(classes, 1):
            results[name].append(class_ap(predictions, targets, category, threshold))
    valid = [scores for scores in results.values() if scores[0] is not None]
    if not valid:
        raise ValueError("评估集没有任何前景标注，无法计算 AP")
    return {"map_50": float(np.mean([scores[0] for scores in valid])),
            "map_50_95_simple": float(np.mean(valid)),
            "per_class_ap50": {name: scores[0] for name, scores in results.items()},
            "classes_without_ground_truth": [name for name, scores in results.items() if scores[0] is None]}
