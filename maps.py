# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

def normalize_class_name(class_name):

    if class_name.startswith('inf_'):
        class_name = class_name[4:]
    return class_name

def parse_box_from_line(line):
    """
    GT: class 0 0 0 0 0 0 0 height width length cx cy cz yaw
    Pred: class 0 0 0 0 0 0 0 width height length cx cy cz yaw confidence
    """
    try:
        parts = line.strip().split()
        if len(parts) < 15:  # minimum required fields
            return None
            
        box = {
            'class': normalize_class_name(parts[0]),
            'height': float(parts[8]),
            'width': float(parts[9]),
            'length': float(parts[10]),
            'cx': float(parts[11]),
            'cy': float(parts[12]),
            'cz': float(parts[13]),
            'yaw': float(parts[14])
        }
        
        # confidence is only in prediction
        if len(parts) > 15:
            box['confidence'] = float(parts[15])
            
        return box
            
    except (IndexError, ValueError) as e:
        print(f"Warning: Failed to parse line: {line}")
        return None

def calculate_3d_iou(box1, box2):
    if box1['class'] != box2['class']:
        return 0.0
        
    box1_min = np.array([
        box1['cx'] - box1['length']/2,
        box1['cy'] - box1['width']/2,
        box1['cz'] - box1['height']/2
    ])
    box1_max = np.array([
        box1['cx'] + box1['length']/2,
        box1['cy'] + box1['width']/2,
        box1['cz'] + box1['height']/2
    ])
    
    box2_min = np.array([
        box2['cx'] - box2['length']/2,
        box2['cy'] - box2['width']/2,
        box2['cz'] - box2['height']/2
    ])
    box2_max = np.array([
        box2['cx'] + box2['length']/2,
        box2['cy'] + box2['width']/2,
        box2['cz'] + box2['height']/2
    ])
    
    intersect_min = np.maximum(box1_min, box2_min)
    intersect_max = np.minimum(box1_max, box2_max)
    
    if np.any(intersect_max < intersect_min):
        return 0.0
    
    intersect_volume = np.prod(intersect_max - intersect_min)
    box1_volume = np.prod(box1_max - box1_min)
    box2_volume = np.prod(box2_max - box2_min)
    
    iou = intersect_volume / (box1_volume + box2_volume - intersect_volume)
    
    return iou

def calculate_2d_iou(box1, box2):
    """
    calculate 2D IoU(only x, y plane)
    """
    if box1['class'] != box2['class']:
        return 0.0
        
    box1_min = np.array([
        box1['cx'] - box1['length']/2,
        box1['cy'] - box1['width']/2
    ])
    box1_max = np.array([
        box1['cx'] + box1['length']/2,
        box1['cy'] + box1['width']/2
    ])
    
    box2_min = np.array([
        box2['cx'] - box2['length']/2,
        box2['cy'] - box2['width']/2
    ])
    box2_max = np.array([
        box2['cx'] + box2['length']/2,
        box2['cy'] + box2['width']/2
    ])
    
    intersect_min = np.maximum(box1_min, box2_min)
    intersect_max = np.minimum(box1_max, box2_max)
    
    if np.any(intersect_max < intersect_min):
        return 0.0
    
    intersect_area = np.prod(intersect_max - intersect_min)
    box1_area = np.prod(box1_max - box1_min)
    box2_area = np.prod(box2_max - box2_min)
    
    iou = intersect_area / (box1_area + box2_area - intersect_area)
    
    return iou

def calculate_ap_from_recalls(precisions, recalls):
    """
    COCO style 101-point interpolation AP
    """
    if not precisions or not recalls:
        return 0.0
    
    if max(recalls) == 0:  # if all recalls are 0
        return 0.0
        
    recall_points = np.linspace(0, 1, 101)
    max_precisions = []
    
    for r in recall_points:
        precs = [p for rec, p in zip(recalls, precisions) if rec >= r]
        max_prec = max(precs) if precs else 0
        max_precisions.append(max_prec)
    
    ap = np.mean(max_precisions)
    return ap

def calculate_3d_map_for_folder(gt_folder, pred_folder, iou_threshold=0.1):
    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))
    
    print(f"Ground Truth 파일 수: {len(gt_files)}")
    print(f"Prediction 파일 수: {len(pred_files)}")
    
    class_stats = {}
    
    for gt_file in gt_files:
        if gt_file in pred_files:
            gt_boxes = []
            with open(os.path.join(gt_folder, gt_file), 'r') as f:
                for line in f:
                    box = parse_box_from_line(line)
                    if box:
                        gt_boxes.append(box)
            
            pred_boxes = []
            with open(os.path.join(pred_folder, gt_file), 'r') as f:
                for line in f:
                    box = parse_box_from_line(line)
                    if box:
                        pred_boxes.append(box)
            
            if not gt_boxes or not pred_boxes:
                continue
                
            for box in gt_boxes:
                cls = box['class']
                if cls not in class_stats:
                    class_stats[cls] = {
                        'total_gt': 0,
                        'predictions': []
                    }
                class_stats[cls]['total_gt'] += 1
            
            for cls in set(box['class'] for box in gt_boxes + pred_boxes):
                cls_gt_boxes = [box for box in gt_boxes if box['class'] == cls]
                cls_pred_boxes = [box for box in pred_boxes if box['class'] == cls]
                
                if not cls_gt_boxes:
                    for pred_box in cls_pred_boxes:
                        if cls not in class_stats:
                            class_stats[cls] = {'total_gt': 0, 'predictions': []}
                        class_stats[cls]['predictions'].append(
                            (pred_box['confidence'], False)
                        )
                    continue
                
                if not cls_pred_boxes:
                    continue
                
                iou_matrix = np.zeros((len(cls_gt_boxes), len(cls_pred_boxes)))
                for i, gt in enumerate(cls_gt_boxes):
                    for j, pred in enumerate(cls_pred_boxes):
                        iou = calculate_3d_iou(gt, pred)
                        iou_matrix[i, j] = iou
                
                cost_matrix = 1 - iou_matrix
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                matched_pred_indices = set()
                for gt_idx, pred_idx in zip(row_indices, col_indices):
                    iou = iou_matrix[gt_idx, pred_idx]
                    if iou >= iou_threshold:
                        class_stats[cls]['predictions'].append(
                            (cls_pred_boxes[pred_idx]['confidence'], True)
                        )
                        matched_pred_indices.add(pred_idx)
                
                for pred_idx, pred_box in enumerate(cls_pred_boxes):
                    if pred_idx not in matched_pred_indices:
                        class_stats[cls]['predictions'].append(
                            (pred_box['confidence'], False)
                        )
    
    aps = []
    print(f"\n클래스별 결과 (3D IoU {iou_threshold} 기준):")
    
    total_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    
    for cls, stats in class_stats.items():
        if stats['total_gt'] == 0 or not stats['predictions']:
            continue
            
        sorted_predictions = sorted(stats['predictions'], 
                                 key=lambda x: x[0], 
                                 reverse=True)
        
        total_gt = stats['total_gt']
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        
        for i, (conf, is_matched) in enumerate(sorted_predictions, 1):
            if is_matched:
                tp += 1
            else:
                fp += 1
            
            precision = tp / i
            recall = tp / total_gt
            
            precisions.append(precision)
            recalls.append(recall)
        
        # FN 계산
        fn = total_gt - tp

        total_metrics['tp'] += tp
        total_metrics['fp'] += fp
        total_metrics['fn'] += fn
        
        final_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        final_recall = tp / total_gt if total_gt > 0 else 0
        
        ap = calculate_ap_from_recalls(precisions, recalls)
        aps.append(ap)
        
    
    if total_metrics['tp'] + total_metrics['fp'] > 0:
        total_precision = total_metrics['tp'] / (total_metrics['tp'] + total_metrics['fp'])
        print(f"Total Precision: {total_precision:.4f}")
    
    if total_metrics['tp'] + total_metrics['fn'] > 0:
        total_recall = total_metrics['tp'] / (total_metrics['tp'] + total_metrics['fn'])
        print(f"Total Recall: {total_recall:.4f}")
    
    if aps:
        mAP = np.mean(aps)
        print(f"\n3D mAP: {mAP:.4f}")
        return mAP
    else:
        print("\n매칭된 클래스가 없습니다.")
        return 0.0


def decode_boxes(reg_preds, anchors):
    """
    reg_preds: (B, H*W*A, 7) - predicted offsets (dx, dy, dz, dh, dw, dl, dr)
    anchors: (B, H*W*A, 7) - anchor boxes (x, y, z, h, w, l, r)
    """
    # Shape 확인 및 변환
    if len(reg_preds.shape) == 5:  # (B, H, W, A, 7)
        B, H, W, A, _ = reg_preds.shape
        reg_preds = reg_preds.reshape(B, H*W*A, 7)
    
    if len(anchors.shape) == 5:    # (B, H, W, A, 7)
        B, H, W, A, _ = anchors.shape
        anchors = anchors.reshape(B, H*W*A, 7)
    
    # Center coordinates
    x = reg_preds[..., 0] * anchors[..., 3] + anchors[..., 0]  # dx * h + x_a
    y = reg_preds[..., 1] * anchors[..., 4] + anchors[..., 1]  # dy * w + y_a
    z = reg_preds[..., 2] * anchors[..., 5] + anchors[..., 2]  # dz * l + z_a
    
    # Dimensions (h, w, l 순서)
    h = torch.exp(reg_preds[..., 3]) * anchors[..., 3]  # exp(dh) * h_a
    w = torch.exp(reg_preds[..., 4]) * anchors[..., 4]  # exp(dw) * w_a
    l = torch.exp(reg_preds[..., 5]) * anchors[..., 5]  # exp(dl) * l_a
    
    # Rotation
    r = reg_preds[..., 6] + anchors[..., 6]  # dr + r_a
    
    return torch.stack([x, y, z, h, w, l, r], dim=-1)

def nms_3d(boxes, scores, iou_threshold):
    """
    3D Non-Maximum Suppression
    Args:
        boxes: (N, 7) tensor of boxes [x,y,z,l,w,h,r]
        scores: (N,) tensor of scores
        iou_threshold: IoU threshold for NMS
    Returns:
        keep: indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    
    # Sort boxes by score
    scores, order = scores.sort(descending=True)
    boxes = boxes[order]
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0])
            break
        
        # Keep the box with highest score
        keep.append(order[0])
        
        # Calculate IoU with rest of boxes
        ious = calculate_3d_iou(boxes[0:1], boxes[1:])
        
        # Remove boxes with IoU > threshold
        mask = ious <= iou_threshold
        order = order[1:][mask]
        boxes = boxes[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def post_process_predictions(cls_preds, reg_preds, anchors, score_threshold=0.1, nms_iou_threshold=0.5):
    """
    cls_preds: (B, H*W*A, num_classes)
    reg_preds: (B, H*W*A, 7)
    anchors: (B, H*W*A, 7)
    """
    batch_size = cls_preds.shape[0]
    predictions = []
    
    # anchors를 reg_preds와 같은 device로 이동
    anchors = anchors.to(reg_preds.device)
    
    for b in range(batch_size):
        # Get scores and boxes for current batch
        scores, class_ids = cls_preds[b].softmax(-1).max(-1)
        boxes = decode_boxes(reg_preds[b:b+1], anchors[b:b+1]).squeeze(0)
        
        # Reshape to (N, 7)
        boxes = boxes.reshape(-1, 7)
        scores = scores.reshape(-1)
        class_ids = class_ids.reshape(-1)
        
        # Filter by score threshold
        mask = scores > score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        batch_predictions = []
        
        # Process each class independently
        for class_id in torch.unique(class_ids):
            class_mask = class_ids == class_id
            if not class_mask.any():
                continue
            
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            
            # Apply NMS
            keep = nms_3d(class_boxes, class_scores, nms_iou_threshold)
            
            # Save predictions
            for idx in keep:
                batch_predictions.append({
                    'box': class_boxes[idx].cpu().numpy(),
                    'score': class_scores[idx].cpu().numpy(),
                    'class_id': class_id.cpu().numpy()
                })
        
        predictions.append(batch_predictions)
    
    return predictions

def calculate_3d_iou(boxes1, boxes2):
    """
    Calculate 3D IoU between two sets of boxes
    Args:
        boxes1: (N, 7) first set of boxes
        boxes2: (M, 7) second set of boxes
    Returns:
        ious: (N, M) tensor of pairwise IoUs
    """
    # Convert boxes to corners
    corners1 = boxes_to_corners_3d(boxes1)  # (N, 8, 3)
    corners2 = boxes_to_corners_3d(boxes2)  # (M, 8, 3)
    
    # Calculate intersection volume
    intersections = box_intersection_3d(corners1, corners2)  # (N, M)
    
    # Calculate volumes
    volumes1 = boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]  # (N,)
    volumes2 = boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]  # (M,)
    
    # Calculate union volume
    unions = volumes1.unsqueeze(1) + volumes2.unsqueeze(0) - intersections
    
    # Calculate IoU
    ious = intersections / (unions + 1e-7)
    
    return ious

def save_predictions(predictions, save_dir, file_names):
    os.makedirs(save_dir, exist_ok=True)
    
    for file_name, frame_preds in zip(file_names, predictions):
        save_path = os.path.join(save_dir, f"{file_name}.txt")
        with open(save_path, 'w') as f:
            for pred in frame_preds:
                line = ' '.join(map(str, pred))
                f.write(line + '\n')