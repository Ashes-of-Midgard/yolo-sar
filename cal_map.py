import os
import argparse
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_root', type=str, required=True)
    parser.add_argument('--labels_root', type=str, required=True)
    parser.add_argument('--preds_root', type=str, required=True)
    return parser.parse_args()

def get_image_sizes(images_root):
    image_sizes = {}
    for img_name in os.listdir(images_root):
        try:
            with Image.open(os.path.join(images_root, img_name)) as img:
                base = os.path.splitext(img_name)[0]
                image_sizes[base] = img.size  # (width, height)
        except:
            continue
    return image_sizes

def parse_objects(file_path, image_size, is_label=False):
    with open(file_path) as f:
        lines = f.readlines()
    
    objs = []
    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 8 + (0 if is_label else 1):
            continue
        
        cls_id = int(parts[0])
        points = np.array(parts[1:9]).reshape(4, 2)
        conf = 1.0 if is_label else parts[9]
        
        # Convert normalized to absolute coordinates
        width, height = image_size
        points[:, 0] *= width
        points[:, 1] *= height
        
        # Create polygon
        try:
            poly = Polygon(points)
            if not poly.is_valid:
                continue
        except:
            continue
        
        objs.append({
            'class': cls_id,
            'polygon': poly,
            'conf': conf
        })
    
    if not is_label:
        objs.sort(key=lambda x: x['conf'], reverse=True)
    return objs

def compute_iou(poly1, poly2):
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0

def calculate_ap(gt_objs, pred_objs, iou_threshold):
    # Track matched ground truths
    matched = [False] * len(gt_objs)
    tp = []
    fp = []
    
    for pred in pred_objs:
        best_iou = 0.0
        best_idx = -1
        
        for i, gt in enumerate(gt_objs):
            if matched[i]:
                continue
            iou = compute_iou(pred['polygon'], gt['polygon'])
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        
        if best_iou >= iou_threshold and best_idx != -1:
            matched[best_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    
    # Compute precision/recall
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    
    recall = tp_cum / (len(gt_objs) + 1e-6)
    precision = np.divide(tp_cum, (tp_cum + fp_cum + 1e-6))
    
    # COCO style AP calculation
    ap = 0.0
    for t in np.arange(0, 1.01, 0.01):
        mask = recall >= t
        if np.sum(mask) == 0:
            p = 0
        else:
            p = np.max(precision[mask])
        ap += p / 101
    
    return ap

def evaluate_dataset(gt_data, pred_data):
    results = {}
    
    # For each class
    all_classes = set(gt_data.keys()).union(pred_data.keys())
    for cls_id in all_classes:
        # Get all images containing this class
        class_gt = gt_data.get(cls_id, {})
        class_pred = pred_data.get(cls_id, {})
        
        # Calculate AP for different IoU thresholds
        aps = []
        for iou in np.arange(0.5, 1.0, 0.05):
            total_ap = 0.0
            num_images = 0
            
            # Process each image
            all_images = set(class_gt.keys()).union(class_pred.keys())
            for img in all_images:
                gt_objs = class_gt.get(img, [])
                pred_objs = [p for p in class_pred.get(img, []) if p['class'] == cls_id]
                
                ap = calculate_ap(gt_objs, pred_objs, iou)
                total_ap += ap
                num_images += 1
            
            aps.append(total_ap / (num_images + 1e-6))
        
        # Save results
        results[cls_id] = {
            'AP50': aps[0],
            'AP': np.mean(aps)
        }
    
    return results

def main():
    args = parse_args()
    image_sizes = get_image_sizes(args.images_root)
    
    # Load ground truth
    gt_data = defaultdict(lambda: defaultdict(list))  # gt_data[class][image] = [objs]
    for label_file in os.listdir(args.labels_root):
        if not label_file.endswith('.txt'):
            continue
        base = os.path.splitext(label_file)[0]
        if base not in image_sizes:
            continue
        label_path = os.path.join(args.labels_root, label_file)
        objs = parse_objects(label_path, image_sizes[base], is_label=True)
        for obj in objs:
            gt_data[obj['class']][base].append(obj)
    
    # Load predictions
    pred_data = defaultdict(lambda: defaultdict(list))  # pred_data[class][image] = [objs]
    for pred_file in os.listdir(args.preds_root):
        if not pred_file.endswith('.txt'):
            continue
        base = os.path.splitext(pred_file)[0]
        if base not in image_sizes:
            continue
        pred_path = os.path.join(args.preds_root, pred_file)
        objs = parse_objects(pred_path, image_sizes[base])
        for obj in objs:
            pred_data[obj['class']][base].append(obj)
    
    # Evaluate
    results = evaluate_dataset(gt_data, pred_data)
    
    # Calculate final metrics
    AP50 = []
    AP = []
    for cls_id in results.values():
        AP50.append(cls_id['AP50'])
        AP.append(cls_id['AP'])
    
    mAP50 = np.mean(AP50) if AP50 else 0.0
    mAP = np.mean(AP) if AP else 0.0
    
    print(f"mAP@50: {mAP50:.4f}")
    print(f"mAP@[50:95]: {mAP:.4f}")

if __name__ == '__main__':
    main()