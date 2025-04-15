import os
import argparse
import numpy as np
from glob import glob
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN
import torch
import torch.nn.functional as F
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_root', required=True)
    parser.add_argument('--labels_root', required=True)
    parser.add_argument('--processed_root', required=True)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--min_samples', type=int, default=5)
    parser.add_argument('--t_d', type=float, default=0.2)
    parser.add_argument('--base_conf', type=float, default=0.5)
    parser.add_argument('--suppress', action='store_true')
    parser.add_argument('--vis', action='store_true')
    return parser.parse_args()

class DetectionResult:
    def __init__(self, label_path, image_path):
        self.label_path = label_path
        self.image_path = image_path
        self.boxes = []
        self.centers = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 10: continue
                class_id, points, conf = int(parts[0]), list(map(float, parts[1:9])), float(parts[9])
                self.boxes.append((class_id, points, conf))
                xs, ys = points[::2], points[1::2]
                self.centers.append((sum(xs)/4, sum(ys)/4))

# 修改后的load_results函数
def load_results(images_root, labels_root):
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    results = []
    label_files = glob(os.path.join(labels_root, '*.txt'))
    
    # 添加加载进度条
    for lf in tqdm(label_files, desc='Loading results'):
        base = os.path.splitext(os.path.basename(lf))[0]
        for ext in image_exts:
            ip = os.path.join(images_root, base + ext)
            if os.path.exists(ip):
                results.append(DetectionResult(lf, ip))
                break
    return results

def compute_distance(P1, P2):
    """改进的距离计算函数"""
    if not P1 or not P2: 
        return float('inf')
    
    # 转换为PyTorch张量
    t1 = torch.tensor(P1, dtype=torch.float32)
    t2 = torch.tensor(P2, dtype=torch.float32)
    
    # 计算距离矩阵
    dist_matrix = torch.cdist(t1, t2)
    
    # 确保矩阵可处理
    m, n = dist_matrix.shape
    if m == 0 or n == 0:
        return float('inf')
    
    # 贪心匹配算法（改进版）
    total_cost = 0.0
    matched_pairs = 0
    for _ in range(min(m, n)):
        # 找到全局最小值
        min_val = torch.min(dist_matrix)
        if min_val > 1e6:  # 处理inf情况
            break
        
        # 找到所有最小值位置
        positions = torch.nonzero(dist_matrix == min_val)
        if positions.size(0) == 0:
            break
            
        # 选择第一个匹配
        i, j = positions[0].tolist()
        total_cost += min_val.item()
        matched_pairs += 1
        
        # 屏蔽已选行列
        dist_matrix[i] = float('inf')
        dist_matrix[:, j] = float('inf')
    
    return total_cost / matched_pairs if matched_pairs > 0 else float('inf')

# 修改后的cluster_results函数
def cluster_results(results, eps, min_samples):
    n = len(results)
    distance_matrix = np.zeros((n, n))
    
    # 添加距离计算进度条
    for i in tqdm(range(n), desc='Computing distances'):
        for j in range(i+1, n):
            dist = compute_distance(results[i].centers, results[j].centers)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    np.fill_diagonal(distance_matrix, 0)
    
    return DBSCAN(eps=eps, min_samples=min_samples, 
                metric='precomputed').fit_predict(distance_matrix)

def find_cluster_representatives(results, labels):
    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1: continue
        clusters.setdefault(label, []).append(results[idx])
    
    representatives = {}
    for cid, members in clusters.items():
        # 计算成员间的平均距离
        avg_dists = []
        for m in members:
            total_dist = sum(compute_distance(m.centers, o.centers) for o in members)
            avg_dists.append((total_dist, m))
        # 选择平均距离最小的作为代表
        avg_dists.sort()
        representatives[cid] = avg_dists[0][1]
    return representatives

def match_and_prune(query_centers, rep_centers, t_d):
    """改进的匹配剪枝算法"""
    if not query_centers or not rep_centers:
        return [], []
    
    # 创建距离矩阵
    dist_matrix = torch.cdist(
        torch.tensor(query_centers),
        torch.tensor(rep_centers)
    ).numpy()
    
    # 贪心匹配
    matched_pairs = []
    rows, cols = dist_matrix.shape
    mask = np.ones_like(dist_matrix, dtype=bool)
    
    while True:
        # 找到当前最小距离
        min_val = np.min(dist_matrix[mask])
        if min_val > t_d or np.isinf(min_val):
            break
            
        # 找到所有最小距离的位置
        positions = np.where((dist_matrix == min_val) & mask)
        if len(positions[0]) == 0:
            break
            
        # 选择第一个匹配
        i, j = positions[0][0], positions[1][0]
        matched_pairs.append((i, j))
        mask[i] = False  # 屏蔽行
        mask[:,j] = False  # 屏蔽列
    
    # 确定匹配和未匹配项
    matched_q = set(p[0] for p in matched_pairs)
    matched_r = set(p[1] for p in matched_pairs)
    return (
        [j for j in range(len(rep_centers)) if j not in matched_r],
        [i for i in range(len(query_centers)) if i not in matched_q]
    )

def crop_region(image, center, size_ratio=1/8):
    w, h = image.size
    crop_w = int(w * size_ratio)
    crop_h = int(h * size_ratio)
    x = int(center[0] * w - crop_w/2)
    y = int(center[1] * h - crop_h/2)
    # Ensure within bounds
    x = max(0, min(x, w - crop_w))
    y = max(0, min(y, h - crop_h))
    return image.crop( (x, y, x+crop_w, y+crop_h) )

def process_anomaly(anomaly_result, rep_result, t_d, base_conf, suppress):
    # Match centers
    missing, fp = match_and_prune(anomaly_result.centers, rep_result.centers, t_d)
    # Process missing (add boxes)
    added_boxes = []
    for idx in missing:
        rep_box = rep_result.boxes[idx]
        # Get center from rep
        rep_center = rep_result.centers[idx]
        # Open images
        try:
            anomaly_img = Image.open(anomaly_result.image_path).convert('RGB')
            rep_img = Image.open(rep_result.image_path).convert('RGB')
        except:
            continue
        # Crop X region (anomaly image around rep_center)
        X = crop_region(anomaly_img, rep_center)
        # Crop K region (rep's box)
        # Get rep box points in pixel coords
        w, h = rep_img.size
        points = [ (rep_box[1][i]*w, rep_box[1][i+1]*h) for i in range(0,8,2) ]
        # Find bounding box of the rotated rectangle
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        K_region = rep_img.crop( (x_min, y_min, x_max, y_max) )
        # Resize K to X's size for convolution
        X = X.resize( (64, 64) )  # 示例调整尺寸，可能需要优化
        K = K_region.resize( (64, 64) )
        # Convert to tensors
        X_tensor = torch.from_numpy(np.array(X)).permute(2,0,1).float()
        K_tensor = torch.from_numpy(np.array(K)).permute(2,0,1).float()
        # Normalize
        X_tensor = F.normalize(X_tensor.unsqueeze(0), dim=1)
        K_tensor = F.normalize(K_tensor.unsqueeze(0), dim=1)
        # Convolution
        similarity = F.conv2d(X_tensor, K_tensor, padding=64//2)
        max_val, max_idx = torch.max(similarity, dim=-1)
        max_val = max_val.item()
        # Generate new box
        # 此处简化处理，实际需根据卷积响应确定位置
        new_center = rep_center  # 示例，实际需根据最大响应计算
        new_conf = min(base_conf + max_val, 1.0)
        # 创建新检测框（示例，需根据实际坐标转换）
        new_box = (rep_box[0], rep_box[1], new_conf)
        added_boxes.append(new_box)
    # Process false positives (suppress)
    new_boxes = [b for i,b in enumerate(anomaly_result.boxes) if i not in fp]
    # Add new boxes
    new_boxes += added_boxes
    return new_boxes

def save_processed(result, new_boxes, processed_root):
    rel_path = os.path.relpath(result.label_path, args.labels_root)
    save_path = os.path.join(processed_root, rel_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        for b in new_boxes:
            line = ' '.join(map(str, [b[0]] + b[1] + [b[2]]))
            f.write(line + '\n')

def visualize(result, cluster_id, output_dir, is_anomaly):
    img = Image.open(result.image_path)
    draw = ImageDraw.Draw(img)
    for box in result.boxes:
        points = box[1]
        # Convert normalized to pixel
        w, h = img.size
        pixel_points = [ (points[i]*w, points[i+1]*h) for i in range(0,8,2) ]
        draw.polygon(pixel_points, outline='red')
    save_dir = os.path.join(output_dir, f'cluster_{cluster_id}')
    os.makedirs(save_dir, exist_ok=True)
    suffix = '-anomaly' if is_anomaly else ''
    base = os.path.basename(result.image_path)
    save_path = os.path.join(save_dir, f'{base}{suffix}.jpg')
    img.save(save_path)

# 修改后的main函数处理循环
def main(args):
    results = load_results(args.images_root, args.labels_root)
    labels = cluster_results(results, args.eps, args.min_samples)
    representatives = find_cluster_representatives(results, labels)
    
    # 添加处理进度条
    for idx, label in enumerate(tqdm(labels, desc='Processing results')):  # 修改这里
        result = results[idx]
        if label != -1:
            save_processed(result, result.boxes, args.processed_root)
            if args.vis:
                visualize(result, label, os.path.join(args.processed_root, 'vis'), False)
            continue
        
        min_dist = float('inf')
        nearest_rep = None
        for cid, rep in representatives.items():
            dist = compute_distance(result.centers, rep.centers)
            if dist < min_dist:
                min_dist = dist
                nearest_rep = rep
        
        if not nearest_rep:
            continue
        
        new_boxes = process_anomaly(result, nearest_rep, args.t_d, args.base_conf, args.suppress)
        save_processed(result, new_boxes, args.processed_root)
        if args.vis:
            visualize(result, cid, os.path.join(args.processed_root, 'vis'), True)

if __name__ == '__main__':
    args = parse_args()
    main(args)