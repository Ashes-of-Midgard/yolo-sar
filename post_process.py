import os
import argparse
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

class Sample:
    def __init__(self, image_path, label_path, image_size, boxes):
        self.image_path = image_path
        self.label_path = label_path
        self.image_size = image_size  # (width, height)
        self.boxes = boxes  # List of dicts: {'class_id', 'points', 'center'}
        self.sorted_centers = []
        self.sorted_boxes = []
        self.vector_v = None
        self.vector_u = None

def read_samples(images_root, labels_root):
    samples = []
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for label_file in os.listdir(labels_root):
        if not label_file.endswith('.txt'):
            continue
            
        label_path = os.path.join(labels_root, label_file)
        base_name = os.path.splitext(label_file)[0]
        image_path = None
        for ext in image_exts:
            path = os.path.join(images_root, base_name + ext)
            if os.path.exists(path):
                image_path = path
                break
        if not image_path:
            continue
            
        # Read image size
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Parse label file
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    continue
                class_id = int(parts[0])
                points = list(map(float, parts[1:]))
                # Calculate center
                xs = points[::2]
                ys = points[1::2]
                center = (sum(xs)/4, sum(ys)/4)
                boxes.append({
                    'class_id': class_id,
                    'points': points,
                    'center': center
                })
        
        # Sort boxes by center: top to bottom, left to right
        boxes.sort(key=lambda b: (b['center'][1], b['center'][0]))
        
        sample = Sample(image_path, label_path, (w, h), boxes)
        samples.append(sample)
    
    return samples

def build_feature_vectors(samples):
    max_n = max(len(s.boxes) for s in samples)
    
    for s in samples:
        # Build vectors for centers and boxes
        centers = []
        boxes = []
        for b in s.boxes:
            centers.extend(b['center'])
            boxes.extend(b['points'])
        
        # Pad with zeros
        pad_centers = [0] * (2 * max_n - len(centers))
        pad_boxes = [0] * (8 * max_n - len(boxes))
        
        s.vector_v = np.array(centers + pad_centers)
        s.vector_u = np.array(boxes + pad_boxes)
    
    V = np.stack([s.vector_v for s in samples])
    U = np.stack([s.vector_u for s in samples])
    return V, U, max_n

def cluster_samples(V, eps=0.5, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(V)
    return clustering.labels_

def find_cluster_representatives(samples, labels, V):
    normal_clusters = [l for l in labels if l != -1]
    unique_clusters = set(normal_clusters)
    
    representatives = []
    for cluster_id in unique_clusters:
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_points = V[cluster_indices]
        cluster_center = cluster_points.mean(axis=0)
        
        # Find closest sample to cluster center
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        rep_idx = cluster_indices[np.argmin(distances)]
        representatives.append(samples[rep_idx])
    
    return representatives

def match_boxes(a_centers, n_centers, t_d):
    a_points = [(a_centers[2*i], a_centers[2*i+1]) for i in range(len(a_centers)//2)]
    n_points = [(n_centers[2*i], n_centers[2*i+1]) for i in range(len(n_centers)//2)]
    
    matches = 0
    used_n = set()
    for i, anomaly_p in enumerate(a_points):
        if anomaly_p == (0, 0):
            continue
        min_dist = float('inf')
        min_j = -1
        for j, normal_p in enumerate(n_points):
            if normal_p == (0, 0) or j in used_n:
                continue
            dist = np.sqrt((anomaly_p[0]-normal_p[0])**2 + (anomaly_p[1]-normal_p[1])**2)
            if dist < t_d and dist < min_dist:
                min_dist = dist
                min_j = j
        if min_j != -1:
            matches += 1
            used_n.add(min_j)
    return matches

def process_anomalies(anomalies, representatives, t_d, max_n, suppress_fp=False):
    for a_sample in tqdm(anomalies, desc="Processing anomalies"):
        best_match = None
        best_score = -1
        
        # Find best matching representative
        for rep in representatives:
            score = match_boxes(a_sample.vector_v, rep.vector_v, t_d)
            if score > best_score:
                best_score = score
                best_match = rep
        
        if not best_match:
            continue
            
        # Match individual boxes
        a_centers = a_sample.vector_v.tolist()
        n_centers = best_match.vector_v.tolist()
        
        # Find matched boxes
        matched_a = set()
        matched_n = set()
        for i in range(max_n):
            a_center = (a_centers[2*i], a_centers[2*i+1])
            if a_center == (0, 0):
                continue
            min_dist = float('inf')
            min_j = -1
            for j in range(max_n):
                n_center = (n_centers[2*j], n_centers[2*j+1])
                if n_center == (0, 0) or j in matched_n:
                    continue
                dist = np.sqrt((a_center[0]-n_center[0])**2 + (a_center[1]-n_center[1])**2)
                if dist < t_d and dist < min_dist:
                    min_dist = dist
                    min_j = j
            if min_j != -1:
                matched_a.add(i)
                matched_n.add(min_j)
        
        # Suppress false positives
        if suppress_fp:
            new_boxes = []
            for i, box in enumerate(a_sample.boxes):
                if i in matched_a:
                    new_boxes.append(box)
            a_sample.boxes = new_boxes
        
        # Complete missing boxes
        for j in range(max_n):
            if (n_centers[2*j], n_centers[2*j+1]) == (0, 0):
                continue
            if j not in matched_n:
                # Find corresponding box in representative
                rep_box = best_match.boxes[j]
                
                # Generate new box
                new_box = complete_box(a_sample, rep_box, best_match)
                if new_box:
                    a_sample.boxes.append(new_box)
        
        # Re-sort boxes after modifications
        a_sample.boxes.sort(key=lambda b: (b['center'][1], b['center'][0]))

def complete_box(a_sample, rep_box, rep_sample):
    # Get region X in anomaly image
    w_a, h_a = a_sample.image_size
    cx, cy = rep_box['center']
    
    # Convert normalized coordinates to pixel coordinates
    x_center = cx * w_a
    y_center = cy * h_a
    region_width = w_a / 8
    region_height = h_a / 8
    
    x0 = max(0, int(x_center - region_width/2))
    y0 = max(0, int(y_center - region_height/2))
    x1 = min(w_a, int(x_center + region_width/2))
    y1 = min(h_a, int(y_center + region_height/2))
    
    if x0 >= x1 or y0 >= y1:
        return None
    
    # Read anomaly image
    img_a = cv2.imread(a_sample.image_path)
    if img_a is None:
        return None
    region_X = img_a[y0:y1, x0:x1]
    X_tensor = torch.from_numpy(region_X).permute(2,0,1).unsqueeze(0).float()
    
    # Get kernel K from representative image
    w_rep, h_rep = rep_sample.image_size
    points = np.array(rep_box['points']).reshape(4, 2)
    points[:, 0] *= w_rep
    points[:, 1] *= h_rep
    points = points.astype(int)
    
    x_min = max(0, points[:, 0].min())
    y_min = max(0, points[:, 1].min())
    x_max = min(w_rep, points[:, 0].max())
    y_max = min(h_rep, points[:, 1].max())
    
    img_rep = cv2.imread(rep_sample.image_path)
    if img_rep is None:
        return None
    region_K = img_rep[y_min:y_max, x_min:x_max]
    if region_K.size == 0:
        return None
    
    # Resize kernel to match X region size
    try:
        region_K = cv2.resize(region_K, (x1-x0, y1-y0))
    except:
        return None
    K_tensor = torch.from_numpy(region_K).permute(2,0,1).unsqueeze(0).float()
    
    # Normalize tensors
    X_tensor = X_tensor / 255.0
    K_tensor = K_tensor / 255.0
    
    # Perform convolution
    with torch.no_grad():
        conv_map = F.conv2d(X_tensor, K_tensor, padding=(K_tensor.shape[2]//2, K_tensor.shape[3]//2))
    
    # Find max response
    max_val, max_idx = torch.max(conv_map.view(-1), dim=0)
    max_y = max_idx // conv_map.shape[3]
    max_x = max_idx % conv_map.shape[3]
    
    # Convert to image coordinates
    new_cx = x0 + max_x.item()
    new_cy = y0 + max_y.item()
    
    # Convert to normalized coordinates
    new_cx_norm = new_cx / w_a
    new_cy_norm = new_cy / h_a
    
    # Create new box using relative positions
    dxs = [rep_box['points'][i] - rep_box['center'][0] if i%2==0 else 
           rep_box['points'][i] - rep_box['center'][1] for i in range(8)]
    
    new_points = []
    for i in range(8):
        if i % 2 == 0:
            new_points.append(new_cx_norm + dxs[i])
        else:
            new_points.append(new_cy_norm + dxs[i])
    
    return {
        'class_id': rep_box['class_id'],
        'points': new_points,
        'center': (new_cx_norm, new_cy_norm)
    }

def save_results(samples, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for sample in samples:
        output_path = os.path.join(output_dir, os.path.basename(sample.label_path))
        with open(output_path, 'w') as f:
            for box in sample.boxes:
                line = [str(box['class_id'])] + [f"{x:.6f}" for x in box['points']]
                f.write(" ".join(line) + "\n")

def main(args):
    # Step 1: Read samples
    samples = read_samples(args.images_root, args.labels_root)
    if not samples:
        print("No valid samples found")
        return
    
    # Step 2: Build feature vectors
    V, U, max_n = build_feature_vectors(samples)
    
    # Step 3: Cluster
    labels = cluster_samples(V, eps=args.eps, min_samples=args.min_samples)
    
    # Split normal and anomalies
    normal_samples = [s for s, l in zip(samples, labels) if l != -1]
    anomalies = [s for s, l in zip(samples, labels) if l == -1]
    
    # Step 4: Find representatives
    representatives = find_cluster_representatives(samples, labels, V)
    
    # Step 5-6: Process anomalies
    process_anomalies(anomalies, representatives, args.t_d, max_n, args.suppress_fp)
    
    # Step 7: Save results
    save_results(samples, args.processed_labels_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_root', type=str, required=True, help='Path to images directory')
    parser.add_argument('--labels_root', type=str, required=True, help='Path to labels directory')
    parser.add_argument('--processed_labels_root', type=str, required=True, help='Output directory for processed labels')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN epsilon parameter')
    parser.add_argument('--min_samples', type=int, default=5, help='DBSCAN min samples parameter')
    parser.add_argument('--t_d', type=float, default=0.1, help='Matching distance threshold')
    parser.add_argument('--suppress_fp', action='store_true', help='Enable false positive suppression')
    
    args = parser.parse_args()
    
    main(args)