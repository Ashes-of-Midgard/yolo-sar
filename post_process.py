import os
import argparse
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from PIL import Image
import torch
import torch.nn.functional as F
from collections import defaultdict
import json
from shapely.geometry import Polygon

class DetectionPostProcessor:
    def __init__(self, args):
        self.args = args
        self.image_files = []
        self.label_files = []
        self.image_shapes = {}
        self.max_objects = 0
        self.V = None
        self.U = None
        self.cluster_info = defaultdict(list)
        self.abnormal_operations = {}

    def load_data(self):
        # 建立图像-标签文件映射
        file_pairs = []
        for fname in os.listdir(self.args.labels_root):
            if fname.endswith('.txt'):
                label_path = os.path.join(self.args.labels_root, fname)
                base = os.path.splitext(fname)[0]
                img_found = False
                for ext in ['.jpg', '.png', '.jpeg']:
                    img_path = os.path.join(self.args.images_root, base + ext)
                    if os.path.exists(img_path):
                        file_pairs.append((img_path, label_path))
                        img_found = True
                        break
                if not img_found:
                    print(f"Warning: No image found for {label_path}")

        # 第一次遍历获取最大目标数和图像尺寸
        all_centers = []
        all_boxes = []
        for img_path, label_path in file_pairs:
            # 获取图像尺寸
            with Image.open(img_path) as img:
                self.image_shapes[label_path] = img.size  # (width, height)
            
            # 读取标签文件
            with open(label_path) as f:
                lines = f.readlines()
            
            objects = []
            for line in lines:
                parts = list(map(float, line.strip().split()))
                if len(parts) != 10:
                    continue
                cls_id = int(parts[0])
                points = np.array(parts[1:9]).reshape(4, 2)
                conf = parts[9]
                
                # 计算中心点
                center = points.mean(axis=0)
                objects.append((center, points.ravel(), conf))
            
            # 排序：从上到下，从左到右
            objects.sort(key=lambda x: (x[0][1], x[0][0]))
            all_centers.append([c for c, _, _ in objects])
            all_boxes.append([b for _, b, _ in objects])
            self.max_objects = max(self.max_objects, len(objects))

        # 构建V和U矩阵
        self.V = np.zeros((len(file_pairs), 2 * self.max_objects))
        self.U = np.zeros((len(file_pairs), 8 * self.max_objects))
        for i, (centers, boxes) in enumerate(zip(all_centers, all_boxes)):
            flat_centers = np.array(centers).flatten()
            flat_boxes = np.array(boxes).flatten()
            self.V[i, :len(flat_centers)] = flat_centers
            self.U[i, :len(flat_boxes)] = flat_boxes

        self.image_files = [p[0] for p in file_pairs]
        self.label_files = [p[1] for p in file_pairs]

    def cluster_samples(self):
        # DBSCAN聚类
        clustering = DBSCAN(eps=self.args.eps, min_samples=self.args.min_samples).fit(self.V)
        labels = clustering.labels_
        
        # 记录聚类信息
        clusters = defaultdict(list)
        for idx, lbl in enumerate(labels):
            if lbl == -1:
                self.cluster_info['abnormal'].append(idx)
            else:
                clusters[lbl].append(idx)
        
        # 寻找每个聚类的代表样本
        self.cluster_centers = {}
        for cid, members in clusters.items():
            cluster_data = self.V[members]
            center = cluster_data.mean(axis=0)
            distances = np.linalg.norm(cluster_data - center, axis=1)
            rep_idx = members[np.argmin(distances)]
            self.cluster_centers[cid] = rep_idx
            self.cluster_info[cid] = members

        return labels

    def match_abnormal_samples(self):
        normal_reps = list(self.cluster_centers.values())
        # v_normal = self.V[normal_reps]
        matches = {}
        
        for a_idx in self.cluster_info['abnormal']:
            v_abnormal = self.V[a_idx]
            
            # 计算匹配分数
            scores = []
            for rep_idx in normal_reps:
                n_centers = self.V[rep_idx].reshape(-1, 2)
                a_centers = v_abnormal.reshape(-1, 2)
                
                # 过滤零填充部分
                valid_n = np.where(np.any(n_centers != 0, axis=1))[0]
                valid_a = np.where(np.any(a_centers != 0, axis=1))[0]
                
                # 计算距离矩阵
                dist_matrix = pairwise_distances(
                    a_centers[valid_a], 
                    n_centers[valid_n], 
                    metric='euclidean'
                )
                
                # 贪心匹配
                matched = set()
                match_count = 0
                for a_order in np.argsort(dist_matrix, axis=None):
                    a_idx_local = a_order // dist_matrix.shape[1]
                    n_idx_local = a_order % dist_matrix.shape[1]
                    if dist_matrix[a_idx_local, n_idx_local] > self.args.t_d:
                        break
                    a_real = valid_a[a_idx_local]
                    n_real = valid_n[n_idx_local]
                    if a_real not in matched and n_real not in matched:
                        matched.add(a_real)
                        matched.add(n_real)
                        match_count += 1
                scores.append(match_count)
            
            best_match = normal_reps[np.argmax(scores)]
            matches[a_idx] = best_match
            self.abnormal_operations[a_idx] = {'match': best_match, 'add': [], 'remove': []}
        
        return matches

    def process_abnormal(self, matches):
        processed = {}
        for a_idx, n_rep in matches.items():
            label_path = self.label_files[a_idx]
            img_path = self.image_files[a_idx]
            img_w, img_h = self.image_shapes[label_path]
            
            # 原始检测结果
            with open(label_path) as f:
                orig_objs = [list(map(float, line.strip().split())) for line in f]
            
            # 获取正常代表的检测框
            n_centers = self.V[n_rep].reshape(-1, 2)
            n_boxes = self.U[n_rep].reshape(-1, 8)
            
            # 获取异常样本的检测框
            a_centers = self.V[a_idx].reshape(-1, 2)
            a_boxes = self.U[a_idx].reshape(-1, 8)
            
            # 匹配检测框
            valid_n = np.where(np.any(n_centers != 0, axis=1))[0]
            valid_a = np.where(np.any(a_centers != 0, axis=1))[0]
            dist_matrix = pairwise_distances(a_centers[valid_a], n_centers[valid_n])
            
            matched_a = set()
            matched_n = set()
            for a_order in np.argsort(dist_matrix, axis=None):
                a_idx_local = a_order // dist_matrix.shape[1]
                n_idx_local = a_order % dist_matrix.shape[1]
                if dist_matrix[a_idx_local, n_idx_local] > self.args.t_d:
                    break
                a_real = valid_a[a_idx_local]
                n_real = valid_n[n_idx_local]
                if a_real not in matched_a and n_real not in matched_n:
                    matched_a.add(a_real)
                    matched_n.add(n_real)
            
            # 补充漏检
            missing_n = [i for i in valid_n if i not in matched_n]
            new_objs = [obj for obj in orig_objs]
            
            for nq in missing_n:
                # 生成新检测框
                n_center = n_centers[nq]
                n_box = n_boxes[nq]
                
                # 裁剪区域
                crop_w = img_w // 8
                crop_h = img_h // 8
                x_center = int(n_center[0] * img_w)
                y_center = int(n_center[1] * img_h)
                x1 = max(0, x_center - crop_w//2)
                y1 = max(0, y_center - crop_h//2)
                x2 = min(img_w, x_center + crop_w//2)
                y2 = min(img_h, y_center + crop_h//2)
                
                # 获取模板
                n_img_path = self.image_files[n_rep]
                with Image.open(n_img_path) as n_img:
                    n_box_pixels = (n_box.reshape(4, 2) * [n_img.width, n_img.height]).astype(int)
                    min_x = np.min(n_box_pixels[:, 0])
                    max_x = np.max(n_box_pixels[:, 0])
                    min_y = np.min(n_box_pixels[:, 1])
                    max_y = np.max(n_box_pixels[:, 1])
                    kernel = n_img.crop((min_x, min_y, max_x, max_y)).convert('L')
                    kernel_tensor = torch.tensor(np.array(kernel)/255.0).float()
                
                # 处理目标图像
                with Image.open(img_path) as a_img:
                    region = a_img.crop((x1, y1, x2, y2)).convert('L')
                    region_tensor = torch.tensor(np.array(region)/255.0).float()
                
                # 卷积匹配
                kernel_tensor = kernel_tensor.unsqueeze(0).unsqueeze(0)
                region_tensor = region_tensor.unsqueeze(0).unsqueeze(0)
                similarity = F.conv2d(
                    F.normalize(region_tensor - region_tensor.mean()),
                    F.normalize(kernel_tensor - kernel_tensor.mean()),
                    padding='same'
                )
                
                # 找到最大响应位置
                max_val, max_idx = torch.max(similarity.view(-1), 0)
                max_y, max_x = np.unravel_index(max_idx, similarity.shape[2:])
                
                # 生成新框
                new_center_x = (x1 + max_x * (x2 - x1)/region.width) / img_w
                new_center_y = (y1 + max_y * (y2 - y1)/region.height) / img_h
                new_box = n_box.reshape(4, 2) - n_center + [new_center_x, new_center_y]
                
                # 记录操作
                self.abnormal_operations[a_idx]['add'].append(
                    (new_center_x, new_center_y))
                new_objs.append([
                    int(n_box[0]),
                    *new_box.ravel().tolist(),
                    min(self.args.base_conf + max_val.item(), 1.0)
                ])
            
            # 抑制虚警
            if self.args.suppress:
                new_objs = [obj for idx, obj in enumerate(new_objs) 
                           if idx < len(orig_objs) and idx in matched_a]
                removed = [idx for idx in range(len(orig_objs)) 
                          if idx not in matched_a]
                self.abnormal_operations[a_idx]['remove'].extend(removed)
            
            processed[label_path] = new_objs
        
        return processed

    def save_results(self, processed, normal_indices):
        # 创建输出目录
        os.makedirs(self.args.processed_labels_root, exist_ok=True)
        
        # 保存处理后的标签
        for label_path, objs in processed.items():
            base = os.path.basename(label_path)
            out_path = os.path.join(self.args.processed_labels_root, base)
            with open(out_path, 'w') as f:
                for obj in objs:
                    line = ' '.join(map(str, obj))
                    f.write(line + '\n')
        
        # 保存正常样本
        for idx in normal_indices:
            label_path = self.label_files[idx]
            base = os.path.basename(label_path)
            out_path = os.path.join(self.args.processed_labels_root, base)
            with open(label_path) as f_in, open(out_path, 'w') as f_out:
                f_out.write(f_in.read())
        
        # 保存聚类信息
        # 修改cluster_report的构建部分
        cluster_report = {
            'parameters': {
                'eps': float(self.args.eps),
                'min_samples': int(self.args.min_samples),
                't_d': float(self.args.t_d),
                'base_conf': float(self.args.base_conf),
                'suppress': bool(self.args.suppress)
            },
            'clusters': {},
            'abnormal': []
        }
        
        for cid, members in self.cluster_info.items():
            if cid == 'abnormal':
                continue
            # 转换cluster_id为int
            cluster_id = int(cid) if isinstance(cid, np.integer) else cid
            cluster_report['clusters'][f'cluster_{cluster_id}'] = [
                os.path.basename(self.label_files[int(i)]) for i in members  # 转换索引为int
            ]
            # 对文件名进行排序
            cluster_report['clusters'][f'cluster_{cluster_id}'] = sorted(cluster_report['clusters'][f'cluster_{cluster_id}'])
        
        for a_idx in self.cluster_info['abnormal']:
            label_name = os.path.basename(self.label_files[int(a_idx)])  # 转换索引为int
            match_idx = int(self.abnormal_operations[a_idx]['match'])  # 转换匹配索引为int
            cluster_id = [k for k, v in self.cluster_centers.items() if v == match_idx][0]
            cluster_id = int(cluster_id) if isinstance(cluster_id, np.integer) else cluster_id
            
            operations = []
            for add in self.abnormal_operations[a_idx]['add']:
                # 转换numpy float为Python float
                operations.append(f"add box at ({float(add[0]):.4f}, {float(add[1]):.4f})")
            for remove in self.abnormal_operations[a_idx]['remove']:
                obj = processed[self.label_files[a_idx]][remove]
                center = np.array(obj[1:9]).reshape(4,2).mean(axis=0)
                # 转换numpy float为Python float
                operations.append(f"remove box at ({float(center[0]):.4f}, {float(center[1]):.4f})")
            
            cluster_report['abnormal'].append({
                'label': label_name,
                'match_cluster': int(cluster_id),  # 确保cluster_id是int
                'operations': operations
            })
        
        with open(os.path.join(self.args.processed_labels_root, 'cluster_report.json'), 'w') as f:
            json.dump(cluster_report, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_root', required=True)
    parser.add_argument('--labels_root', required=True)
    parser.add_argument('--processed_labels_root', required=True)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--min_samples', type=int, default=5)
    parser.add_argument('--t_d', type=float, default=0.1)
    parser.add_argument('--base_conf', type=float, default=0.5)
    parser.add_argument('--suppress', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    processor = DetectionPostProcessor(args)
    
    # 步骤1：加载数据
    processor.load_data()
    
    # 步骤2：聚类
    processor.cluster_samples()
    
    # 步骤3/4：匹配异常样本
    matches = processor.match_abnormal_samples()
    
    # 步骤5/6：处理异常样本
    processed = processor.process_abnormal(matches)
    
    # 步骤7/8：保存结果
    normal_indices = [i for i in range(len(processor.label_files)) 
                     if i not in processor.cluster_info['abnormal']]
    processor.save_results(processed, normal_indices)

if __name__ == '__main__':
    main()