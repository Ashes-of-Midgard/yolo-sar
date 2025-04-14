import os
from tqdm import tqdm
import argparse

from ultralytics import YOLO, YOLOAdv
from ultralytics.models.yolo.obb import OBBAdvTrainer
from convert import *


if __name__=="__main__":
    # 图像的目录和保存结果的目录
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--images_root', type=str, required=True)
    parser.add_argument('--labels_root', type=str, required=True)
    args = parser.parse_args()

    # 读取训练后权重
    model = YOLO(args.model, task="obb")

    # 遍历预测每一张图像
    for file_name in tqdm(os.listdir(args.images_root)):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            result = model(os.path.join(args.images_root, file_name), verbose=False)[0]
            result.save_txt(os.path.join(args.labels_root, file_name[:-3]+"txt"), save_conf=True)
    
