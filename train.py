from ultralytics import YOLO

if __name__=="__main__":
    # 加载预训练的yolo12权重
    model = YOLO("yolo12m.pt")

    # 用自定义的SAR数据集微调
    results = model.train(data="datasets/OGSOD_yolo/dataset.yaml", epochs=100)

    # 评测模型的效果
    results = model.val()

