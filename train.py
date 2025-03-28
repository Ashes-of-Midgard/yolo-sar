from ultralytics import YOLO, YOLOAdv, YOLOAdvDn
# from ultralytics.models.yolo.detect.train import DetectionAdvTrainer

if __name__=="__main__":
    # 加载预训练的yolo12权重
    model = YOLOAdvDn("yolo12m.yaml", task="detect")
    check_point_model = YOLO("yolo12m.pt")
    model.load_state_dict(check_point_model.state_dict(), strict=False)

    # 用自定义的SAR数据集微调
    results = model.train(data="datasets/OGSOD_yolo/dataset.yaml", epochs=100)

    # 评测模型的效果
    results = model.val()

