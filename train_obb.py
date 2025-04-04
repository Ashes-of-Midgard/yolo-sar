from ultralytics import YOLO, YOLOAdv
from convert import *


if __name__=="__main__":
    # 从配置文件读取模型结构
    model = YOLO("yolo12n-obb.yaml", task="obb")
    # model = YOLOAdv("yolo12n-obb.yaml", task="obb") # 开启对抗训练
    
    # 加载预训练模型（可选）
    # !!!不要在初始化时直接读取预训练模型!!!
    # 这里的YOLO模型只是一个包装器，真正的模型是内部的成员属性model
    # 如果直接读取预训练模型，就会覆盖掉内部属性model的类型，导致自定义的结构不生效
    model_pretrained = YOLO("yolo12n.pt")
    model.load_state_dict(convert_state_dict_obb(model_pretrained.state_dict()), strict=False)

    # 用自定义的SAR数据集训练/微调
    results = model.train(data="datasets/SRSDD-V1.0-yolo/dataset.yaml", epochs=15, amp=False) # 不建议开启混合精度，非常容易出现数据溢出

    # 评测模型的效果
    results = model.val()
