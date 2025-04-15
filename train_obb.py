from ultralytics import YOLO, YOLOAdv
from ultralytics.models.yolo.obb import OBBAdvTrainer
from convert import *


if __name__=="__main__":
    # 从配置文件读取模型结构
    # model = YOLO("yolo12m-obb.yaml", task="obb")
    # model = YOLO("yolo12x-obb-wtconv.yaml", task="obb")
    # model = YOLO("yolo12m-obb-wtconv-2.yaml", task="obb")
    # model = YOLO("yolo12m-obb-deno.yaml", task="obb")
    # model = YOLO("yolo12m-obb-deno-2.yaml", task="obb")
    # model = YOLO("yolo12m-obb-deno-3.yaml", task="obb")
    # model = YOLO("yolo12m-obb-sc.yaml", task="obb")
    model = YOLO("/root/yolo-sar/runs/obb/train3/weights/best.pt", task="obb")
    
    # 加载预训练模型（可选）
    # !!!不要在初始化时直接读取预训练模型!!!
    # 这里的YOLO模型只是一个包装器，真正的模型是内部的成员属性model
    # 如果直接读取预训练模型，就会覆盖掉内部属性model的类型，导致自定义的结构不生效
    # model_pretrained = YOLO("yolo12m.pt")
    # model.load_state_dict(convert_state_dict_obb_wtconv(model_pretrained.state_dict()), strict=False)

    # 用自定义的SAR数据集训练/微调
    results = model.train(data="/root/yolo-sar/datasets/SRSDD-competition-aug-2/dataset.yaml", epochs=100, amp=True, batch=16, device=[0,1,2,3]) # 不建议开启混合精度，非常容易出现数据溢出

    # 评测模型的效果
    results = model.val(data="/root/yolo-sar/datasets/SRSDD-competition/dataset.yaml")
