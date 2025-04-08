from ultralytics import YOLO, YOLOAdv
from convert import *


if __name__=="__main__":
    # 从配置文件读取模型结构
    model = YOLO("yolo12s.yaml", task="detect")
    # model = YOLOAdv("yolo12n.yaml", task="detect") # 开启对抗训练
    
    # 加载预训练模型（可选）
    # !!!不要在初始化时直接读取预训练模型!!!
    # 这里的YOLO模型只是一个包装器，真正的模型是内部的成员属性model
    # 如果直接读取预训练模型，就会覆盖掉内部属性model的类型，导致自定义的结构不生效
    model_pretrained = YOLO("yolo12s.pt")
    model.load_state_dict(model_pretrained.state_dict(), strict=False)

    # 用自定义的SAR数据集训练/微调
    results = model.train(data="datasets/SSDD_yolo/dataset.yaml", epochs=100, amp=False, batch=1) # 不建议开启混合精度，非常容易出现数据溢出

    # 评测模型的效果
    results = model.val()

    # 这是目前测试结果比较好的配置，正式训练的时候启用下面这段
    # # 从配置文件读取模型结构
    # model = YOLO("yolo12x-wtconv.yaml", task="detect")
    # # model = YOLOAdv("yolo12n.yaml", task="detect") # 开启对抗训练
    
    # # 加载预训练模型（可选）
    # # !!!不要在初始化时直接读取预训练模型!!!
    # # 这里的YOLO模型只是一个包装器，真正的模型是内部的成员属性model
    # # 如果直接读取预训练模型，就会覆盖掉内部属性model的类型，导致自定义的结构不生效
    # model_pretrained = YOLO("yolo12x.pt")
    # model.load_state_dict(convert_state_dict_wtconv(model_pretrained.state_dict()), strict=False)

    # # 用自定义的SAR数据集训练/微调
    # results = model.train(data="datasets/SARdet_100K_yolo/dataset.yaml", epochs=300, amp=False) # 不建议开启混合精度，非常容易出现数据溢出

    # # 评测模型的效果
    # results = model.val()