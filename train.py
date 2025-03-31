from ultralytics import YOLO, YOLOAdv
# from ultralytics.models.yolo.detect.train import DetectionAdvTrainer


def convert_state_dict_lsk(state_dict:dict) -> dict:
    converted_state_dict = {}
    for key, value in state_dict.items():
        key_split = key.split(".")
        layer_num = int(key_split[2])
        if layer_num >= 15 and layer_num <= 17:
            converted_layer_num = layer_num+1
        elif layer_num >= 18 and layer_num <= 20:
            converted_layer_num = layer_num+2
        elif layer_num >= 21:
            converted_layer_num = layer_num+3
        else:
            converted_layer_num = layer_num
        converted_key = ""
        for i in range(2):
            converted_key += key_split[i]
            converted_key += "."
        converted_key += str(layer_num)
        for i in range(3, -1):
            converted_key += "."
            converted_key += key_split[i]
        converted_state_dict[key[:12]+str(converted_layer_num)+key[12+len(key.split(".")[4]):]] = value
    return converted_state_dict


if __name__=="__main__":
    # 加载预训练的yolo12权重
    model = YOLO("yolo12m-lsk.yaml", task="detect")
    check_point_model = YOLO("yolo12m.pt")
    model.load_state_dict(convert_state_dict_lsk(check_point_model.state_dict()), strict=False)

    # 用自定义的SAR数据集微调
    results = model.train(data="datasets/SSDD_yolo/dataset.yaml", epochs=100)

    # 评测模型的效果
    results = model.val()
