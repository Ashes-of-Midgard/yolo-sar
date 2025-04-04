""" 用于将预训练权重的state_dict转化成自定义模型能读取的形式
    每一个函数对应一个自定义模型结构
"""


def convert_state_dict_lsk(state_dict:dict) -> dict:
    """ 将预训练yolo12权重转化为配置文件yolo12-lsk中模型能够读取的形式
        修改部分权重的层数字段, 以适应新插入的模块
    """
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
        converted_key += str(converted_layer_num)
        for i in range(3, len(key_split)):
            converted_key += "."
            converted_key += key_split[i]
        converted_state_dict[converted_key] = value
    return converted_state_dict


def convert_state_dict_afpn(state_dict:dict) -> dict:
    """ 将预训练yolo12权重转化为配置文件yolo12-afpn中模型能够读取的形式
        只保留backbone和检测头
    """
    converted_state_dict = {}
    for key, value in state_dict.items():
        key_split = key.split(".")
        layer_num = int(key_split[2])
        if layer_num < 8:
            converted_state_dict[key] = value
        if layer_num == 21: # 检测头的层数发生了变化
            converted_state_dict[key[:12]+str(12)+key[12+len(key.split(".")[4]):]] = value
    return converted_state_dict


def convert_state_dict_wtconv(state_dict:dict) -> dict:
    """ 将预训练yolo12权重转化为配置文件yolo12-lsk中模型能够读取的形式
        修改部分权重的层数字段, 以适应新插入的模块
    """
    converted_state_dict = {}
    for key, value in state_dict.items():
        key_split = key.split(".")
        layer_num = int(key_split[2])
        if layer_num >= 2 and layer_num <= 4:
            converted_layer_num = layer_num+1
        elif layer_num >= 4:
            converted_layer_num = layer_num+2
        else:
            converted_layer_num = layer_num
        converted_key = ""
        for i in range(2):
            converted_key += key_split[i]
            converted_key += "."
        converted_key += str(converted_layer_num)
        for i in range(3, len(key_split)):
            converted_key += "."
            converted_key += key_split[i]
        converted_state_dict[converted_key] = value
    return converted_state_dict


def convert_state_dict_obb(state_dict:dict) -> dict:
    """ 将预训练yolo12权重转化为旋转检测模型结构(yolo12没有官方的旋转检测预训练权重)
        删除最后检测头的权重
    """
    converted_state_dict = {}
    for key, value in state_dict.items():
        key_split = key.split(".")
        layer_num = int(key_split[2])
        if layer_num < 21:
            converted_state_dict[key] = value
    return converted_state_dict