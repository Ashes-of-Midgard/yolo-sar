import torch.nn as nn
import torch.nn.functional as F


class AFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, select):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.select= select

        # 统一通道
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])

        # 注意力融合
        self.fusion_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, len(in_channels_list), 1)
        )

        # SE-like通道增强
        self.context_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels, 1),
                nn.Sigmoid()
            )
            for _ in range(3)
        ])

        # 输出卷积
        self.out_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(3)
        ])

    def forward(self, inputs):
        # 检查输入通道
        for feat, expected_c in zip(inputs, self.in_channels_list):
            assert feat.shape[1] == expected_c, f"AFPN: expected {expected_c} channels, got {feat.shape[1]}"

        # 通道统一
        feats = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        size = feats[0].shape[2:]
        feats = [F.interpolate(f, size=size, mode='nearest') if f.shape[2:] != size else f for f in feats]

        # 融合
        fused = sum(feats)
        weights = F.softmax(self.fusion_attn(fused), dim=1)
        fused = sum(w * f for w, f in zip(weights.chunk(len(feats), dim=1), feats))

        # 多尺度输出
        outs = []
        for i in range(3):
            context = self.context_convs[i](fused)
            x = fused * context
            outs.append(self.out_convs[i](x))
            if i < 2:
                fused = F.max_pool2d(x, 2)  # 下采样

        # 返回解包的输出（不是 tuple/list）
        p3, p4, p5 = outs
        if self.select == 3:
            return p3
        elif self.select == 4:
            return p4
        elif self.select == 5:
            return p5
