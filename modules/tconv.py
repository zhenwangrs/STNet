import math
import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class MultiScale_TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1, 2, 3, 4], ):
        super().__init__()

        # Multiple branches of temporal convolution
        self.num_branches = 4

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels // self.num_branches,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=dilation),
                nn.BatchNorm1d(out_channels // self.num_branches)
            )
            for dilation in dilations
        ])

        # self.fuse = nn.Conv1d(in_channels * self.num_branches, out_channels, kernel_size=1)
        # self.fuse = nn.Conv2d(in_channels, out_channels, kernel_size=(4,1))
        # self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        # out = torch.stack(branch_outs, dim=2)
        # out = self.fuse(out).squeeze(2)
        # out = self.bn(out)
        return out


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1, dropout_ration=0.3):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K5', 'K5', "P2"]
        elif self.conv_type == 4:
            self.kernel_size = ['K5', 'K5']
        elif self.conv_type == 5:
            self.kernel_size = ['K5', "P2", 'K5']
        elif self.conv_type == 6:
            self.kernel_size = ["P2", 'K5', 'K5']
        elif self.conv_type == 7:
            self.kernel_size = ["P2", 'K5', "P2", 'K5']
        elif self.conv_type == 8:
            self.kernel_size = ["P2", "P2", 'K5', 'K5']

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0),
                    # MultiScale_TemporalConv(input_sz, self.hidden_size),
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)
        self.dropout = nn.Dropout(p=dropout_ration)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

        self.positional_encoding = PositionalEncoding(hidden_size)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
                # pass
        return feat_len

    def forward(self, frame_feat, lgt):
        # frame_feat = frame_feat.permute(0, 2, 1)
        # frame_feat = self.positional_encoding(frame_feat)
        # frame_feat = frame_feat.permute(0, 2, 1)

        visual_feat = self.temporal_conv(frame_feat)
        visual_feat = self.dropout(visual_feat)
        lgt = self.update_lgt(lgt)
        if self.num_classes == -1:
            logits = None
        else:
            logits = self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
            # logits = self.dropout(logits)
        # logits = None if self.num_classes == -1 else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }


class SlidingWindowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, window_size, stride):
        super(SlidingWindowLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.window_size = window_size
        self.stride = stride
        self.fc = None
        self.kernel_size = 1

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        feat_len = torch.ceil(torch.div(feat_len - self.window_size, self.stride)) + 1
        return feat_len

    def forward(self, x, lgt):
        # x: b, c, t
        x = x.permute(0, 2, 1)
        # x的维度应为window_size + N * stride, 否则padding
        if (x.size(1) - self.window_size) % self.stride != 0:
            pad_len = self.stride - (x.size(1) - self.window_size) % self.stride
            x = F.pad(x, (0, 0, 0, pad_len), "constant", 0)

        # 滑动窗口处理
        short_sequences = []
        for i in range(0, x.size(1) - self.window_size + 1, self.stride):
            window = x[:, i:i + self.window_size, :]
            # window = x[:, i:, :]
            output, (h_n, _) = self.lstm(window)
            h_n = torch.mean(h_n.permute(1, 0, 2), dim=1)
            # h_n = h_n.permute(1, 0, 2)[:, -1, :]
            short_sequences.append(h_n)
        # 将短序列拼接成输出
        output = torch.stack(short_sequences, dim=1)  # b, t, c

        # 全连接层
        logits = self.fc(output)
        lgt = self.update_lgt(lgt)

        return {
            "visual_feat": output.permute(1, 0, 2),  # t, b, c
            "conv_logits": logits.permute(1, 0, 2),
            "feat_len": lgt.cpu(),
        }
        # return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()


def right_padding(input_tensor, kernel_size):
    batch_size, channels, temp = input_tensor.size()
    # 计算目标输出宽度
    target_width = temp + kernel_size - 1
    # 计算需要的填充量
    padding_size = max(target_width - temp, 0)
    # 使用 F.pad 进行填充，只在右侧进行填充
    padded_tensor = F.pad(input_tensor, (0, padding_size, 0, 0), mode='constant', value=0)
    return padded_tensor


def right_padding_dilation(input_tensor, kernel_size, dilation=1):
    batch_size, channels, temp = input_tensor.size()
    # 计算经过卷积之后的输出宽度
    target_width = temp + (kernel_size - 1) * dilation
    # 计算需要的填充量
    padding_size = max(target_width - temp, 0)
    # 使用 F.pad 进行填充，只在右侧进行填充
    padded_tensor = F.pad(input_tensor, (0, padding_size, 0, 0), mode='constant', value=0)
    return padded_tensor

class PermuteLayer(nn.Module):
    def __init__(self, *dims):
        super(PermuteLayer, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class TemporalPositionWiseMaxPooling(nn.Module):
    def __init__(self):
        super(TemporalPositionWiseMaxPooling, self).__init__()

    def forward(self, x):
        # 获取数据维度
        B, T, C = x.size()
        # 初始化结果
        result = []
        # 对每个位置进行maxpooling
        last_pooled_value = torch.zeros(B, C).cuda()
        for i in range(T):
            # 创建相应位置的kernel大小
            kernel_size = (T - i,)
            # 应用maxpooling
            pooled_value = F.avg_pool1d(x[:, i:].permute(0, 2, 1), kernel_size).squeeze(-1) - last_pooled_value
            # 将结果添加到列表中
            result.append(pooled_value)
            last_pooled_value = torch.mean(torch.stack(result, dim=1), dim=1)
        # 将结果堆叠在一起
        result = torch.stack(result, dim=1)
        return result


class TemporalConvLayer(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_sizes=[2, 4, 6, 8], dilations=[1, 1, 1, 1], avg_kernel=2, mode='cat'):
        super(TemporalConvLayer, self).__init__()
        assert hidden_size % len(kernel_sizes) == 0 or mode == 'add'
        assert mode in ['cat', 'add']

        self.mode = mode
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        conv_output_size = hidden_size // len(kernel_sizes) if self.mode == 'cat' else hidden_size
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    hidden_size,
                    conv_output_size,
                    kernel_size=k,
                    dilation=d,
                    padding=0),
                # nn.GroupNorm(8, hidden_size // len(self.kernel_sizes)),
                # nn.BatchNorm1d(hidden_size // len(self.kernel_sizes)),
                # PermuteLayer(0, 2, 1),
                # nn.LayerNorm(hidden_size),
                # TemporalPositionWiseMaxPooling(),
                # PermuteLayer(0, 2, 1),
                # nn.ReLU(inplace=True),
            )
            for k, d in zip(self.kernel_sizes, self.dilations)
        ])
        self.avgpool = nn.MaxPool1d(kernel_size=avg_kernel, ceil_mode=True)
        self.bn = nn.GroupNorm(1, hidden_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        conv_res = []
        if self.mode == 'cat':
            for i, conv in enumerate(self.convs):
                padded_x = right_padding_dilation(x, self.kernel_sizes[i], dilation=self.dilations[i])
                conv_res.append(conv(padded_x))
            x = torch.cat(conv_res, dim=1)
            x = self.relu(shortcut + self.bn(x))
            x = self.avgpool(x)
            return x
        elif self.mode == 'add':
            for i, conv in enumerate(self.convs):
                padded_x = right_padding_dilation(x, self.kernel_sizes[i], dilation=self.dilations[i])
                x = x + F.relu(conv(padded_x))
            x = self.avgpool(x)
            return x


class MultiScaleConv(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiScaleConv, self).__init__()
        self.kernel_size = 1

        # self.kernel_sizes1 = [3, 5, 7, 9]
        # self.dilation1 = [1, 1, 1, 2]
        self.avg_kernel1 = 2
        # self.kernel_sizes2 = [3, 5, 7, 9]
        # self.dilation2 = [1, 1, 1, 2]
        self.avg_kernel2 = 2

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.linear = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(input_size)
        self.cat_convs = nn.Sequential(
            TemporalConvLayer(hidden_size, hidden_size, kernel_sizes=[2, 4, 6, 8], dilations=[1, 1, 1, 1], avg_kernel=self.avg_kernel1),
            TemporalConvLayer(hidden_size, hidden_size, kernel_sizes=[2, 3, 4, 5], dilations=[1, 1, 1, 1], avg_kernel=self.avg_kernel2),
        )
        self.add_convs = nn.Sequential(
            TemporalConvLayer(hidden_size, hidden_size, kernel_sizes=[2, 5, 8], dilations=[1, 1, 1], avg_kernel=self.avg_kernel1, mode='add'),
            TemporalConvLayer(hidden_size, hidden_size, kernel_sizes=[2, 5, 8], dilations=[1, 1, 1], avg_kernel=self.avg_kernel2, mode='add'),
        )
        # self.fc = nn.Linear(len(kernel_sizes) * hidden_size, output_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        feat_len = torch.ceil(torch.div(feat_len, self.avg_kernel1))
        feat_len = torch.ceil(torch.div(feat_len, self.avg_kernel2))
        return feat_len

    def forward(self, x, lgt):
        # x = x.permute(0, 2, 1)
        # x = self.positional_encoding(x)
        # x = x.permute(0, 2, 1)

        # Conv1d expects input in the format (batch, channels, sequence length)
        xx = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        xx = self.add_convs(xx)

        # 全连接层
        logits = self.fc(xx.transpose(1, 2)).transpose(1, 2)
        lgt = self.update_lgt(lgt)

        return {
            "visual_feat": xx.permute(2, 0, 1),  # t, b, c
            "conv_logits": logits.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }
