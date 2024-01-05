import pdb
import copy

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from torchaudio.models import emformer_rnnt_base, emformer_rnnt_model

import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet_sk as resnet
from modules.gnn import GNN
from modules.randomwalk import CRW
from modules.rnnt import RNNTModel
from modules.tconv import SlidingWindowLSTM, MultiScaleConv
from torchaudio.functional import rnnt_loss


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class FramewiseLoss(nn.Module):
    def __init__(self):
        super(FramewiseLoss, self).__init__()
        self.loss = nn.CosineEmbeddingLoss()

    def forward(self, seq):
        b, c, t = seq.shape
        seq = seq.permute(0, 2, 1) # b, t, c
        seq = seq.reshape(-1, c)
        # seq1 = seq[:, :-1, :]
        # seq2 = seq[:, 1:, :]
        # seq1 = seq1.reshape(-1, c)
        # seq2 = seq2.reshape(-1, c)
        # 从seq随机取200个样本
        idx = torch.randperm(seq.shape[0])[:200]
        seq1 = seq[idx]
        idx = torch.randperm(seq.shape[0])[:200]
        seq2 = seq[idx]

        # self.draw_tsne(seq.detach().cpu().numpy(), color='#5CB9F7')
        # self.draw_tsne(seq1[1].detach().cpu().numpy())
        # plt.show()
        # plt.savefig('./paper/tsne.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        # plt.close()

        # seq1 = F.normalize(seq1, p=2, dim=1)
        # seq2 = F.normalize(seq2, p=2, dim=1)
        y = torch.ones(seq1.shape[0]).to(seq.device) * -1
        loss = self.loss(seq1, seq2, y)
        # print('Framewise Loss: ', loss.item())
        return loss

    def draw_tsne(self, X, color='b'):
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        # Calculate pairwise distances in high-dimensional space
        distances_high_dim = pairwise_distances(X, metric='euclidean')
        # Calculate pairwise distances in low-dimensional space
        distances_low_dim = pairwise_distances(X_tsne, metric='euclidean')
        # Compute total average distance in high-dimensional space
        avg_distance_high_dim = np.mean(distances_high_dim)
        # Compute total average distance in low-dimensional space
        avg_distance_low_dim = np.mean(distances_low_dim)

        # Plot the t-SNE results
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color)
        plt.title(f't-SNE Visualization, average distance: {avg_distance_high_dim:.2f}')
        # plt.show()
        # plt.savefig('./paper/tsne.pdf', format='pdf', dpi=1000, bbox_inches='tight')

        # Display total average distances
        print(f'Total Average Distance in High-Dimensional Space: {avg_distance_high_dim}')
        print(f'Total Average Distance in Low-Dimensional Space: {avg_distance_low_dim}')


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True, lstm_layers=2, dropout=0.5,
            gcl_layers=1, num_K=4,
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        # self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()
        #self.sk_loss_fn = SinkhornDistanceLoss()

        gnn_dim = 128
        self.gnn = GNN(gnn_dim, gnn_dim, gnn_dim, 2, 4, gcl_layers, num_K)
        # self.mcf = MultiScaleCrossAttentionModule(512, 512)
        # self.randomwalk = CRW(())

        self.framewise_loss = FramewiseLoss()
        # self.conv1d = TemporalConv(input_size=512,
        #                            hidden_size=hidden_size,
        #                            conv_type=conv_type,
        #                            use_bn=use_bn,
        #                            num_classes=num_classes)
        self.conv1d = MultiScaleConv(
            input_size=512,
            hidden_size=hidden_size,
            output_size=num_classes
        )
        # self.conv1d = SlidingWindowLSTM(
        #     input_size=512,
        #     hidden_size=hidden_size,
        #     output_size=num_classes,
        #     window_size=12,
        #     stride=4,
        # )
        self.decoder = utils.Decode(gloss_dict, num_classes, 'max')
        self.beam_decoder = utils.Decode(gloss_dict, num_classes, 'fast_beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM',
                                          input_size=hidden_size,
                                          hidden_size=hidden_size,
                                          num_layers=lstm_layers,
                                          bidirectional=True,
                                          dropout=dropout)
        # 判断一个序列有多少个词
        self.csl_num_cls = nn.Linear(hidden_size, 30)
        self.csl_num_loss_fn = nn.CrossEntropyLoss()

        if weight_norm:
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
            self.classifier = NormLinear(hidden_size, self.num_classes)
        else:
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
            self.classifier = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier

        self.rnnt = RNNTModel(input_size=hidden_size, hidden_size=hidden_size // 2, output_size=hidden_size, dropout=dropout)

        # self.register_backward_hook(self.backward_hook)

    # def backward_hook(self, module, grad_input, grad_output):
    #     for g in grad_input:
    #         g[g != g] = 0

    # 注册 forward hook
    def forward_hook(self, module, input, output):
        # 在 forward 过程中记录中间结果
        setattr(module, '_feature_maps', output)

    # 注册 backward hook
    def backward_hook(self, module, grad_input, grad_output):
        # 在 backward 过程中记录梯度
        setattr(module, '_grad_output', grad_output[0])

    # 注册 hook
    def register_hooks(self, model):
        for layer in model.children():
            layer.register_forward_hook(self.forward_hook)
            layer.register_full_backward_hook(self.backward_hook)

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None, is_train=True):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            # inputs = x.reshape(batch * temp, channel, height, width)
            # framewise = self.masked_bn(inputs, len_x)
            # framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)

            framewise = self.conv2d(x.permute(0, 2, 1, 3, 4))

            #sk_loss = self.sk_loss_fn(hidden_x)
            framewise = framewise.view(batch, temp, -1).permute(0, 2, 1)# btc -> bct
            frame_loss = self.framewise_loss(framewise)

            # randomwalk_loss = self.randomwalk(hidden_x)

            # gnn_x, gnn_mean_x, gnn_loss = self.gnn(x, is_train)
            # framewise = torch.cat([framewise, gnn_mean_x], dim=1)

            # framewise, _, _ = self.mcf(framewise, gnn_x)
            #framewise = gnn_x
        else:
            # frame-wise features
            framewise = x
            frame_loss = 0

        gnn_loss = 0
        # sk_loss = 0
        # randomwalk_loss = torch.zeros(1).to(x.device)

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']


        '''
        target_seq_list = label.cpu().int().numpy().tolist()
        target_seq = [target_seq_list[i:i + length] + [0] * (max(label_lgt) - length) for i, length in
                      enumerate(label_lgt)]
        target_seq = torch.LongTensor(target_seq).cuda().int()
        rt_output = self.rnnt(
            sources=conv1d_outputs['visual_feat'].permute(1, 0, 2),
            source_lengths=conv1d_outputs['feat_len'].cuda().int(),
            targets=target_seq,
            target_lengths=label_lgt.cuda().int()
        )
        rl = rnnt_loss(rt_output[0], target_seq, conv1d_outputs['feat_len'].cuda().int(), label_lgt.cuda().int(), reduction='mean', blank=0)
        '''

        tm_outputs = self.temporal_model(x, lgt)
        # tm_outputs = self.rnnt(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])

        # 根据LSTM输出判断一个序列有多少个词
        # hidden = torch.mean(tm_outputs['hidden'].permute(1, 0, 2), dim=1)
        # csl_num_cls = self.csl_num_cls(hidden)
        # csl_loss = self.csl_num_loss_fn(csl_num_cls, label_lgt)
        csl_loss = 0

        #outputs = conv1d_outputs['conv_logits']
        conv_pred = None if self.training else self.decoder.decode(conv1d_outputs['conv_logits'] + outputs, lgt, batch_first=False, probs=False)
        pred = None if self.training else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)

        return {
            # "framewise_features": framewise,
            # "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
            "gnn_loss": gnn_loss * self.loss_weights['GNN'],
            'frame_loss': frame_loss * self.loss_weights['Frame'],
            'csl_loss': csl_loss * self.loss_weights['CSL'],
            # 'sk_loss': sk_loss * self.loss_weights['SK'],
            # 'randomwalk_loss': randomwalk_loss * self.loss_weights['RandomWalk'],
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        gnn_loss = ret_dict["gnn_loss"]
        # sk_loss = ret_dict['sk_loss']
        frame_loss = ret_dict['frame_loss']
        csl_loss = ret_dict['csl_loss']
        conv_ctc_loss = 0
        seq_ctc_loss = 0
        dist_loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                conv_ctc_loss = weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                              label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                              label_lgt.cpu().int()).mean()

            elif k == 'SeqCTC':
                seq_ctc_loss = weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                             label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                             label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                dist_loss = weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                               ret_dict["sequence_logits"].detach(),
                                                               use_blank=False)
        # loss = conv_ctc_loss + seq_ctc_loss + dist_loss + frame_loss #+ csl_loss
        loss = frame_loss #+ csl_loss
        # loss = seq_ctc_loss + frame_loss + csl_loss
        #loss = conv_ctc_loss + gnn_loss
        return loss, conv_ctc_loss, seq_ctc_loss, dist_loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        # self.loss['CTCLoss'] = FocalCTCLoss()
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss


class FocalCTCLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, blank=0):
        super(FocalCTCLoss, self).__init__()
        self.ctc_loss = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc_loss(logits, targets, input_lengths, target_lengths)
        pt = torch.exp(-ctc_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ctc_loss
        return focal_loss
