import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss


def sinkhorn_knopp_nonlinear(a, b, C, epsilon=1e-3, max_iters=100, nonlinear_weight=0.1):
    """
    Sinkhorn-Knopp algorithm with a nonlinear orthogonality condition for batched matrices.

    Parameters:
    - a: First probability matrix (batch_size x n x p) (PyTorch tensor)
    - b: Second probability matrix (batch_size x m x p) (PyTorch tensor)
    - C: Cost matrix (batch_size x n x m) (PyTorch tensor)
    - epsilon: Convergence threshold
    - max_iters: Maximum number of iterations
    - nonlinear_weight: Weight for the nonlinear orthogonality condition

    Returns:
    - P: Optimal transportation matrix (batch_size x n x m) (PyTorch tensor)
    """
    # Normalize the vectors along the second dimension (sum over p)
    a /= torch.sum(a, dim=2, keepdim=True)
    b /= torch.sum(b, dim=2, keepdim=True)

    # Get the dimensions
    batch_size, n, p = a.size()
    _, m, _ = b.size()

    # Initialize the transportation matrix
    P = torch.exp(-nonlinear_weight * C)

    for _ in range(max_iters):
        # Update rows
        P *= (a.unsqueeze(-1) / (torch.sum(P, dim=-1, keepdim=True) + 1e-10))
        # Update columns
        P *= (b.unsqueeze(-2) / (torch.sum(P, dim=-2, keepdim=True) + 1e-10))

        # Check convergence
        err = torch.norm(torch.sum(P, dim=-1) - a) + torch.norm(torch.sum(P, dim=-2) - b)
        if err.max() < epsilon:
            break

    return P


def affinity(x1, x2):
    in_t_dim = x1.ndim
    if in_t_dim < 4:  # add in time dimension if not there
        x1, x2 = x1.unsqueeze(-2), x2.unsqueeze(-2)

    A = torch.einsum('bctn,bctm->btnm', x1, x2)
    # if self.restrict is not None:
    #     A = self.restrict(A)

    return A.squeeze(1) if in_t_dim < 4 else A


def zeroout_diag(A, zero=0):
    mask = (torch.eye(A.shape[-1]).unsqueeze(0).repeat(A.shape[0], 1, 1).bool() < 1).float().cuda()
    return A * mask


def stoch_mat(A, zero_diagonal=False, do_dropout=True, do_sinkhorn=False, edgedrop_rate=0, temperature=0.07):
    ''' Affinity -> Stochastic Matrix '''
    if zero_diagonal:
        A = zeroout_diag(A)

    if do_dropout and edgedrop_rate > 0:
        A[torch.rand_like(A) < edgedrop_rate] = -1e20

    if do_sinkhorn:
        sk = sinkhorn_knopp((A/temperature).exp(), tol=0.01, max_iter=100, verbose=False)
        return sk

    return F.softmax(A/temperature, dim=-1)


def sinkhorn_knopp(A, tol=0.01, max_iter=100, verbose=False):
    _iter = 0

    if A.ndim > 2:
        S = A.sum(-1).sum(-1)
        S = S[:, None, None]
        A = A / A.sum(-1).sum(-1)[:, None, None]
    else:
        A = A / A.sum(-1).sum(-1)[None, None]

    A1 = A2 = A

    while (A2.sum(-2).std() > tol and _iter < max_iter) or _iter == 0:
        A1 = F.normalize(A2, p=1, dim=-2)
        A2 = F.normalize(A1, p=1, dim=-1)

        _iter += 1
        if verbose:
            print(A2.max(), A2.min())
            print('row/col sums', A2.sum(-1).std().item(), A2.sum(-2).std().item())

    if verbose:
        print('------------row/col sums aft', A2.sum(-1).std().item(), A2.sum(-2).std().item())

    return A2


class SKLayer(nn.Module):
    def __init__(self, channels):
        super(SKLayer, self).__init__()
        self.temperature = 0.07
        self.edgedrop_rate = 0.1
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, A):
        b, c, t, h, w = A.shape
        A = A.reshape(b, c, t, h * w)  # b,c,t,h*w
        # with torch.no_grad():
        seq1 = torch.cat([A[:, :, :1], A[:, :, :-1]], dim=2)
        seq2 = torch.cat([A[:, :, 1:], A[:, :, -1:]], dim=2)
        aff = affinity(seq1, seq2)
        # aff = F.normalize(aff, p=2, dim=-1)
        aff = F.sigmoid(aff) - 0.5

        A12s = [stoch_mat(aff[:, i], do_dropout=False) for i in range(t)]
        sk = torch.stack(A12s, dim=1)
        sk = F.sigmoid(sk) - 0.5
        aff_feature = torch.einsum('bcth,bthh->bcth', seq1, sk)

        # aff_feature = torch.einsum('bcth,bthh->bcth', seq1, aff)
        A = A + aff_feature # b,c,t,h*w
        A = A.reshape(b, c, t, h, w)
        A = self.relu(A)
        return A

class SinkhornDistanceLoss(nn.Module):
    def __init__(self):
        super(SinkhornDistanceLoss, self).__init__()
        # self.sk_loss_fn = SamplesLoss(loss='sinkhorn', p=2, blur=.05, scaling=.8, debias=False, potentials=True, reach=None, diameter=None, cost='l2', backend='tensorized', retain_graph=False)
        self.sk_loss_fn = SamplesLoss("sinkhorn", blur=0.1, scaling=0.9, debias=False)

    def forward(self, sequence):
        b, t, c, h, w = sequence.shape
        sequence = sequence.reshape(b, t, c, h * w)
        sequence = sequence.permute(0, 1, 3, 2) # b, t, hw, c
        seq1 = sequence[:, :-1, :, :]
        seq2 = sequence[:, 1:, :, :]

        frame1 = seq1.reshape(b * (t - 1), h * w, c).contiguous()
        frame2 = seq2.reshape(b * (t - 1), h * w, c).contiguous()
        loss = self.sk_loss_fn(frame1, frame2)
        loss = torch.mean(loss)
        return loss


class AdapterLayer(nn.Module):
    def __init__(self, in_channels, adapter_channels=2048):
        super(AdapterLayer, self).__init__()
        # 1x1 Convolution for feature transformation
        self.conv_up = nn.Conv3d(in_channels, adapter_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv_down = nn.Conv3d(adapter_channels, in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = F.relu(self.conv_up(x))
        x = F.relu(self.conv_down(x))
        x = self.bn1(x + res)
        return x


class AffSkLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.adapter = AdapterLayer(channels, channels*4)

        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.up_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

        reduction_channel = channels//4
        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)
        self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 1, 1), groups=reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 2, 2), dilation=(1, 2, 2), groups=reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 3, 3), dilation=(1, 3, 3), groups=reduction_channel)
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=False)
        self.weights2 = nn.Parameter(torch.ones(3) / 3, requires_grad=False)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x

        # x = self.adapter(x)

        x = self.down_conv2(x)
        normed_x = F.normalize(x, p=2, dim=1)
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t, h * w)
        normed_x = normed_x.reshape(b, c, t, h * w)

        seq1 = torch.cat([x[:, :, :1, :], x[:, :, :-1, :]], dim=2)
        seq2 = torch.cat([x[:, :, 1:, :], x[:, :, -1:, :]], dim=2)
        seq3 = x

        normed_seq1 = torch.cat([normed_x[:, :, :1, :], normed_x[:, :, :-1, :]], dim=2)
        normed_seq2 = torch.cat([normed_x[:, :, 1:, :], normed_x[:, :, -1:, :]], dim=2)
        normed_seq3 = normed_x

        features = self.sk(x, normed_x, seq1, normed_seq1) * self.weights2[0] + \
                   self.sk(x, normed_x, seq2, normed_seq2) * self.weights2[1] + \
                   self.sk(x, normed_x, seq3, normed_seq3) * self.weights2[2]
        features = features.reshape(b, c, t, h, w)
        x = x.reshape(b, c, t, h, w)

        xx = self.down_conv(x)
        aggregated_x = self.spatial_aggregation1(xx) * self.weights[0] + self.spatial_aggregation2(xx) * self.weights[1] + self.spatial_aggregation3(xx) * self.weights[2]
        aggregated_x = self.conv_back(aggregated_x)
        # score = aggregated_x > 0
        # score = score.float()
        score = F.relu(aggregated_x)
        # score = F.sigmoid(aggregated_x) - 0.5
        features = features * score

        features = self.up_conv2(features)
        x = self.bn1(shortcut + features)
        x = self.relu(x)
        return x

    def sk(self, x, normed_x, seq, normed_seq):
        b, c, t, h_w = x.shape
        aff = affinity(normed_x, normed_seq)

        # norm_aff1 = F.normalize(aff1, p=2, dim=-1)
        # aff1 = F.sigmoid(aff1) - 0.5

        aff = aff.reshape(b * t, h_w, h_w)
        A12s = stoch_mat(aff, do_dropout=False)
        # A12s = aff1 @ A12s
        sk = A12s.reshape(b, t, h_w, h_w)
        # aff1_feature = torch.einsum('bcth,bthh->bcth', seq1, sk1)
        aff_feature = torch.einsum('bthh,bcth->bcth', sk, seq)
        return aff_feature


if __name__ == '__main__':
    A = torch.rand(2, 128, 6, 7, 7) # b,t,c,h,w
    net = AffSkLayer(128)
    A = net(A)

