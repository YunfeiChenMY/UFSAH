import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from spectral_norm import spectral_norm as SpectralNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm, Dropout, Linear
from torch.nn.functional import normalize
from pygat.layers import GraphAttentionLayer
from torch.nn.functional import normalize
import scipy.sparse as sp
import numpy as np
from torch.autograd import Variable
from torch_geometric.nn import HypergraphConv
from tools import build_G_from_S, generate_robust_S
from model import FastKAN


class ImgNet(nn.Module):
    def __init__(self, code_len, img_feat_len):
        super(ImgNet, self).__init__()
        # self.alexnet = torchvision.models.alexnet(pretrained=True)
        # self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        b = 4096
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.fc_encode1 = nn.Linear(img_feat_len, b)
        self.fc_encode2 = nn.Linear(b, int(code_len))
        # out_feature = 1024
        # nid = 8
        # self.attentions = [GraphAttentionLayer(img_feat_len, out_feature, dropout=0.6, alpha=0.2, concat=True) for _ in
        #                    range(nid)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        # self.out_att = GraphAttentionLayer(out_feature * nid, int(code_len), dropout=0.6, alpha=0.2, concat=False)
        #
        #
        # self.fc_encode11 = nn.Linear(img_feat_len, b)
        # self.fc_encode21 = nn.Linear(b, int(code_len))
        #
        # self.fc_encode12 = nn.Linear(img_feat_len, b)
        # self.fc_encode22 = nn.Linear(b, int(code_len/4))
        #
        # self.fc_encode13 = nn.Linear(img_feat_len, b)
        # self.fc_encode23 = nn.Linear(b, int(code_len/4))
        # self.BN2 = nn.BatchNorm1d(code_len)
        self.alpha = 1.0
        torch.nn.init.normal_(self.fc_encode2.weight, mean=0.0, std=1)
        self.dropout = 0.3


    def forward(self, x):
        self.alpha = 1
        feat = torch.relu(self.fc_encode1(x))
        feat = self.fc_encode2(self.dp(feat))
        code = torch.tanh(self.alpha * feat)#2

        # x1 = F.dropout(x, self.dropout, training=True)
        # x1 = torch.cat([att(x1, code.mm(code.t())) for att in self.attentions], dim=1)
        #
        # x1 = F.dropout(x1, self.dropout, training=True)
        # feat1 = self.out_att(x1, x.mm(x.t()))
        # code1 = torch.tanh(self.alpha * feat1)

        # feat1 = torch.relu(self.fc_encode11(x.mm(x.t()).mm(x)))
        # feat1 = self.fc_encode21(self.dp(feat1))
        # code1 = torch.tanh(self.alpha * feat1)#2
        #
        # feat2 = torch.relu(self.fc_encode12(x))
        # feat2 = self.fc_encode22(self.dp(feat2))
        # code2 = torch.tanh(self.alpha * feat2)#2
        #
        # feat3 = torch.relu(self.fc_encode13(x))
        # feat3 = self.fc_encode23(self.dp(feat3))
        # code3 = torch.tanh(self.alpha * feat3)#2

        return code#torch.concat((code, code1), dim=1).cuda()

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        a = 4096
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(txt_feat_len, a)
        self.fc3 = nn.Linear(a, int(code_len))
        self.alpha = 1.0
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=1)
        self.dropout = 0.3

    def forward(self, x):
        self.alpha = 1
        feat = torch.relu(self.fc1(x))
        feat = self.fc3(self.dp(feat))
        code = torch.tanh(self.alpha * feat)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class NetHY(nn.Module):
    def __init__(self, code_len, img_feat_len, eplision, k):
        super(NetHY, self).__init__()
        b = 4096
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.fc_encode1 = HypergraphConv(img_feat_len, b)
        self.fc_encode2 = HypergraphConv(b, int(code_len))
        self.alpha = 1.0
        self.dropout = 0.3
        self.eplision = eplision
        self.k = k


    def forward(self, x, S):
        G = build_G_from_S(S, self.k , self.eplision)
        self.alpha = 1
        G = torch.LongTensor(edge_list(G)).cuda()
        feat = torch.relu(self.fc_encode1(x, G))
        feat = self.fc_encode2(self.dp(feat), G)
        code = torch.tanh(self.alpha * feat)#2


        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class ImgNetHY(nn.Module):
    def __init__(self, code_len, img_feat_len, eplision, k):
        super(ImgNetHY, self).__init__()
        b = 4096
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.fc_encode1 = FastKAN([img_feat_len, b])
        self.fc_encode2 = HypergraphConv(b, int(code_len))
        self.fc_Decode1 = FastKAN([int(code_len), img_feat_len*2])


    def forward(self, x, G):
        feat = torch.relu(self.fc_encode1(x))
        feat = self.fc_encode2(self.dp(feat), G)
        code = torch.tanh(10 * feat)
        feat = torch.relu(self.fc_Decode1(code))
        return code, feat

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


def get_hyperedge_attr(features, hyperedge_index, type='mean'):
    if type == 'mean':
        hyperedge_attr = None
        samples = features[hyperedge_index[0]]
        labels = hyperedge_index[1]

        labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        hyperedge_attr = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
        hyperedge_attr = hyperedge_attr / labels_count.float().unsqueeze(1)
    return hyperedge_attr

def edge_list(G):
    mask = G != -1.5

    # Use the mask to find the indices
    list_e, cla = mask.nonzero(as_tuple=True)

    # Convert indices to lists if needed (they are returned as tensors)
    list_e = list_e.tolist()
    cla = cla.tolist()

    # Combine the lists into a final result
    res = [list_e, cla]
    return res
class TxtNetHY(nn.Module):
    def __init__(self, code_len, txt_feat_len, eplision, k):
        super(TxtNetHY, self).__init__()
        b = 4096
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.fc_encode1 = FastKAN([txt_feat_len, b])
        self.fc_encode2 = HypergraphConv(b, int(code_len))
        self.fc_Decode1 = FastKAN([int(code_len), 2*txt_feat_len])

    def forward(self, x, G):
        feat = torch.relu(self.fc_encode1(x))
        feat = self.fc_encode2(self.dp(feat), G)
        code = torch.tanh(10 * feat)
        feat = torch.relu(self.fc_Decode1(code))
        return code, feat

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class DeImgNetHY(nn.Module):
    def __init__(self, code_len, img_feat_len, eplision, k):
        super(DeImgNetHY, self).__init__()
        b = 1024
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        # self.fc_encode1 = HypergraphConv(code_len, b)
        self.fc_encode1 = FastKAN([code_len, b])
        # self.fc_encode2 = HypergraphConv(b, int(img_feat_len))
        self.fc_encode2 = FastKAN([b, int(img_feat_len)])
        self.alpha = 1.0
        self.dropout = 0.3
        self.eplision = eplision
        self.k = k


    def forward(self, x, G):
        feat = torch.relu(self.fc_encode1(x))
        feat = torch.relu(self.fc_encode2(feat))

        return feat

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class DeTxtNetHY(nn.Module):
    def __init__(self, code_len, txt_feat_len, eplision, k):
        super(DeTxtNetHY, self).__init__()
        a = 1024
        # self.BN1 = nn.BatchNorm1d(a)
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        # self.fc1 = HypergraphConv(code_len, a)
        self.fc1 = FastKAN([code_len, a])
        # self.fc3 = HypergraphConv(a, int(txt_feat_len))
        self.fc3 = FastKAN([a, int(txt_feat_len)])
        self.alpha = 1.0
        self.dropout = 0.3
        self.eplision = eplision
        self.k = k

    def forward(self, x, G):
        self.alpha = 1
        feat = torch.relu(self.fc1(x))
        feat = torch.relu(self.fc3(feat))
        return feat

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class DeImgNet(nn.Module):
    def __init__(self, code_len, img_feat_len):
        super(DeImgNet, self).__init__()
        # self.alexnet = torchvision.models.alexnet(pretrained=True)
        # self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.fc_encode1 = nn.Linear(code_len, img_feat_len)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        # self.BN1 = nn.BatchNorm1d(4096)
        self.fc_encode2 = nn.Linear(img_feat_len, img_feat_len)
        self.fc_encode3 = nn.Linear(img_feat_len, img_feat_len)
        self.BN2 = nn.BatchNorm1d(img_feat_len)
        # self.fc_encode = nn.Linear(code_len, 4096)
        self.alpha = 1.0
        self.Dealpha = 0.5


    def forward(self, x):
        # x = self.alexnet.features(x)
        # x = x.view(x.size(0), -1)
        # feat = self.alexnet.classifier(x)
        # hid = F.relu(self.BN1(self.fc_encode1(x)))
        self.alpha = 1
        hid = torch.sigmoid(self.alpha * self.fc_encode1(x))
        # hid = self.fc_encode2(hid)
        # hid = self.fc_encode3(hid)
        # hid = self.fc_encode3(hid)

        # hid = self.fc_encode(x)
        # code = hid #* self.Dealpha

        return x, hid, hid

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class DeTxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(DeTxtNet, self).__init__()
        a = 4096
        self.fc1 = nn.Linear(code_len, txt_feat_len)
        # self.BN1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(a, a)
        self.fc3 = nn.Linear(a, txt_feat_len)
        # self.BN2 = nn.BatchNorm1d(txt_feat_len)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.alpha = 1.0
        self.Dealpha = 0.5

    def forward(self, x):
        # feat = F.relu(self.BN1(self.fc1(x)))
        self.alpha = 1
        hid = torch.sigmoid(self.alpha * self.fc1(x))
        # feat = self.fc2(hid)
        # # hid = self.BN2(self.fc3(feat))
        # hid = self.fc3(feat)
        # code = hid #* self.Dealpha + self.Dealpha
        return hid, hid, hid

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class GenHash(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(GenHash, self).__init__()
        a = 1024
        self.fc1 = nn.Linear(code_len * 2, code_len)
        # self.BN1 = nn.BatchNorm1d(code_len * 2)
        # # self.fc2 = nn.Linear(1024, 1024)
        self.dp = nn.Dropout(0.3)
        self.fc3 = nn.Linear(a, code_len)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.BN2 = nn.BatchNorm1d(code_len)
        self.alpha = 1.0
        self.Dealpha = 0.5
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=1)

    def forward(self, x):
        # feat = F.relu(self.BN1(self.fc1(x)))
        # feat = F.relu(self.fc1(x))
        # hid = self.fc2(feat)
        # hid = self.BN2(self.fc3(feat))
        hid = self.fc1(x)
        # hid = torch.relu(self.fc3(self.dp(hid)))
        self.alpha = 1
        code = torch.tanh(self.alpha * hid) #* self.Dealpha + self.Dealpha
        B = torch.sign(code)
        return code, B

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class FuseTransEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nhead, code_len):
        super(FuseTransEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        self.transformerEncoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.d_model = hidden_size
        self.sigal_d = int(self.d_model / 2)
        # self.fc_encode1 = nn.Linear(self.sigal_d, int(code_len/2))
        # self.fc_encode2 = nn.Linear(self.sigal_d, int(code_len/2))
        self.fc_encode2 = nn.Linear(self.d_model, int(code_len))

    def forward(self, tokens):
        # tokens = tokens.reshape((1, tokens.shape[0], tokens.shape[1]))
        encoder_X = self.transformerEncoder(tokens)
        # encoder_X = encoder_X.reshape((encoder_X.shape[1], encoder_X.shape[2]))
        encoder_X_r = encoder_X.reshape(-1, self.d_model)
        encoder_X_r = normalize(encoder_X_r, p=2, dim=1)
        img, txt = encoder_X_r[:, :self.sigal_d], encoder_X_r[:, self.sigal_d:]
        # hashHI = torch.tanh(self.fc_encode1(img))
        # hashHT = torch.tanh(self.fc_encode2(txt))
        # hashH = torch.concat((hashHI, hashHT), dim=1).cuda()
        hashH = torch.tanh(self.fc_encode2(encoder_X_r))
        hashB = torch.sign(hashH)
        return hashB, hashH, img, txt

class GetITNet(nn.Module):
    def __init__(self, code_len, img_feat_len, nhid=1024, image_size=1024, dropout=0.6, alpha=0.2, nheads=8):
        super(GetITNet, self).__init__()

        # num_layers, hidden_size, nhead = 2, img_feat_len, 4
        #
        # encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        # self.transformerEncoder = TransformerEncoder(encoder_layer= encoder_layer, num_layers= num_layers)
        # self.alexnet = torchvision.models.alexnet(pretrained=True)
        # self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.dropout = dropout
        self.decode = nn.Linear(code_len, image_size)
        self.alpha = 1.0
        b = 4096
        self.fc_encode1 = nn.Linear(img_feat_len, b)
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        # self.fc_encode3 = nn.Linear(b, b)
        self.fc_encode2 = nn.Linear(b, code_len)
        # self.BN2 = nn.BatchNorm1d(code_len)
        self.alpha = 1.0
        torch.nn.init.normal_(self.fc_encode2.weight, mean=0.0, std=1)
        self.attentions = [GraphAttentionLayer(img_feat_len, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, code_len, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):
        # x = self.alexnet.features(x)
        # x = x.view(x.size(0), -1)
        # feat = self.alexnet.classifier(x)
        # feat = F.relu(self.BN1(self.fc_encode1(x)))
        # x = x.reshape((1, x.shape[0], x.shape[1]))
        # # print(x.shape)
        #
        # x = self.transformerEncoder(x)
        #
        # x = x.reshape((x.shape[1], x.shape[2]))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.out_att(x, adj)
        # # self.alpha = 1
        # code = torch.tanh(self.alpha * x)

        # code= normalize(code, p=2, dim=1)
        feat = torch.relu(self.fc_encode1(x))
        # feat = F.relu(self.fc_encode3(feat))
        feat = self.fc_encode2(self.dp(feat))
        # hid = self.BN2(hid)
        code = torch.tanh(1 * feat)#2

        return code, code, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GetImgNet(nn.Module):
    def __init__(self, code_len, img_feat_len, nhid=1024, image_size=1024, dropout=0.6, alpha=0.2, nheads=8):
        super(GetImgNet, self).__init__()

        # num_layers, hidden_size, nhead = 2, img_feat_len, 4
        #
        # encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        # self.transformerEncoder = TransformerEncoder(encoder_layer= encoder_layer, num_layers= num_layers)
        # self.alexnet = torchvision.models.alexnet(pretrained=True)
        # self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.dropout = dropout
        self.decode = nn.Linear(code_len, image_size)
        self.alpha = 1.0
        b = 4096
        self.fc_encode1 = nn.Linear(img_feat_len, b)
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        # self.fc_encode3 = nn.Linear(b, b)
        self.fc_encode2 = nn.Linear(b, code_len)
        # self.BN2 = nn.BatchNorm1d(code_len)
        self.alpha = 1.0
        torch.nn.init.normal_(self.fc_encode2.weight, mean=0.0, std=1)
        self.attentions = [GraphAttentionLayer(img_feat_len, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, code_len, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # x = self.alexnet.features(x)
        # x = x.view(x.size(0), -1)
        # feat = self.alexnet.classifier(x)
        # feat = F.relu(self.BN1(self.fc_encode1(x)))
        # x = x.reshape((1, x.shape[0], x.shape[1]))
        # # print(x.shape)
        #
        # x = self.transformerEncoder(x)
        #
        # x = x.reshape((x.shape[1], x.shape[2]))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        # self.alpha = 1
        code = torch.tanh(self.alpha * x)

        # code= normalize(code, p=2, dim=1)
        # feat = torch.relu(self.fc_encode1(x))
        # # feat = F.relu(self.fc_encode3(feat))
        # feat = self.fc_encode2(self.dp(feat))
        # # hid = self.BN2(hid)
        # code = torch.tanh(1 * feat)#2

        return code, code, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GetTxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len, nhid=1024, image_size=1024, dropout=0.6, alpha=0.2, nheads=8):
        super(GetTxtNet, self).__init__()
        a = 4096
        self.dropout = dropout
        self.decode = nn.Linear(code_len, image_size)
        self.alpha = 1.0
        self.fc1 = nn.Linear(txt_feat_len, a)
        # self.BN1 = nn.BatchNorm1d(a)
        self.dp = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(a, a)
        # self.fc21 = nn.Linear(a, a)
        # self.fc22 = nn.Linear(a, a)
        self.fc3 = nn.Linear(a, code_len)
        # self.BN2 = nn.BatchNorm1d(code_len)
        self.alpha = 1.0
        # torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=1)
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=1)
        self.attentions = [GraphAttentionLayer(txt_feat_len, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, code_len, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # feat = F.relu(self.BN1(self.fc1(x)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        # self.alpha = 1
        code = torch.tanh(self.alpha * x)
        # feat = torch.relu(self.fc1(x))
        # # feat = self.relu(self.fc2(feat))
        # # feat = ac(self.fc21(feat))
        # # feat = ac(self.fc22(feat))
        # # hid = self.BN2(self.fc3(feat))
        # feat = self.fc3(self.dp(feat))
        # code = torch.tanh(1 * feat)#20

        # code= normalize(code, p=2, dim=1)
        return code, code, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GeImgNet(nn.Module):
    def __init__(self, code_len):
        super(GeImgNet, self).__init__()
        # self.alexnet = torchvision.models.alexnet(pretrained=True)
        # self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.fc_encode1 = nn.Linear(code_len, 512)
        # self.fc_encode2 = nn.Linear(1024, 1024)
        self.fc_encode3 = nn.Linear(512, 4096)
        self.alpha = 1.0
        self.Dealpha = 1.0


    def forward(self, x):
        # x = self.alexnet.features(x)
        # x = x.view(x.size(0), -1)
        # feat = self.alexnet.classifier(x)
        hid = self.fc_encode1(x)
        # hid = self.fc_encode2(hid)
        hid = self.fc_encode3(hid)
        code = F.relu(self.alpha * hid) #* self.Dealpha

        return x, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class GeTxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(GeTxtNet, self).__init__()
        self.fc1 = nn.Linear(code_len, 512)
        # self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(512, txt_feat_len)
        self.alpha = 1.0
        self.Dealpha = 0.5

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        hid = self.fc3(hid)
        code = (F.tanh(self.alpha * hid)) #* self.Dealpha
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class Txt2Img(nn.Module):
    def __init__(self, txt_code_len, img_code_len):
        super(Txt2Img, self).__init__()
        self.fc1 = nn.Linear(txt_code_len, 1024)
        self.fc2 = nn.Linear(1024, img_code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = F.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class Img2Txt(nn.Module):
    def __init__(self, img_code_len, txt_code_len):
        super(Img2Txt, self).__init__()
        self.fc1 = nn.Linear(img_code_len, 1024)
        self.fc2 = nn.Linear(1024, txt_code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = F.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)



class DiscriminatorImg(nn.Module):
    """
    Discriminator network with PatchGAN.
    Reference: https://github.com/yunjey/stargan/blob/master/model.py
    """
    def __init__(self, code_len):
        super(DiscriminatorImg, self).__init__()
        self.fc_encode = nn.Linear(4096, code_len + 1)
        self.alpha = 1.0

    def forward(self, x):
        hid = self.fc_encode(x)
        code = F.tanh(self.alpha * hid)
        return code.squeeze()

class DiscriminatorText(nn.Module):
    """
    Discriminator network with PatchGAN.
    Reference: https://github.com/yunjey/stargan/blob/master/model.py
    """
    def __init__(self, code_len):
        super(DiscriminatorText, self).__init__()
        self.fc_encode = nn.Linear(1386, code_len + 1)
        self.alpha = 1.0

    def forward(self, x):
        hid = self.fc_encode(x)
        code = F.tanh(self.alpha * hid)
        return code.squeeze()


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self, gan_mode, target_real_label=0.0, target_fake_label=1.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, label, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            real_label = self.real_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, real_label], dim=-1)
        else:
            fake_label = self.fake_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, fake_label], dim=-1)
        return target_tensor

    def __call__(self, prediction, label, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(label, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class ImageMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024, 128, 1024], dropout=0.1):
        super(ImageMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()
        torch.nn.init.normal_(self.tohash.weight, mean=0.0, std=1)

    def _ff_block(self, x):
        # x = normalize(x, p=2, dim=1)
        feat = torch.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = torch.tanh(hid)
        return out

    def forward(self, X):
        mlp_output = self._ff_block(X)
        mlp_output = normalize(mlp_output, p=2, dim=1)
        return mlp_output


class TextMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024, 128, 1024], dropout=0.1):
        super(TextMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()
        torch.nn.init.normal_(self.tohash.weight, mean=0.0, std=1)

    def _ff_block(self, x):
        # x = normalize(x, p=2, dim=1)
        feat = torch.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = torch.tanh(hid)
        return out

    def forward(self, X):
        mlp_output = self._ff_block(X)
        mlp_output = normalize(mlp_output, p=2, dim=1)
        return mlp_output

class GATNet(nn.Module):
    def normalize_adj(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    def generate_txt_graph(self, txt):
        """Generate text graph structure."""
        txt_feature = txt
        txt = F.normalize(txt)
        adj = txt.mm(txt.t())
        adj[adj < 0.8] = 0
        adj = torch.sign(adj)
        adj2triad = sp.csr_matrix(adj)
        #adj2triad = adj2triad + adj2triad.T.multiply(adj2triad.T > adj2triad) - adj2triad.multiply(adj2triad.T > adj2triad)
        adj = self.normalize_adj(adj2triad)
        adj = torch.FloatTensor(np.array(adj.todense()))
        #adjacencyMatrix = self.sparse_mx_to_torch_sparse_tensor(adj)
        return txt_feature, adj
    def __init__(self, code_len, txt_feat_len, nhid=1024, image_size=512, dropout=0.6, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GATNet, self).__init__()
        self.dropout = dropout
        self.decode = nn.Linear(code_len, image_size)
        self.alpha = 1.0

        self.attentions = [GraphAttentionLayer(txt_feat_len, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, code_len, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):
        txt_feature, adjacencyMatrix = self.generate_txt_graph(x)
        adj = adjacencyMatrix.cuda()
        x = Variable(torch.FloatTensor(x.numpy()).cuda()).cuda()
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        code = torch.tanh(self.alpha * x)
        decoded_i = torch.sigmoid(self.decode(code))
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class GAINet(nn.Module):
    def normalize_adj(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    def generate_txt_graph(self, txt):
        """Generate text graph structure."""
        txt_feature = txt
        txt = F.normalize(txt)
        adj = txt.mm(txt.t())
        adj[adj < 0.8] = 0
        adj = torch.sign(adj)
        # adj = torch.tanh(adj)
        adj2triad = sp.csr_matrix(adj)
        #adj2triad = adj2triad + adj2triad.T.multiply(adj2triad.T > adj2triad) - adj2triad.multiply(adj2triad.T > adj2triad)
        adj = self.normalize_adj(adj2triad)
        adj = torch.FloatTensor(np.array(adj.todense()))
        #adjacencyMatrix = self.sparse_mx_to_torch_sparse_tensor(adj)
        return txt_feature, adj
    def __init__(self, code_len, img_feat_len, nhid=1024, image_size=512, dropout=0.6, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAINet, self).__init__()
        self.dropout = dropout
        self.decode = nn.Linear(code_len, image_size)
        self.alpha = 1.0

        self.attentions = [GraphAttentionLayer(img_feat_len, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, code_len, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):
        txt_feature, adjacencyMatrix = self.generate_txt_graph(x)
        adj = adjacencyMatrix.cuda()
        x = Variable(torch.FloatTensor(x.numpy()).cuda()).cuda()
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        code = torch.tanh(self.alpha * x)
        decoded_i = torch.sigmoid(self.decode(code))
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)