import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from metric_duapreDual import compress_wiki, compress, calculate_map, calculate_top_map, p_topK, ContrastiveLoss, compress_nus, calculate_top_map_nus
# import datasetspre as datasets
import settingsnuspre as settings
from models3pre3Dual import ImgNetHY, TxtNetHY, DeTxtNet, DeImgNet, GenHash, GetTxtNet, GetImgNet, FuseTransEncoder, GetITNet, DeImgNetHY, DeTxtNetHY
from load_data import get_loader_flickr, get_loader_nus, get_loader_coco, get_loader_flickr_CLIP, get_loader_flickr_fea, get_loader_nuswide_CLIP, get_loader_coco_CLIP
import os.path as osp
import sys

import pandas as pd
from RL_brainQ import QLearningTable
from tools import build_G_from_S, generate_robust_S, edge_list
import torch.nn.functional as F
from utils import *
import time

import argparse
import optuna
import networkx as nx

import ast
import re


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda:0', temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())
    def forward(self, emb_i, emb_j):
        z_i = F.normalize(torch.exp(emb_i), dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(torch.exp(emb_j), dim=1)     # (bs, dim)  --->  (bs, dim)
        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2).cuda() #+ scipy.spatial.distance.cdist(representations.unsqueeze(1).detach().numpy(), representations.unsqueeze(0).detach().numpy(), 'hamming').cuda()    # simi_mat: (2*bs, 2*bs)

        # similarity_matrix = torch.where(similarity_matrix < 0.3, 0, 1)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
class Session:
    def __init__(self, train_loader, test_loader, database_loader, train_dataset, test_dataset, database_dataset, data_train, a, a2, a3):

        self.logger = settings.logger

        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.cuda.set_device(settings.GPU_ID)
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.database_dataset = database_dataset
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.database_loader = database_loader
        self.I_tr, self.T_tr, self.L_tr = data_train

        # txt_feat_len = datasets.txt_feat_len
        txt_feat_len = self.T_tr.shape[1]
        img_feat_len = self.I_tr.shape[1]
        self.ContrastiveLoss = ContrastiveLoss(batch_size=settings.BATCH_SIZE, device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu"))

        self.CodeNet_I = ImgNetHY(code_len=settings.CODE_LEN, img_feat_len=img_feat_len, eplision = 0.9, k = 5)
        self.FeatNet_I = ImgNetHY(code_len=settings.CODE_LEN, img_feat_len=img_feat_len, eplision = 0.9, k = 5)
        self.CodeNet_T = TxtNetHY(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len, eplision = 0.9, k = 5)


        self.DeCodeNet_I = DeImgNetHY(code_len=settings.CODE_LEN, img_feat_len=img_feat_len, eplision = 0.9, k = 5)
        self.DeCodeNet_T = DeTxtNetHY(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len, eplision = 0.9, k = 5)


        self.GetNet_I = GetImgNet(code_len=settings.CODE_LEN, img_feat_len=img_feat_len)
        self.GetNet_T = GetTxtNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)
        self.GetNet = GetITNet(code_len=settings.CODE_LEN, img_feat_len=5000)

        # self.DeCodeNet_I = DeImgNet(code_len=settings.CODE_LEN, img_feat_len=img_feat_len)
        # self.DeCodeNet_T = DeTxtNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)
        self.genHash = GenHash(code_len=settings.CODE_LEN, txt_feat_len=settings.BATCH_SIZE)
        num_layers, nhead = 2,  4

        self.FuseTrans = FuseTransEncoder(num_layers, 1024, nhead, settings.CODE_LEN).cuda()

        if settings.DATASET == "WIKI":
            self.opt_I = torch.optim.SGD([{'params': self.CodeNet_I.fc_encode.parameters(), 'lr': settings.LR_IMG},
                                          {'params': self.CodeNet_I.alexnet.classifier.parameters(),
                                           'lr': settings.LR_IMG}],
                                         momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)
            self.opt_DeI = torch.optim.SGD([{'params': self.DeCodeNet_I.fc_encode.parameters(), 'lr': settings.LR_IMG}],
                                           momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

        if settings.DATASET == "MIRFlickr" or settings.DATASET == "NUSWIDE"or settings.DATASET == "MSCOCO":
            # self.opt_I = torch.optim.Adam(self.CodeNet_I.parameters(), lr=settings.LR_IMG, betas=(0.5, 0.999))
            self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)
            # self.opt_DeI = torch.optim.SGD(self.DeCodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

            self.opt_DeI = torch.optim.Adam(self.DeCodeNet_I.parameters(), lr=settings.LR_IMG)

        self.optimizer_FuseTrans = torch.optim.Adam(self.FuseTrans.parameters(), lr=1e-4, betas=(0.5, 0.999))

        # self.opt_T = torch.optim.Adam(self.CodeNet_T.parameters(), lr=settings.LR_TXT, betas=(0.5, 0.999))

        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)
        # self.opt_DeT = torch.optim.SGD(self.DeCodeNet_T.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)
        self.opt_DeT = torch.optim.Adam(self.DeCodeNet_T.parameters(), lr=settings.LR_TXT)

        img_norm = F.normalize(torch.Tensor(self.I_tr)).cuda()
        txt_norm = F.normalize(torch.Tensor(self.T_tr)).cuda()
        self.img_norm = img_norm
        self.txt_norm = txt_norm
        self.cS = img_norm.mm(txt_norm.t()).cuda()

        f_norm = torch.concat((img_norm, txt_norm), dim=1).cuda()
        self.gsI = np.corrcoef(img_norm.cpu(), rowvar = 1)#*2-1#[0][1]
        self.gsI = (torch.tensor(self.gsI, dtype=torch.float32))
        self.gsT = np.corrcoef(txt_norm.cpu(), rowvar = 1)#*2-1#[0][1]
        self.gsT = (torch.tensor(self.gsT, dtype=torch.float32))
        self.gs = np.corrcoef(f_norm.cpu(), rowvar = 1)#*2-1#[0][1]
        self.gs = (torch.tensor(self.gs, dtype=torch.float32))
        self.gsid = torch.eye(4992)
        self.ContrastiveLoss = ContrastiveLoss(batch_size=settings.BATCH_SIZE, device=self.device)

    def train(self, epoch, l1, l2, l3, l4, l5, l6, l7, l8):
        self.CodeNet_I.cuda().train()
        self.CodeNet_T.cuda().train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)



        for No, (F_I, F_T, _, index_) in enumerate(self.train_loader):
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())
            F_I = Variable(torch.FloatTensor(F_I.numpy()).cuda())

            F_F = torch.cat((F_I, F_T), 1)

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            SI = F.normalize(F_I).mm(F.normalize(F_I).t()).cuda()
            ST = F.normalize(F_T).mm(F.normalize(F_T).t()).cuda()
            # 计算各元素之间的欧式距离矩阵
            SDI = torch.cdist(F_I, F_I, p=2).cuda()
            SDT = torch.cdist(F_T, F_T, p=2).cuda()
            II = SDI.max(dim=1, keepdim=True)[0]
            SDI = (SDI/II)**int(20*l8+10)
            IT = SDT.max(dim=1, keepdim=True)[0]
            SDT = (SDT / IT)**int(20*l8+10)
            SI = SI / torch.exp(SDI)
            ST = ST / torch.exp(SDT)
            GV = l3 * ST + (1 - l3) * SI
            A = GV.mm(GV) / settings.BATCH_SIZE
            G = l4 * GV + (1-l4)*GV.mm(GV)/settings.BATCH_SIZE
            GT = build_G_from_S(G, 5, l6)
            GRT = torch.LongTensor(edge_list(GT)).cuda()

            code_I, FI_ = self.CodeNet_I(F_I.cuda(), GRT)
            code_T, FT_ = self.CodeNet_T(F_T.cuda(), GRT)

            # B_I = F.normalize(code_I)
            # B_T = F.normalize(code_T)
            loss8 = self.ContrastiveLoss(code_I, code_T)
            # loss10 = self.ContrastiveLoss(FI_, FT_)
            loss3 = F.mse_loss(code_I.cuda(), torch.sign(code_T).cuda()) + F.mse_loss(code_T.cuda(), torch.sign(code_I).cuda())
            baseline = torch.eye(settings.BATCH_SIZE)
            baseline = 2 * baseline - 1

            BI_BI = code_I.mm(code_I.t()).cuda()/128
            BT_BT = code_T.mm(code_T.t()).cuda()/128
            BI_BT = code_I.mm(code_T.t()).cuda()/128

            loss1 = F.mse_loss(BI_BI.cuda(), baseline.cuda()) + F.mse_loss(BI_BT.cuda(), baseline.cuda()) + F.mse_loss(BT_BT.cuda(), baseline.cuda())
            loss4 = F.mse_loss(FI_.cuda(), F_F) + F.mse_loss(FT_.cuda(), F_F)

            loss = l1 * loss8 + l2 * loss3 + l5*loss4 + l7*loss1
            loss.backward()
            self.opt_I.step()
            self.opt_T.step()

    def loss_maxcut_weighted(self, probs, C, dct, weights, hyper=False):
        x = probs.squeeze()
        loss = 0
        for c, w in zip(C, weights):
            temp_1s = 1
            temp_0s = 1
            if hyper:
                for index in c:
                    temp_1s *= (1 - x[dct[index]])
                    temp_0s *= (x[dct[index]])
            else:
                for index in c[0:2]:
                    temp_1s *= (1 - x[dct[index]])
                    temp_0s *= (x[dct[index]])
            temp = (temp_1s + temp_0s)
            loss += (temp * w)
        return loss

    def greedy_dominating_set(self, graph):
        """
            贪心算法求解支配集问题
            :param graph: NetworkX 图
            :return: 支配集顶点集合
            """
        # 初始化集合和所有节点集合
        dominating_set = set()
        all_nodes = set(graph.nodes())
        covered_nodes = set()

        # 贪心选择，直到覆盖所有节点
        while covered_nodes != all_nodes:
            # 选择覆盖未被覆盖节点数最多的节点
            max_node = max((node for node in all_nodes if node not in covered_nodes),
                           key=lambda node: len(set(graph.neighbors(node)) - covered_nodes))
            dominating_set.add(max_node)
            covered_nodes.update(graph.neighbors(max_node))
            covered_nodes.add(max_node)

        return dominating_set
    def eval(self, l1, l2, l3, l4, l5, l6, l7, l8):
        # self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()

        if settings.DATASET == "WIKI":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_wiki(self.database_loader, self.test_loader,
                                                                   self.CodeNet_I, self.CodeNet_T, self.database_dataset,
                                                                   self.test_dataset)

        if settings.DATASET == "MIRFlickr":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I,
                                                              self.CodeNet_T,
                                                              self.database_dataset, self.test_dataset, l3, l4, l5, l6, l8)
        if settings.DATASET == "NUSWIDE" or settings.DATASET == "MSCOCO":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_nus(self.database_loader, self.test_loader, self.CodeNet_I,
                                                              self.CodeNet_T,
                                                              self.database_dataset, self.test_dataset)

        # MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        # MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        # MAP_I2TA = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
        # MAP_T2IA = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
        # K = [5000]
        # MAP_I2T5 = p_topK(qu_BI, re_BT, qu_L, re_L, K)
        # MAP_T2I5 = p_topK(qu_BT, re_BI, qu_L, re_L, K)
        # MAP_I2T5 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=1)
        # MAP_T2I5 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=1)
        # print(MAP_I2T5)
        # print(MAP_T2I5)
        # MAP_I2T5 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=200)
        # MAP_T2I5 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=200)
        # print(MAP_I2T5)
        # print(MAP_T2I5)
        # MAP_I2T5 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=400)
        # MAP_T2I5 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=400)
        # print(MAP_I2T5)
        # print(MAP_T2I5)
        # MAP_I2T5 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=1000)
        # MAP_T2I5 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=1000)
        # print(MAP_I2T5)
        # print(MAP_T2I5)
        # MAP_I2T5 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=1500)
        # MAP_T2I5 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=1500)
        # print(MAP_I2T5)
        # print(MAP_T2I5)
        # MAP_I2T5 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=2000)
        # MAP_T2I5 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=2000)
        # print(MAP_I2T5)
        # print(MAP_T2I5)
        # MAP_I2T5 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=2500)
        # MAP_T2I5 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=2500)
        # print(MAP_I2T5)
        # print(MAP_T2I5)
        # MAP_I2T5 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=3000)
        # MAP_T2I5 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=3000)
        # print(MAP_I2T5)
        # print(MAP_T2I5)
        # MAP_I2T5 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=3500)
        # MAP_T2I5 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=3500)
        # print(MAP_I2T5)
        # print(MAP_T2I5)
        # MAP_I2T5 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=4000)
        # MAP_T2I5 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=4000)
        # print(MAP_I2T5)
        # print(MAP_T2I5)
        # MAP_I2T5 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=4500)
        # MAP_T2I5 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=4500)
        # print(MAP_I2T5)
        # print(MAP_T2I5)
        MAP_I2T5 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I5 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        # print(MAP_I2T5)
        # print(MAP_T2I5)
        MAP_I2T = 0#calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = 0#calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        # self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        self.logger.info('MAP: %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f' % (l1, l2, l3, l4, l5, l6, l7, l8, MAP_I2T5, MAP_T2I5, MAP_I2T5+MAP_T2I5))
        # self.logger.info('--------------------------------------------------------------------')
        # K = [200, 5000]
        # MAP_I2T0 = p_topK(qu_BI, re_BT, qu_L, re_L, K)
        # MAP_T2I0 = p_topK(qu_BT, re_BI, qu_L, re_L, K)
        # self.logger.info(MAP_I2T0)
        # self.logger.info(MAP_T2I0)
        return MAP_I2T5+MAP_T2I5

    def save_checkpoints(self, step, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])

    def cal_similarity2(self, F_I, F_T):
        a1 = settings.BETA
        a2 = 0.6
        K = 3000
        batch_size = F_I.size(0)
        size = batch_size
        top_size = K

        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        F_T = F.normalize(F_T)
        S_T = F_T.mm(F_T.t())

        S1 = a1 * S_I + (1 - a1) * S_T

        m, n1 = S1.sort()
        S1[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(-1)] = 0.

        S2 = 2.0 / (1 + torch.exp(-S1)) - 1 + torch.eye(S1.size(0)).cuda()
        S2 = (S2 + S2.t())/2
        S = a2 * S1 + (1 - a2) * S2

        return S
    def cal_similarity(self, F_I, F_T, l1, l4, l3):
        a1 = 0.4
        a2 = 0.4
        a3 = 0.3
        a4 = 2.0
        l2 = 0.8
        knn_number = 320  # l3

        # print(l3)
        scale = 1
        batch_size = F_I.size(0)

        F_I = F.normalize(F_I).cuda()
        S_I = F_I.mm(F_I.t()).cuda()
        F_T = F.normalize(F_T).cuda()
        S_T = F_T.mm(F_T.t()).cuda()
        a4 = 0.2
        a6 = 0.8
        S_pair = a6 * S_I.cuda() + (1 - a6) * S_T.cuda()
        a5 = 0.8
        Sp = (1 - a5) * S_pair.cuda() + a5 * (
                a4 * S_T.mm(S_T.t()).cuda() + (1 - a4) * S_I.mm(S_T.t()).cuda()) / batch_size
        pro = Sp  # * settings.MU
        size = batch_size
        top_size = knn_number
        m, n1 = S_pair.sort()
        pro[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(
            -1)] = 0.
        pro[torch.arange(size).view(-1), n1[:, -1:].contiguous().view(
            -1)] = 0.
        # pro = pro / pro.sum(1).view(-1, 1)
        # pro_dis = pro.mm(pro.t())
        pro_dis = pro * scale
        S = pro_dis
        S = torch.eye(batch_size,batch_size)
        S = S * 2.0 - 1
        # for i in range(batch_size):
        #     S[i] = S[i] / S[i][i]
        return S

    def crossview_contrastive_Loss(self, view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
        """Contrastive loss for maximizng the consistency"""
        _, k = view1.size()
        # bn, k = view1.size()
        assert (view2.size(0) == _ and view2.size(1) == k)

        p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
        p_i_j = p_i_j.sum(dim=0)
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise
        # p_i_j = compute_joint(view1, view2)
        assert (p_i_j.size() == (k, k))

        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).detach()
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).detach()

        p_i_j[(p_i_j < EPS).data] = EPS
        p_j[(p_j < EPS).data] = EPS
        p_i[(p_i < EPS).data] = EPS

        loss = - p_i_j * (torch.log(p_i_j) \
                          - (lamb + 1) * torch.log(p_j) \
                          - (lamb + 1) * torch.log(p_i))

        # loss = loss.sum()
        loss = torch.sum(loss)

        return loss


def main():

    if settings.DATASET == "MIRFlickr":
        dataloader, data_train = get_loader_flickr_CLIP(settings.BATCH_SIZE)
        # dataloader, data_train = get_loader_flickr_fea(settings.BATCH_SIZE)
        train_dataset = dataloader['train']
        test_dataset = dataloader['query']
        database_dataset = dataloader['database']
    if settings.DATASET == "NUSWIDE":
        dataloader, data_train, dataloader2 = get_loader_nuswide_CLIP(settings.BATCH_SIZE)
        train_dataset = dataloader['train']
        test_dataset = dataloader['query']
        database_dataset = dataloader['database']
    if settings.DATASET == "MSCOCO":
        dataloader, data_train = get_loader_coco_CLIP(settings.BATCH_SIZE)
        train_dataset = dataloader['train']
        test_dataset = dataloader['query']
        database_dataset = dataloader['database']
    train_loader = train_dataset

    test_loader = test_dataset

    database_loader = database_dataset
    n_features = 4
    n_actions = 3 ** n_features
    # RL = DeepQNetwork(n_actions, n_features,
    #                   learning_rate=0.01,
    #                   reward_decay=0.9,
    #                   e_greedy=0.9,
    #                   replace_target_iter=100,
    #                   memory_size=500,
    #                   output_graph=True,
    #                   e_greedy_increment=0.1
    #                   )

    # RL = SarsaLambdaTable(actions=list(range(n_actions)))
    RL = QLearningTable(actions=list(range(n_actions)))
    observation = torch.ones(1, n_features)
    # l5 = 0.9
    RL.epsilon = 0.9
    max_e = 500
    max_status = 1.88
    max_value = 0
    l5 = 1
    l6 = 0.9
    l2 = 1
    l3 = 1
    l4 = 1
    l7 = 0.6
    q_table = pd.DataFrame(columns=list(range(1)), dtype=np.float64)

    for i in range(10000):
        step = 0
        observation = torch.ones(1, n_features)#*10**5
        # observation[0, :] = 0.3

        if i > 1:
            max_value1 = RL.q_table.max(axis=1)

            # 找到最大值所在的行
            observation = max_value1.idxmax()
            # 将字符串转换为Python列表

            observation = re.sub(r'[^0-9,\[\]\s.]', '', observation)
            list_data = ast.literal_eval(observation)

            # 将Python列表转换为PyTorch张量
            observation = torch.tensor(list_data, dtype=torch.float32)

        #observation[0, 0] = 0.3

        # RL.epsilon = 0.9
        # action = RL.choose_action(str(observation))

        # initial all zero eligibility trace
        # RL.eligibility_trace *= 0
        e_step = 0
        # action = RL.choose_action(observation)
        # print(action)
        a = 10 ** (-6)
        b = -0.1
        VAL = 0
        print(observation)
        while True:

            # l1, l2, l3, l4, l5, l6, l7 = observation[0, :]
            # l1, l2, l3, l4, l6, l7 = 10, 10, 1000, 0.1, 0.1, 1000
            # l1, l2, l3, l4, l6, l7 = 10, 10, 1000, 0.1, 0.001, 0.1
            # RL.epsilon = 0.9

            action = RL.choose_action(str(observation))


            observation_ = observation.clone()
            action1 = action
            bits = 10
            for j in range(n_features):
                if action1 % 3 == 0:
                    if observation_[0, j] / bits > 0.00001:
                        observation_[0, j] = float(observation[0, j]) / bits
                    else:
                        observation_[0, j] = 0
            #
                elif action1 % 3 == 1 and observation_[0, j] * bits < 100000:
                    observation_[0, j] = observation[0, j] * bits
                if observation_[0, j] > 0.9:
                    observation_[0, j] = int(observation_[0, j])
                action1 = action1 // 3
            setpj = 0
            #
            # for j in range(3):
            #      if action1 % 3 == 0:
            #          if observation_[0, setpj+j] - 0.1 > -0.1:
            #              observation_[0, setpj+j] = observation[0, setpj+j] - 0.1
            #          else:
            #              observation_[0, setpj+j] = 0
            #
            #      elif action1 % 3 == 1 and observation_[0, setpj+j] + 0.1 <= 1.1:
            #          observation_[0, setpj+j] = observation[0, setpj+j] + 0.1
            #      action1 = action1 // 3



            l1, l2, l3, l4, l5, l6, l7, l8 = 1.000000, 1.000000, 0.300000, 0.400000, 0.001000, 0.400000, 1.000000, 0.700000#0.000010, 1.000000, 0.100000, 0.500000, 0.000100, 0.400000, 1.000000, 0.700000
            l1, l2, l5, l7 = observation_[0, :]#l1, l2, l5, l7,

            A = 0
            # # l1, l2, l3, l4, l5 = 10.000000, 0.010000, 0.000100, 1000.000000, 0.010000
            # # a = a * 10
            # b = b + 0.1
            # l8 = b
            # if b == 0.6:
            #     continue
            state = str(observation_)

            status = False

            chae = 0.01
            if state not in q_table.index:
            # if True:
                l1, l2, l3, l4, l5, l6, l7, l8 = float(l1), float(l2), float(l3), float(l4), float(l5), float(l6), float(l7), float(l8)
                sess = Session(train_loader, test_loader, database_loader, train_dataset,
                               test_dataset, database_dataset, data_train, l5, l5, l7)
                B = 0
                Bmax = 0
                currmax = 0
                for epoch in range(settings.NUM_EPOCH):
                    # train the Model
                    sess.train(epoch, l1, l2, l3, l4, l5, l6, l7, l8)
                    # eval the Model
                    if ((epoch + 1) % settings.EVAL_INTERVAL == 0) and (epoch < 15 or epoch > 120):
                        B = sess.eval(l1, l2, l3, l4, l5, l6, l7, l8)
                        if max_value < B:
                            max_value = B
                        if B >= Bmax:
                            Bmax = B

                        # # sess.save_checkpoints(step=epoch + 1)
                        if B > A:
                            A = B
                        #     if B >= max_status:
                        #         max_status = B
                                # status = True
                        # else:
                        #     break
                        # print(A)
                        if B < currmax:
                            break
                        else:
                            currmax = B
                        if (B < 1.7):
                            break

                # append new state to q table
                q_table = q_table.append(
                    pd.Series(
                        Bmax,
                        index=q_table.columns,
                        name=state,
                    )
                )
            else:
                #step = step - 1
                A = 0
                B = q_table.loc[str(observation_), 0]
                # chae = 0.01
                if B > A:
                    A = B

                    # if B >= max_status:
                    #     max_status = B
                    #     status = True
                # print(A)
                # if B < max_status + chae:
                #     break
                print('^^^^^^exit:%s,B:%.6f' % (str(observation_),  B))


            # A = sess.eval(l1, l2, l3, l4, l5, l6, l7) - 1.8
            # A = A - max_status  # 1.8
            # reward = A * 100
            A = A - max_status

            if A < 0:
                reward = -2 ** ((-A) * 5)
            else:
                reward = 2 ** (A * 200)
            # reward = -2 ** ((-A) * 20)
            # if A < -chae:
            #     reward = -2 ** ((-A) * 20)
            # else:
            #     if status:
            #         reward = 100  # 2 ** (A * 1000)
            #     else:
            #         reward = -2 ** ((-A) * 20)#2 ** ((A+chae)*500)
            if reward < -10:
                reward = -10
            # if e_step > max_e:
            #     # observation_ = 'terminal'
            #     break
            # elif A > chae:
            #     reward = 2 ** (A * 500)
            # else:
            #     reward = 1
            # if e_step > max_e:
            #     # observation_ = 'terminal'
            #     break

            # if e_step > max_e:
            #     if A < 0:
            #         reward = -2 ** ((-A) * 15)
            #     else:
            #         reward = 2 ** ((A) * 50)
            # else:
            #     reward = A*10

            # print(reward)
            # print(i)

            # action_ = RL.choose_action(str(observation_))

            print('%.6f,%.6f,%.6f,%.6f,max:%.6f,value:%.6f' % (action, reward, i, RL.epsilon, max_status, max_value))

            # print(str(observation))
            # print(str(observation_))
            # RL.learn(str(observation), action, reward, str(observation_), action_)

            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            # action = action_
            e_step = e_step + 1
            if B > VAL:
                VAL = B
            else:
                break
            # if B < 1.3:
            # # # if (l1,l2, l3, l4,l5,l6, l7) == (0, 0, 0, 0,0,0,0):
            #     break
            # RL.store_transition(observation, action, reward, observation_)
            # if (i > 50) :#and (step % 5 == 0):
            #     # print(No)
            #     # RL.learn()
            #     RL.epsilon = 0.9
            # else:
            #     RL.epsilon = 0.01 * i + 0.4
            if step > max_e:  # or A > 0
                # print(observation)
                break
            step += 1


#
if __name__ == '__main__':
    main()



def param_list(log, config):
    log.info('>>> Configs List <<<')
    log.info('--- Dadaset:{}'.format(config.DATASET))
    log.info('--- SEED:{}'.format(config.SEED))
    log.info('--- Bit:{}'.format(config.HASH_BIT))
    log.info('--- Batch:{}'.format(config.BATCH_SIZE))
    log.info('--- Lr_IMG:{}'.format(config.LR_IMG))
    log.info('--- Lr_TXT:{}'.format(config.LR_TXT))
    log.info('--- LR_MyNet:{}'.format(config.LR_MyNet))

    log.info('--- lambda1:{}'.format(config.lambda1))
    log.info('--- lambda2:{}'.format(config.lambda2))
    log.info('--- lambda3:{}'.format(config.lambda3))
    log.info('--- lambda4:{}'.format(config.lambda4))
    log.info('--- lambda5:{}'.format(config.lambda5))
    log.info('--- gamma:{}'.format(config.gamma))
    log.info('--- beta:{}'.format(config.beta))
    log.info('--- sigma:{}'.format(config.sigma))
    log.info('--- alpha:{}'.format(config.alpha))
    log.info('--- K:{}'.format(config.K))
    log.info('--- a1:{}'.format(config.a1))
    log.info('--- a2:{}'.format(config.a2))
    log.info('--- b1:{}'.format(config.b1))
    log.info('--- b2:{}'.format(config.b2))
    log.info('--- b3:{}'.format(config.b3))
    log.info('--- thres_T:{}'.format(config.thres_T))
    log.info('--- thres_I:{}'.format(config.thres_I))
    log.info('--- K_T:{}'.format(config.K_T))
    log.info('--- K_I:{}'.format(config.K_I))

def objective(trial):
    if settings.DATASET == "MIRFlickr":
        dataloader, data_train = get_loader_flickr_CLIP(settings.BATCH_SIZE)
        # dataloader, data_train = get_loader_flickr_fea(settings.BATCH_SIZE)
        train_dataset = dataloader['train']
        test_dataset = dataloader['query']
        database_dataset = dataloader['database']
    if settings.DATASET == "NUSWIDE":
        # dataloader, data_train = get_loader_nus(settings.BATCH_SIZE)
        dataloader, data_train = get_loader_nuswide_CLIP(settings.BATCH_SIZE)
        train_dataset = dataloader['train']
        test_dataset = dataloader['query']
        database_dataset = dataloader['database']

    if settings.DATASET == "MSCOCO":
        # dataloader, data_train = get_loader_nus(settings.BATCH_SIZE)
        dataloader, data_train = get_loader_coco_CLIP(settings.BATCH_SIZE)
        train_dataset = dataloader['train']
        test_dataset = dataloader['query']
        database_dataset = dataloader['database']

        # train_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform)
        # test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
        # database_dataset = datasets.MIRFlickr(train=False, database=True, transform=datasets.mir_test_transform)

    # if settings.DATASET == "NUSWIDE":
    #     train_dataset = datasets.NUSWIDE(train=True, transform=datasets.nus_train_transform)
    #     test_dataset = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
    #     database_dataset = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)

    # Data Loader (Input Pipeline)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=settings.BATCH_SIZE,
    #                                            shuffle=True,
    #                                            num_workers=settings.NUM_WORKERS,
    #                                            drop_last=True)
    #
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                           batch_size=settings.BATCH_SIZE,
    #                                           shuffle=False,
    #                                           num_workers=settings.NUM_WORKERS)
    #
    # database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
    #                                               batch_size=settings.BATCH_SIZE,
    #                                               shuffle=False,
    #                                               num_workers=settings.NUM_WORKERS)
    train_loader = train_dataset

    test_loader = test_dataset

    database_loader = database_dataset
    config = argparse.Namespace(
        lambda1=trial.suggest_categorical('lambda1', [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]),
        lambda2=trial.suggest_categorical('lambda2', [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]),
        lambda5=trial.suggest_categorical('lambda5', [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]),
        lambda7=trial.suggest_categorical('lambda7', [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]),
        lambda3=trial.suggest_categorical('lambda3', [0.1, 0.2]),
        lambda4=trial.suggest_categorical('lambda4', [0.7, 0.8, 0.9]),
        lambda6=trial.suggest_categorical('lambda6', [0.6, 0.7, 0.8, 0.9]),
        lambda8=trial.suggest_categorical('lambda8', [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]),
    )

    # logName = config.DATASET + '_' + str(config.HASH_BIT) + '_optuna'
    # logger = settings.logger
    # log = logger(logName)
    # param_list(log, config)

    return train_and_evaluate(config, train_loader, test_loader, database_loader, train_dataset,
                   test_dataset, database_dataset, data_train)

def train_and_evaluate(config, train_loader, test_loader, database_loader, train_dataset,
                   test_dataset, database_dataset, data_train):
    config.lambda2 = 1
    config.lambda5 = 1
    import os

    sess = Session(train_loader, test_loader, database_loader, train_dataset,
                   test_dataset, database_dataset, data_train, config.lambda5, config.lambda5, config.lambda7)
    A = 0
    B = 0
    Bmax = 0
    # start_time = time.time()
    for epoch in range(settings.NUM_EPOCH):
        # train the Model
        sess.train(epoch, config.lambda1, config.lambda2, config.lambda3, config.lambda4, config.lambda5, config.lambda6, config.lambda7, config.lambda8)
        # eval the Model
        if ((epoch + 1) % settings.EVAL_INTERVAL == 0):
            B = sess.eval(config.lambda1, config.lambda2, config.lambda3, config.lambda4, config.lambda5, config.lambda6, config.lambda7, config.lambda8)
            if B > Bmax:
                Bmax = B

    return Bmax

# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='Ours')
#     parser.add_argument('--n_trials', default=5000, type=int, help='number of trials for hyperparameter optimization')
#     args = parser.parse_args()
#
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=args.n_trials)
#
#     print("Best hyperparameters: ", study.best_params)