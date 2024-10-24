import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import math
import numpy as np
import scipy
import scipy.spatial
from scipy.spatial.distance import cdist
from tools import build_G_from_S, generate_robust_S, edge_list
import settingsnuspre as settings

def compress_wiki(train_loader, test_loader, modeli, modelt, train_dataset, test_dataset, classes=10):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, target, _) in enumerate(train_loader):
        var_data_I = Variable(data_I.cuda())
        _,_,code_I = modeli(var_data_I.to(torch.float))
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        re_L.extend(target)
        
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _,_,code_T = modelt(var_data_T.to(torch.float))
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, target, _) in enumerate(test_loader):
        var_data_I = Variable(data_I.cuda())
        _,_,code_I = modeli(var_data_I.to(torch.float))
        # _,_,code_I = model_IT(code_I)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        qu_L.extend(target)
        
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _,_,code_T = modelt(var_data_T.to(torch.float))
        # _,_,code_T = model_TI(code_T)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = np.eye(classes)[np.array(re_L)]

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = np.eye(classes)[np.array(qu_L)]
    
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def compressHY(train_loader, test_loader, model_I, model_T, train_dataset, test_dataset):
    model_I.cuda().eval()
    model_T.cuda().eval()
    re_BI = torch.Tensor()  # list([])
    re_BT = torch.Tensor()  # list([])
    re_L = torch.Tensor()  # list([])
    for _, (data_I, data_T, data_L, _) in enumerate(train_loader):
        var_data_I = Variable(data_I.cuda())
        F_I = Variable(torch.FloatTensor(data_I.numpy()).cuda())
        code_I = model_I(var_data_I.to(torch.float), F.normalize(F_I).mm(F.normalize(F_I).t()).cuda())
        code_I = torch.sign(code_I).cuda()
        # re_BI.extend(code_I.cpu().data.numpy())
        re_BI = torch.cat((re_BI.cuda(), code_I.cuda()), dim=0)

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        code_T = model_T(var_data_T.to(torch.float), F.normalize(var_data_T).mm(F.normalize(var_data_T).t()))
        code_T = torch.sign(code_T).cuda()
        # re_BT.extend(code_T.cpu().data.numpy())
        # re_L.extend(data_L.cpu().data.numpy())
        re_BT = torch.cat((re_BT.cuda(), code_T.cuda()), dim=0)
        re_L = torch.cat((re_L.cuda(), data_L.cuda()), dim=0)

    qu_BI = torch.Tensor()  # list([])
    qu_BT = torch.Tensor()  # list([])
    qu_L = torch.Tensor()  # list([])
    for _, (data_I, data_T, data_L, _) in enumerate(test_loader):
        var_data_I = Variable(data_I.cuda())
        F_I = Variable(torch.FloatTensor(data_I.numpy()).cuda())
        code_I = model_I(var_data_I.to(torch.float), F.normalize(F_I).mm(F.normalize(F_I).t()))
        # _,_,code_I = model_IT(code_I)
        code_I = torch.sign(code_I).cuda()
        # qu_BI.extend(code_I.cpu().data.numpy())
        qu_BI = torch.cat((qu_BI.cuda(), code_I.cuda()), dim=0)

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        code_T = model_T(var_data_T.to(torch.float), F.normalize(var_data_T).mm(F.normalize(var_data_T).t()))
        # _,_,code_T = model_TI(code_T)
        code_T = torch.sign(code_T).cuda()
        # qu_BT.extend(code_T.cpu().data.numpy())
        # qu_L.extend(data_L.cpu().data.numpy())
        qu_BT = torch.cat((qu_BT.cuda(), code_T.cuda()), dim=0)
        qu_L = torch.cat((qu_L.cuda(), data_L.cuda()), dim=0)

    # re_BI = np.array(re_BI)
    # re_BT = np.array(re_BT)
    # re_L = np.array(re_L)
    #
    # qu_BI = np.array(qu_BI)
    # qu_BT = np.array(qu_BT)
    # qu_L = np.array(qu_L)
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

def compress(train_loader, test_loader, model_I, model_T, train_dataset, test_dataset, l3, l4, l7, l6, l8):
    l7 = 0.1
    l2 = 5
    model_I.cuda().eval()
    model_T.cuda().eval()
    re_BI = torch.Tensor()#list([])
    re_BT = torch.Tensor()#list([])
    re_L = torch.Tensor()#list([])
    for _, (data_I, data_T, data_L, _) in enumerate(train_loader):

        F_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        F_I = Variable(torch.FloatTensor(data_I.numpy()).cuda())
        SI = F.normalize(F_I).mm(F.normalize(F_I).t()).cuda()
        ST = F.normalize(F_T).mm(F.normalize(F_T).t()).cuda()
        # 计算各元素之间的欧式距离矩阵
        SDI = torch.cdist(F_I, F_I, p=2).cuda()
        SDT = torch.cdist(F_T, F_T, p=2).cuda()
        II = SDI.max(dim=1, keepdim=True)[0]
        IT = SDT.max(dim=1, keepdim=True)[0]

        SDI = (torch.exp(SDI / II)) ** int(2 * l8 + 0.1)
        SDT = (torch.exp(SDT / IT)) ** int(2 * l8 + 0.1)
        SI = SI / SDI
        ST = ST / SDT

        # SDI = (SDI / II) ** int(20 * l8 + 2)
        # SDT = (SDT / IT) ** int(20 * l8 + 2)
        # SI = SI / torch.exp(SDI)
        # ST = ST / torch.exp(SDT)
        GV = l3 * ST + (1 - l3) * SI
        GVT = torch.sign(GV + l7)
        G = l4 * GV + (1 - l4) * GVT.mm(GVT) / settings.BATCH_SIZE
        GT = build_G_from_S(G, l2, l6)
        GRT = torch.LongTensor(edge_list(GT)).cuda()

        var_data_I = Variable(data_I.cuda())
        code_I, _ = model_I(var_data_I.to(torch.float), GRT)
        code_I = torch.sign(code_I).cuda()
        re_BI = torch.cat((re_BI.cuda(), code_I.cuda()), dim=0)
        
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        code_T, _ = model_T(var_data_T.to(torch.float), GRT)
        code_T = torch.sign(code_T).cuda()
        re_BT = torch.cat((re_BT.cuda(), code_T.cuda()), dim=0)
        re_L = torch.cat((re_L.cuda(), data_L.cuda()), dim=0)

    qu_BI = torch.Tensor()#list([])
    qu_BT = torch.Tensor()#list([])
    qu_L = torch.Tensor()#list([])
    for _, (data_I, data_T, data_L, _) in enumerate(test_loader):

        F_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        F_I = Variable(torch.FloatTensor(data_I.numpy()).cuda())
        SI = F.normalize(F_I).mm(F.normalize(F_I).t()).cuda()
        ST = F.normalize(F_T).mm(F.normalize(F_T).t()).cuda()
        # 计算各元素之间的欧式距离矩阵
        SDI = torch.cdist(F_I, F_I, p=2).cuda()
        SDT = torch.cdist(F_T, F_T, p=2).cuda()
        II = SDI.max(dim=1, keepdim=True)[0]
        IT = SDT.max(dim=1, keepdim=True)[0]

        SDI = (torch.exp(SDI / II)) ** int(2 * l8 + 0.1)
        SDT = (torch.exp(SDT / IT)) ** int(2 * l8 + 0.1)
        SI = SI / SDI
        ST = ST / SDT
        GV = l3 * ST + (1 - l3) * SI

        GVT = torch.sign(GV + l7)
        G = l4 * GV + (1 - l4) * GVT.mm(GVT) / settings.BATCH_SIZE
        GT = build_G_from_S(G, l2, l6)
        GRT = torch.LongTensor(edge_list(GT)).cuda()

        var_data_I = Variable(data_I.cuda())
        code_I, _ = model_I(var_data_I.to(torch.float), GRT)
        # _,_,code_I = model_IT(code_I)
        code_I = torch.sign(code_I).cuda()
        # qu_BI.extend(code_I.cpu().data.numpy())
        qu_BI = torch.cat((qu_BI.cuda(), code_I.cuda()), dim=0)
        
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        code_T, _ = model_T(var_data_T.to(torch.float), GRT)
        # _,_,code_T = model_TI(code_T)
        code_T = torch.sign(code_T).cuda()
        # qu_BT.extend(code_T.cpu().data.numpy())
        # qu_L.extend(data_L.cpu().data.numpy())
        qu_BT = torch.cat((qu_BT.cuda(), code_T.cuda()), dim=0)
        qu_L = torch.cat((qu_L.cuda(), data_L.cuda()), dim=0)

    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def compress_nus(train_loader, test_loader, model_I, model_T, train_dataset, test_dataset, l3, l4, l7, l6, l8):
    l7 = 0.1
    l2 = 5
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, data_L, _) in enumerate(train_loader):
        F_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        F_I = Variable(torch.FloatTensor(data_I.numpy()).cuda())
        SI = F.normalize(F_I).mm(F.normalize(F_I).t()).cuda()
        ST = F.normalize(F_T).mm(F.normalize(F_T).t()).cuda()
        # 计算各元素之间的欧式距离矩阵
        SDI = torch.cdist(F_I, F_I, p=2).cuda()
        SDT = torch.cdist(F_T, F_T, p=2).cuda()
        II = SDI.max(dim=1, keepdim=True)[0]
        IT = SDT.max(dim=1, keepdim=True)[0]

        SDI = (torch.exp(SDI / II)) ** int(2 * l8 + 0.1)
        SDT = (torch.exp(SDT / IT)) ** int(2 * l8 + 0.1)
        SI = SI / SDI
        ST = ST / SDT
        GV = l3 * ST + (1 - l3) * SI

        GVT = torch.sign(GV + l7)
        G = l4 * GV + (1 - l4) * GVT.mm(GVT) / settings.BATCH_SIZE
        GT = build_G_from_S(G, l2, l6)
        GRT = torch.LongTensor(edge_list(GT)).cuda()

        var_data_I = Variable(data_I.cuda())
        code_I, _ = model_I(var_data_I.to(torch.float), GRT)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        code_T, _ = model_T(var_data_T.to(torch.float), GRT)
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())
        re_L.extend(data_L.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, data_L, _) in enumerate(test_loader):
        var_data_I = Variable(data_I.cuda())
        F_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        F_I = Variable(torch.FloatTensor(data_I.numpy()).cuda())
        SI = F.normalize(F_I).mm(F.normalize(F_I).t()).cuda()
        ST = F.normalize(F_T).mm(F.normalize(F_T).t()).cuda()
        # 计算各元素之间的欧式距离矩阵
        SDI = torch.cdist(F_I, F_I, p=2).cuda()
        SDT = torch.cdist(F_T, F_T, p=2).cuda()
        II = SDI.max(dim=1, keepdim=True)[0]
        IT = SDT.max(dim=1, keepdim=True)[0]

        SDI = (torch.exp(SDI / II)) ** int(2 * l8 + 0.1)
        SDT = (torch.exp(SDT / IT)) ** int(2 * l8 + 0.1)
        SI = SI / SDI
        ST = ST / SDT
        GV = l3 * ST + (1 - l3) * SI

        GVT = torch.sign(GV + l7)
        G = l4 * GV + (1 - l4) * GVT.mm(GVT) / settings.BATCH_SIZE
        GT = build_G_from_S(G, l2, l6)
        GRT = torch.LongTensor(edge_list(GT)).cuda()

        var_data_I = Variable(data_I.cuda())
        code_I, _ = model_I(var_data_I.to(torch.float), GRT)
        # _,_,code_I = model_IT(code_I)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        code_T, _ = model_T(var_data_T.to(torch.float), GRT)
        # _,_,code_T = model_TI(code_T)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())
        qu_L.extend(data_L.cpu().data.numpy())


    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = np.array(re_L)
    # re_L = train_dataset.train_labels

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = np.array(qu_L)
    # qu_L = test_dataset.train_labels
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1] # max inner product value
    distH = 0.5 * (leng - B1.mm(B2.t()))
    distH = distH[0].cpu().data.numpy()
    return distH


def calculate_map(qu_B, re_B, qu_L, re_L):
    """
       :param qu_B: {-1,+1}^{mxq} query bits
       :param re_B: {-1,+1}^{nxq} retrieval bits
       :param qu_L: {0,1}^{mxl} query label
       :param re_L: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        tsum = np.int64(np.sum(gnd))
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map

def fx_calc_map_multilabel_k(retrieval, retrieval_labels, query, query_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(query, retrieval, metric)
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i].reshape(-1)

        tmp_label = (np.dot(retrieval_labels[order], query_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)
def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        # gnd = (np.dot(qu_L[iter, :].cpu().data.numpy(), re_L.cpu().data.numpy().transpose()) > 0).astype(np.float32)
        A = (qu_L[iter, :].reshape(1, qu_L.size(1))).mm(re_L.t()).cpu().data.numpy()
        gnd = (A[0] > 0).astype(np.float32)
        # gnd = (((qu_L[iter].reshape(1, qu_L.size(1))).mm(re_L.t())).cpu().data.numpy() > 0).astype(np.float32)[0]
        hamm = calculate_hamming(qu_B[iter, :].reshape(1, qu_B.size(1)), re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.int64(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def calculate_hamming_nus(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1] # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH
def calculate_top_map_nus(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float64)
        hamm = calculate_hamming_nus(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.int64(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def p_topK(qB, rB, query_label, retrieval_label, K):
    num_query = query_label.shape[0]
    p = [0] * len(K)
    # query_label = torch.Tensor(query_label)
    # retrieval_label = torch.Tensor(retrieval_label)
    # qB = torch.Tensor(qB)
    # rB = torch.Tensor(rB)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        # hamm = torch.Tensor(hamm)
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm).indices[:int(total)]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p


# 画 P-R 曲线
def pr_curve(qF, rF, qL, rL, what=0, topK=-1):
    """Input:
    what: {0: cosine 距离, 1: Hamming 距离}
    topK: 即 mAP 中的 position threshold，只取前 k 个检索结果，默认 `-1` 是全部，见 [3]
    """
    n_query = qF.shape[0]
    if topK == -1 or topK > rF.shape[0]:  # top-K 之 K 的上限
        topK = rF.shape[0]

    # Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    Gnd = (qL).mm(rL.t()).cpu().data.numpy()
    Gnd = (Gnd > 0).astype(np.float32)
    if what == 0:
        Rank = np.argsort(cdist(qF.cpu().data.numpy(), rF.cpu().data.numpy(), 'cosine'))
    else:
        Rank = np.argsort(cdist(qF.cpu().data.numpy(), rF.cpu().data.numpy(), 'hamming'))

    P, R = [], []
    for i in range(1, 10):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = np.zeros(n_query)  # 各 query sample 的 Precision@R
        r = np.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)  # 整个被检索数据库中的相关样本数
            k = int(gnd_all*0.1*i)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all

        P.append(np.mean(p))
        R.append(np.mean(r))
        print(np.mean(p))
        print(np.mean(r))
        print(P)
        print(R)
    print(P)
    print(R)

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda:0', temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())
    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        print(similarity_matrix.shape)
        print(self.batch_size)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
