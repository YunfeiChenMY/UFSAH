import random

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn import preprocessing
import pdb
import torch
import h5py
import scipy.io as sio
import pickle
import random
import os

torch.multiprocessing.set_sharing_strategy('file_system')

class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]

        label = self.labels[index]
        return img, text, label, index

    def __len__(self):
        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count


def get_loader_flickr(batch_size):
    path = 'E:/Hash/dataset/CIRH/MIRFlickr/'

    # x: images   y:tags   L:labels
    train_set = sio.loadmat(path + 'mir_train.mat')
    train_L = np.array(train_set['L_tr'], dtype=np.float)
    train_x = np.array(train_set['I_tr'], dtype=np.float)
    train_y = np.array(train_set['T_tr'], dtype=np.float)

    test_set = sio.loadmat(path + 'mir_query.mat')
    query_L = np.array(test_set['L_te'], dtype=np.float)
    query_x = np.array(test_set['I_te'], dtype=np.float)
    query_y = np.array(test_set['T_te'], dtype=np.float)

    db_set = sio.loadmat(path + 'mir_database.mat')
    retrieval_L = np.array(db_set['L_db'], dtype=np.float)
    retrieval_x = np.array(db_set['I_db'], dtype=np.float)
    retrieval_y = np.array(db_set['T_db'], dtype=np.float)


    imgs = {'train': train_x, 'query': query_x, 'database': retrieval_x}
    texts = {'train': train_y, 'query': query_y, 'database': retrieval_y}
    labels = {'train': train_L, 'query': query_L, 'database': retrieval_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['query', 'train', 'database']}

    return dataloader, (train_x, train_y, train_L)

def get_loader_flickr_CLIP2(batch_size):
    path = 'E:/Hash/dataset/CIRH/MIRFlickr/'

    # x: images   y:tags   L:labels
    train_set = sio.loadmat(path + 'mir_train.mat')
    train_L = np.array(train_set['L_tr'], dtype=np.float)
    train_x = np.array(train_set['I_tr'], dtype=np.float)
    train_y = np.array(train_set['T_tr'], dtype=np.float)

    test_set = sio.loadmat(path + 'mir_query.mat')
    query_L = np.array(test_set['L_te'], dtype=np.float)
    query_x = np.array(test_set['I_te'], dtype=np.float)
    query_y = np.array(test_set['T_te'], dtype=np.float)

    db_set = sio.loadmat(path + 'mir_database.mat')
    retrieval_L = np.array(db_set['L_db'], dtype=np.float)
    retrieval_x = np.array(db_set['I_db'], dtype=np.float)
    retrieval_y = np.array(db_set['T_db'], dtype=np.float)


    imgs = {'train': train_x, 'query': query_x, 'database': retrieval_x}
    texts = {'train': train_y, 'query': query_y, 'database': retrieval_y}
    labels = {'train': train_L, 'query': query_L, 'database': retrieval_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['query', 'train', 'database']}
    train_loc = 'E:/Hash/dataset/CLIP/CLIP/mirflickr/train.pkl'

    # x: images   y:tags   L:labels
    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_L1 = np.array(data['label'], dtype=np.float)
        train_x1 = np.array(data['image'], dtype=np.float)
        train_y1 = np.array(data['text'], dtype=np.float)

    return dataloader, (train_x1, train_y1, train_L1)
def get_loader_flickr_CLIP(batch_size):
    path = 'E:/Hash/dataset/CIRH/MIRFlickr/'
    dataset = 'mirflickr'
    train_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/train.pkl'
    query_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/query.pkl'
    retrieval_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/retrieval.pkl'

    # x: images   y:tags   L:labels
    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_L = np.array(data['label'], dtype=np.float64)
        train_x = np.array(data['image'], dtype=np.float64)
        train_y = np.array(data['text'], dtype=np.float64)
    # train_set = sio.loadmat(train_loc)


    with open(query_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        query_L = np.array(data['label'], dtype=np.float64)
        query_x = np.array(data['image'], dtype=np.float64)
        query_y = np.array(data['text'], dtype=np.float64)

    # test_set = sio.loadmat(query_loc)
    # query_L = np.array(test_set['label'], dtype=np.float)
    # query_x = np.array(test_set['image'], dtype=np.float)
    # query_y = np.array(test_set['text'], dtype=np.float)

    with open(retrieval_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        retrieval_L = np.array(data['label'], dtype=np.float64)
        retrieval_x = np.array(data['image'], dtype=np.float64)
        retrieval_y = np.array(data['text'], dtype=np.float64)
    # db_set = sio.loadmat(retrieval_loc)
    # retrieval_L = np.array(db_set['label'], dtype=np.float)
    # retrieval_x = np.array(db_set['image'], dtype=np.float)
    # retrieval_y = np.array(db_set['text'], dtype=np.float)


    imgs = {'train': train_x[:4992], 'query': query_x, 'database': retrieval_x}
    texts = {'train': train_y[:4992], 'query': query_y, 'database': retrieval_y}
    labels = {'train': train_L[:4992], 'query': query_L, 'database': retrieval_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['query', 'train', 'database']}

    return dataloader, (train_x[:4992], train_y[:4992], train_L[:4992])

def get_loader_flickr_fea(batch_size):
    root = 'E:/Hash/dataset/UCCH/MIRFLICKR25K/'
    data_img = sio.loadmat(os.path.join(root, 'mirflickr25k-iall-vgg.mat'))['XAll']
    data_txt = sio.loadmat(os.path.join(root, 'mirflickr25k-yall.mat'))['YAll']
    labels = sio.loadmat(os.path.join(root, 'mirflickr25k-lall.mat'))['LAll']
    test_size = 2000
    train_size = 5000

    # x: images   y:tags   L:labels
    # with open(train_loc, 'rb') as f_pkl:
    #     data = pickle.load(f_pkl)
    #     train_L = np.array(data['label'], dtype=np.float)
    #     train_x = np.array(data['image'], dtype=np.float)
    #     train_y = np.array(data['text'], dtype=np.float)
    # train_set = sio.loadmat(train_loc)
    train_x, train_y, train_L = data_img[0: train_size], data_txt[0: train_size], labels[0: train_size]



    # with open(query_loc, 'rb') as f_pkl:
    #     data = pickle.load(f_pkl)
    #     query_L = np.array(data['label'], dtype=np.float)
    #     query_x = np.array(data['image'], dtype=np.float)
    #     query_y = np.array(data['text'], dtype=np.float)
    query_x, query_y, query_L = data_img[-test_size::], data_txt[-test_size::], labels[-test_size::]

    # test_set = sio.loadmat(query_loc)
    # query_L = np.array(test_set['label'], dtype=np.float)
    # query_x = np.array(test_set['image'], dtype=np.float)
    # query_y = np.array(test_set['text'], dtype=np.float)

    # with open(retrieval_loc, 'rb') as f_pkl:
    #     data = pickle.load(f_pkl)
    #     retrieval_L = np.array(data['label'], dtype=np.float)
    #     retrieval_x = np.array(data['image'], dtype=np.float)
    #     retrieval_y = np.array(data['text'], dtype=np.float)
    retrieval_x, retrieval_y, retrieval_L = data_img[0: -test_size], data_txt[0: -test_size], labels[0: -test_size]
    # db_set = sio.loadmat(retrieval_loc)
    # retrieval_L = np.array(db_set['label'], dtype=np.float)
    # retrieval_x = np.array(db_set['image'], dtype=np.float)
    # retrieval_y = np.array(db_set['text'], dtype=np.float)


    imgs = {'train': train_x[:4992], 'query': query_x, 'database': retrieval_x}
    texts = {'train': train_y[:4992], 'query': query_y, 'database': retrieval_y}
    labels = {'train': train_L[:4992], 'query': query_L, 'database': retrieval_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['query', 'train', 'database']}

    return dataloader, (train_x[:4992], train_y[:4992], train_L[:4992])
# def MIRFlickr25K_fea(partition):
#     root = 'E:/Hash/dataset/UCCH/MIRFLICKR25K/'
#     data_img = sio.loadmat(os.path.join(root, 'mirflickr25k-iall-vgg.mat'))['XAll']
#     data_txt = sio.loadmat(os.path.join(root, 'mirflickr25k-yall.mat'))['YAll']
#     labels = sio.loadmat(os.path.join(root, 'mirflickr25k-lall.mat'))['LAll']
#
#     test_size = 2000
#     train_size = 5000
#     if 'test' in partition.lower():
#         data_img, data_txt, labels = data_img[-test_size::], data_txt[-test_size::], labels[-test_size::]
#     elif 'train' in partition.lower():
#         data_img, data_txt, labels = data_img[0: train_size], data_txt[0: train_size], labels[0: train_size]
#     else:
#         data_img, data_txt, labels = data_img[0: -test_size], data_txt[0: -test_size], labels[0: -test_size]
#
#     return data_img, data_txt, labels

def get_loader_flickr_CLIP5000(batch_size):
    np.random.seed(1)
    path = 'E:/Hash/dataset/CIRH/MIRFlickr/'
    dataset = 'mirflickr'
    train_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/train.pkl'
    query_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/query.pkl'
    retrieval_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/retrieval.pkl'

    # x: images   y:tags   L:labels
    # with open(train_loc, 'rb') as f_pkl:
    #     data = pickle.load(f_pkl)
    #     train_L = np.array(data['label'], dtype=np.float)
    #     train_x = np.array(data['image'], dtype=np.float)
    #     train_y = np.array(data['text'], dtype=np.float)
    # # train_set = sio.loadmat(train_loc)
    #
    #
    # with open(query_loc, 'rb') as f_pkl:
    #     data = pickle.load(f_pkl)
    #     query_L = np.array(data['label'], dtype=np.float)
    #     query_x = np.array(data['image'], dtype=np.float)
    #     query_y = np.array(data['text'], dtype=np.float)

    # test_set = sio.loadmat(query_loc)
    # query_L = np.array(test_set['label'], dtype=np.float)
    # query_x = np.array(test_set['image'], dtype=np.float)
    # query_y = np.array(test_set['text'], dtype=np.float)

    with open(retrieval_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        retrieval_L = np.array(data['label'], dtype=np.float)
        retrieval_x = np.array(data['image'], dtype=np.float)
        retrieval_y = np.array(data['text'], dtype=np.float)
        index = np.random.choice(np.arange(retrieval_L.shape[0]), size=10000, replace=False)

        train_L = retrieval_L[index]
        train_x = retrieval_x[index]
        train_y = retrieval_y[index]

        index2= np.random.choice(np.arange(retrieval_L.shape[0]), size=5000, replace=False)
        query_L = retrieval_L[index2]
        query_x = retrieval_x[index2]
        query_y = retrieval_y[index2]
    # db_set = sio.loadmat(retrieval_loc)
    # retrieval_L = np.array(db_set['label'], dtype=np.float)
    # retrieval_x = np.array(db_set['image'], dtype=np.float)
    # retrieval_y = np.array(db_set['text'], dtype=np.float)


    imgs = {'train': train_x, 'query': query_x, 'database': retrieval_x}
    texts = {'train': train_y, 'query': query_y, 'database': retrieval_y}
    labels = {'train': train_L, 'query': query_L, 'database': retrieval_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['query', 'train', 'database']}

    return dataloader, (train_x, train_y, train_L)

def get_loader_nuswide_CLIP(batch_size):
    dataset = 'nus-wide'
    train_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/train.pkl'
    query_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/query.pkl'
    retrieval_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/retrieval.pkl'

    # x: images   y:tags   L:labels
    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_L = np.array(data['label'], dtype=np.float64)
        train_x = np.array(data['image'], dtype=np.float64)
        train_y = np.array(data['text'], dtype=np.float64)
    # train_set = sio.loadmat(train_loc)


    with open(query_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        query_L = np.array(data['label'], dtype=np.float64)
        query_x = np.array(data['image'], dtype=np.float64)
        query_y = np.array(data['text'], dtype=np.float64)

    # test_set = sio.loadmat(query_loc)
    # query_L = np.array(test_set['label'], dtype=np.float)
    # query_x = np.array(test_set['image'], dtype=np.float)
    # query_y = np.array(test_set['text'], dtype=np.float)

    with open(retrieval_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        retrieval_L = np.array(data['label'], dtype=np.float64)
        retrieval_x = np.array(data['image'], dtype=np.float64)
        retrieval_y = np.array(data['text'], dtype=np.float64)
    # db_set = sio.loadmat(retrieval_loc)
    # retrieval_L = np.array(db_set['label'], dtype=np.float)
    # retrieval_x = np.array(db_set['image'], dtype=np.float)
    # retrieval_y = np.array(db_set['text'], dtype=np.float)


    imgs = {'train': train_x[:4992], 'query': query_x, 'database': retrieval_x[:]}
    texts = {'train': train_y[:4992], 'query': query_y, 'database': retrieval_y[:]}
    labels = {'train': train_L[:4992], 'query': query_L, 'database': retrieval_L[:]}


    imgs2 = {'train': train_x[:4992], 'query': query_x, 'database': retrieval_x[:]}
    texts2 = {'train': train_y[:4992], 'query': query_y, 'database': retrieval_y[:]}
    labels2 = {'train': train_L[:4992], 'query': query_L, 'database': retrieval_L[:]}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    dataset2 = {x: CustomDataSet(images=imgs2[x], texts=texts2[x], labels=labels2[x])
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['query', 'train', 'database']}

    dataloader2 = {x: DataLoader(dataset2[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['query', 'train', 'database']}

    return dataloader, (train_x[:4992], train_y[:4992], train_L[:4992]), dataloader2

def get_loader_coco_CLIP(batch_size):
    path = 'E:/Hash/dataset/CIRH/MIRFlickr/'
    dataset = 'mscoco'
    train_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/train.pkl'
    query_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/query.pkl'
    retrieval_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/retrieval.pkl'

    # # x: images   y:tags   L:labels
    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_L = np.array(data['label'], dtype=np.float64)
        train_x = np.array(data['image'], dtype=np.float64)
        train_y = np.array(data['text'], dtype=np.float64)
    # train_set = sio.loadmat(train_loc)


    with open(query_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        query_L = np.array(data['label'], dtype=np.float64)
        query_x = np.array(data['image'], dtype=np.float64)
        query_y = np.array(data['text'], dtype=np.float64)

    # test_set = sio.loadmat(query_loc)
    # query_L = np.array(test_set['label'], dtype=np.float)
    # query_x = np.array(test_set['image'], dtype=np.float)
    # query_y = np.array(test_set['text'], dtype=np.float)

    with open(retrieval_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        retrieval_L = np.array(data['label'], dtype=np.float64)
        retrieval_x = np.array(data['image'], dtype=np.float64)
        retrieval_y = np.array(data['text'], dtype=np.float64)
    # db_set = sio.loadmat(retrieval_loc)
    # retrieval_L = np.array(db_set['label'], dtype=np.float)
    # retrieval_x = np.array(db_set['image'], dtype=np.float)
    # retrieval_y = np.array(db_set['text'], dtype=np.float)

    # (train_loc, 'rb') as f_pkl:
    #     data = pickle.load(f_pkl)
    #     train_L = np.array(data['label'], dtype=np.float64)
    #     train_x = np.array(data['image'], dtype=np.float64)
    #     train_y = np.array(data['text'],
    # with open dtype=np.float64)

    # train_x = np.concatenate((train_x, retrieval_x[-7800:]))
    # train_L = np.concatenate((train_L, retrieval_L[-7800:]))
    # train_y = np.concatenate((train_y, retrieval_y[-7800:]))
    # train_L = retrieval_L[11*10000:11*10000+896]
    # train_x = retrieval_x[11*10000:11*10000+896]
    # train_y = retrieval_y[11*10000:11*10000+896]
    # for i in range(11):
    #     train_x = np.concatenate((train_x, retrieval_x[i*10000:i*10000+896]))
    #     train_L = np.concatenate((train_L, retrieval_L[i*10000:i*10000+896]))
    #     train_y = np.concatenate((train_y, retrieval_y[i*10000:i*10000+896]))

    imgs = {'train': np.concatenate((train_x, retrieval_x[:120])), 'query': query_x, 'database': retrieval_x}
    texts = {'train': np.concatenate((train_y, retrieval_y[:120])), 'query': query_y, 'database': retrieval_y}
    labels = {'train': np.concatenate((train_L, retrieval_L[:120])), 'query': query_L, 'database': retrieval_L}


    # imgs = {'train': train_x, 'query': query_x, 'database': retrieval_x}
    # texts = {'train': train_y, 'query': query_y, 'database': retrieval_y}
    # labels = {'train': train_L, 'query': query_L, 'database': retrieval_L}
    # imgs = {'train': train_x[:], 'query': query_x, 'database': retrieval_x}
    # texts = {'train': train_y[:], 'query': query_y, 'database': retrieval_y}
    # labels = {'train': train_L[:], 'query': query_L, 'database': retrieval_L}

    # imgs2 = {'train': train_x, 'query': query_x,
    #          'database': retrieval_x[:]}
    # texts2 = {'train': train_y, 'query': query_y, 'database': retrieval_y}
    # labels2 = {'train': train_L, 'query': query_L, 'database': retrieval_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    # dataset2 = {x: CustomDataSet(images=imgs2[x], texts=texts2[x], labels=labels2[x])
    #            for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['query', 'train', 'database']}

    # dataloader2 = {x: DataLoader(dataset2[x], batch_size=batch_size,
    #                             shuffle=shuffle[x], num_workers=0) for x in ['query', 'train', 'database']}

    return dataloader, (np.concatenate((train_x, retrieval_x[:120])), np.concatenate((train_y, retrieval_y[:120])), np.concatenate((train_L, retrieval_L[:120])))#, dataloader2


def get_loader_coco_CLIP10000(batch_size):
    path = 'E:/Hash/dataset/CIRH/MIRFlickr/'
    dataset = 'mscoco'
    train_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/train.pkl'
    query_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/query.pkl'
    retrieval_loc = 'E:/Hash/dataset/CLIP/CLIP/' + dataset + '/retrieval.pkl'

    # x: images   y:tags   L:labels
    # with open(train_loc, 'rb') as f_pkl:
    #     data = pickle.load(f_pkl)
    #     train_L = np.array(data['label'], dtype=np.float)
    #     train_x = np.array(data['image'], dtype=np.float)
    #     train_y = np.array(data['text'], dtype=np.float)
    # train_set = sio.loadmat(train_loc)


    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        query_L = np.array(data['label'], dtype=np.float)
        query_x = np.array(data['image'], dtype=np.float)
        query_y = np.array(data['text'], dtype=np.float)

    # test_set = sio.loadmat(query_loc)
    # query_L = np.array(test_set['label'], dtype=np.float)
    # query_x = np.array(test_set['image'], dtype=np.float)
    # query_y = np.array(test_set['text'], dtype=np.float)

    with open(retrieval_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        retrieval_L = np.array(data['label'], dtype=np.float)
        retrieval_x = np.array(data['image'], dtype=np.float)
        retrieval_y = np.array(data['text'], dtype=np.float)

        train_L = np.array(data['label'], dtype=np.float)[:10000]
        train_x = np.array(data['image'], dtype=np.float)[:10000]
        train_y = np.array(data['text'], dtype=np.float)[:10000]
    # db_set = sio.loadmat(retrieval_loc)
    # retrieval_L = np.array(db_set['label'], dtype=np.float)
    # retrieval_x = np.array(db_set['image'], dtype=np.float)
    # retrieval_y = np.array(db_set['text'], dtype=np.float)


    imgs = {'train': train_x, 'query': query_x, 'database': retrieval_x}
    texts = {'train': train_y, 'query': query_y, 'database': retrieval_y}
    labels = {'train': train_L, 'query': query_L, 'database': retrieval_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['query', 'train', 'database']}

    return dataloader, (train_x, train_y, train_L)

def get_loader_nus(batch_size):
    path = 'E:/Hash/dataset/CIRH/NUS-WIDE/'

    # x: images   y:tags   L:labels
    train_set = sio.loadmat(path + 'nus_train.mat')
    train_L = np.array(train_set['L_tr'], dtype=np.float)
    train_x = np.array(train_set['I_tr'], dtype=np.float)
    train_y = np.array(train_set['T_tr'], dtype=np.float)

    test_set = sio.loadmat(path + 'nus_query.mat')
    query_L = np.array(test_set['L_te'], dtype=np.float)
    query_x = np.array(test_set['I_te'], dtype=np.float)
    query_y = np.array(test_set['T_te'], dtype=np.float)

    db_set = sio.loadmat(path + 'nus_database.mat')
    retrieval_L = np.array(db_set['L_db'], dtype=np.float)
    retrieval_x = np.array(db_set['I_db'], dtype=np.float)
    retrieval_y = np.array(db_set['T_db'], dtype=np.float)

    imgs = {'train': train_x, 'query': query_x, 'database': retrieval_x}
    texts = {'train': train_y, 'query': query_y, 'database': retrieval_y}
    labels = {'train': train_L, 'query': query_L, 'database': retrieval_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['query', 'train', 'database']}

    return dataloader, (train_x, train_y, train_L)


def get_loader_coco(batch_size):
    path = './datasets/MSCOCO/'

    # x: images   y:tags   L:labels
    train_set = sio.loadmat(path + 'COCO_train.mat')
    train_L = np.array(train_set['L_tr'], dtype=np.float)
    train_x = np.array(train_set['I_tr'], dtype=np.float)
    train_y = np.array(train_set['T_tr'], dtype=np.float)

    test_set = sio.loadmat(path + 'COCO_query.mat')
    query_L = np.array(test_set['L_te'], dtype=np.float)
    query_x = np.array(test_set['I_te'], dtype=np.float)
    query_y = np.array(test_set['T_te'], dtype=np.float)

    db_set = h5py.File(path + 'COCO_database.mat', 'r', libver='latest', swmr=True)
    retrieval_L = np.array(db_set['L_db'], dtype=np.float).T
    retrieval_x = np.array(db_set['I_db'], dtype=np.float).T
    retrieval_y = np.array(db_set['T_db'], dtype=np.float).T
    db_set.close()

    imgs = {'train': train_x, 'query': query_x, 'database': retrieval_x}
    texts = {'train': train_y, 'query': query_y, 'database': retrieval_y}
    labels = {'train': train_L, 'query': query_L, 'database': retrieval_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in
                  ['query', 'train', 'database']}

    return dataloader, (train_x, train_y, train_L)


def get_loader_wiki(batch_size):
    path = 'E:/Hash/dataset/hash/'

    # x: images   y:tags   L:labels
    train_set = h5py.File(path + 'wiki_data.mat')

    # I_test = np.transpose(np.float32(mat['I_te']))
    # L_test = np.transpose(np.float32(mat['L_te']))
    # T_test = np.transpose(np.float32(mat['T_te']))
    # I_train = np.transpose(np.float32(mat['I_tr']))
    # L_train = np.transpose(np.float32(mat['L_tr']))
    # T_train = np.transpose(np.float32(mat['T_tr']))
    train_L = np.transpose(np.array(np.float32(train_set['L_tr']), dtype=np.float))
    train_x = np.transpose(np.array(np.float32(train_set['I_tr']), dtype=np.float))
    train_y = np.transpose(np.array(np.float32(train_set['T_tr']), dtype=np.float))

    query_L = np.transpose(np.array(np.float32(train_set['L_te']), dtype=np.float))
    query_x = np.transpose(np.array(np.float32(train_set['I_te']), dtype=np.float))
    query_y = np.transpose(np.array(np.float32(train_set['T_te']), dtype=np.float))

    retrieval_L = np.transpose(np.array(np.float32(train_set['L_tr']), dtype=np.float))
    retrieval_x = np.transpose(np.array(np.float32(train_set['I_tr']), dtype=np.float))
    retrieval_y = np.transpose(np.array(np.float32(train_set['T_tr']), dtype=np.float))


    imgs = {'train': train_x, 'query': query_x, 'database': retrieval_x}
    texts = {'train': train_y, 'query': query_y, 'database': retrieval_y}
    labels = {'train': train_L, 'query': query_L, 'database': retrieval_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['query', 'train', 'database']}

    return dataloader, (train_x, train_y, train_L)

