#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    训练并保存模型
"""
import os
import sys
import pickle
from time import time
import numpy as np
from optparse import OptionParser
from utils import read_pkl, SentenceDataUtil
from utils import is_interactive, parse_int_list
from model import TextCNN

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader


op = OptionParser()
op.add_option('--ri', dest='root_idx', default='./data/train_idx', type='str', help='数据索引根目录')
op.add_option('--rv', dest='root_voc', default='./res/voc', type='str', help='字典根目录')
op.add_option('--re', dest='root_embed', default='./res/embed', type='str', help='embed根目录')
op.add_option('--ml', dest='max_len', default=50, type='int', help='实例最大长度')
op.add_option('--ds', dest='dev_size', default=0.2, type='float', help='开发集占比')
op.add_option('--nc', dest='nb_class', type='int', help='类别数')
op.add_option('--wd', dest='word_dim', default=50, type='int', help='词向量维度')
op.add_option('--fs', dest='filter_size', default=[2, 3, 4], type='str',
              action='callback', callback=parse_int_list, help='卷积核尺寸')
op.add_option('--fn', dest='filter_num', default=[256, 256, 256], type='str',
              action='callback', callback=parse_int_list, help='卷积核数量')
op.add_option('--dp', dest='dropout', default=0.5, type='float', help='dropout rate')
op.add_option('--ne', dest='nb_epoch', default=100, type='int', help='迭代次数')
op.add_option('--mp', dest='max_patience', default=5,
              type='int', help='最大耐心值')
op.add_option('--rm', dest='root_model', default='./model/',
              type='str', help='模型根目录')
op.add_option('--bs', dest='batch_size', default=64, type='int', help='batch size')
op.add_option('-g', '--cuda', dest='cuda', action='store_true', default=False, help='是否使用GPU加速')
op.add_option('--nw', dest='nb_work', default=4, type='int', help='加载数据的线程数')
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if not opts.nb_class:
    op.print_help()
    exit()

# 初始化数据参数
root_idx = opts.root_idx
path_num = os.path.join(root_idx, 'instance.csv')
max_len = opts.max_len
root_voc = opts.root_voc
word2id_dict = read_pkl(os.path.join(root_voc, 'word2id.pkl'))
label2id_dict = read_pkl(os.path.join(root_voc, 'label2id.pkl'))
has_label = True
dev_size = opts.dev_size
batch_size = opts.batch_size
num_worker = opts.nb_work

# 初始化数据
dataset = SentenceDataUtil(
    path_num, root_idx, max_len, word2id_dict, has_label, label2id_dict, shuffle=True)
dataset_train, dataset_dev = dataset.split_train_and_dev(dev_size=dev_size)

# 划分训练集和开发集
data_loader_train = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_worker)
data_loader_dev = DataLoader(
    dataset_dev, batch_size=batch_size, shuffle=False, num_workers=num_worker)


# 初始化模型参数
nb_classes = opts.nb_class
word_dim = opts.word_dim
vocab_size = len(word2id_dict) + 1
filter_size = opts.filter_size
filter_num = opts.filter_num
if len(filter_num) != len(filter_size):
    for i in range(len(filter_size)-len(filter_num)):
        filter_num.append(filter_num[0])
filter_num = filter_num[:len(filter_size)]
filter_shape = list(zip(filter_size, filter_num))
path_embed = os.path.join(opts.root_embed, 'word2vec.pkl')
pretrained_embed = None
if os.path.exists(path_embed):
    pretrained_embed = read_pkl(path_embed)
dropout_rate = opts.dropout
kwargs = {'nb_classes': nb_classes, 'vocab_size': vocab_size, 'input_size': word_dim,
          'filter_shape': filter_shape, 'pretrained_embed': pretrained_embed,
          'dropout_rate': dropout_rate}


# 初始化模型
use_cuda = opts.cuda
text_cnn = TextCNN(kwargs)
print(text_cnn)
if use_cuda:
    text_cnn = text_cnn.cuda()
optimizer = torch.optim.Adam(text_cnn.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 训练
t0 = time()
nb_epoch = opts.nb_epoch
max_patience = opts.max_patience
current_patience = 0
root_model = opts.root_model
if not os.path.exists(root_model):
    os.makedirs(root_model)
path_model = os.path.join(root_model, 'textcnn.model')
best_dev_loss = 1000.
for epoch in range(nb_epoch):
    sys.stdout.write('epoch {0} / {1}: \r'.format(epoch, nb_epoch))
    total_loss, dev_loss = 0., 0.
    text_cnn.train()
    current_count = 0
    for i_batch, sample_batched in enumerate(data_loader_train):
        optimizer.zero_grad()
        data = Variable(sample_batched['data'])
        label = Variable(sample_batched['label'])
        if use_cuda:
            data = data.cuda()
            label = label.cuda()
        target = text_cnn(data)
        loss = criterion(target, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]

        current_count += sample_batched['data'].size()[0]
        sys.stdout.write('epoch {0} / {1}: {2} / {3}\r'.format(
            epoch, nb_epoch, current_count, len(dataset_train)))

    sys.stdout.write('epoch {0} / {1}: {2} / {3}\n'.format(
        epoch, nb_epoch, current_count, len(dataset_train)))

    # 计算开发集loss
    text_cnn.eval()
    for i_batch, sample_batched in enumerate(data_loader_dev):
        data = Variable(sample_batched['data'])
        label = Variable(sample_batched['label'])
        if use_cuda:
            data = data.cuda()
            label = label.cuda()
        pred = text_cnn(data)
        loss = criterion(pred, label)
        dev_loss += loss.data[0]

    total_loss /= float(len(data_loader_train))
    dev_loss /= float(len(data_loader_dev))
    print('\ttrain loss: {:.4f}, dev loss: {:.4f}'.format(total_loss, dev_loss))

    # 根据开发集loss保存模型
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        # 保存模型
        torch.save(text_cnn, path_model)
        print('\tmodel has saved to {0}!'.format(path_model))
        current_patience = 0
    else:
        current_patience += 1
        print('\tno improvement, current patience: {0} / {1}'.format(current_patience, max_patience))
        if max_patience <= current_patience:
            print('finished training! (early stopping, max patience: {0})'.format(max_patience))
            break
duration = time() - t0
print('finished training!')
print('done in {:.1f}s!'.format(duration))
