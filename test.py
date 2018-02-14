#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import sys
import codecs
from time import time
from optparse import OptionParser
from utils import read_pkl, SentenceDataUtil
from utils import is_interactive
from model import TextCNN

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader


op = OptionParser()
op.add_option('--ri', '--root_idx', dest='root_idx', default='./data/test_idx', type='str', help='数据索引根目录')
op.add_option('--rv', '--root_voc', dest='root_voc', default='./res/voc', type='str', help='字典根目录')
op.add_option('--ml', '--max_len', dest='max_len', default=50, type='int', help='实例最大长度')
op.add_option('--pm', '--path_model', dest='path_model', default='./model/textcnn.model',
              type='str', help='模型路径')
op.add_option('--bs', '--batch_size', dest='batch_size', default=64, type='int', help='batch size')
op.add_option('-g', '--cuda', dest='cuda', action='store_true', default=False, help='是否使用GPU加速')
op.add_option('--nw', dest='nb_work', default=4, type='int', help='加载数据的线程数')
op.add_option('--pr', '--path_result', dest='path_result', default='./result.txt',
              type='str', help='预测结果存放路径')
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)

# 初始化数据参数
root_idx = opts.root_idx
path_num = os.path.join(root_idx, 'instance.csv')
max_len = opts.max_len
root_voc = opts.root_voc
word2id_dict = read_pkl(os.path.join(root_voc, 'word2id.pkl'))
label2id_dict = read_pkl(os.path.join(root_voc, 'label2id.pkl'))
has_label = False
batch_size = opts.batch_size
use_cuda = opts.cuda
num_worker = opts.nb_work
path_result = opts.path_result

t0 = time()

# 初始化数据
dataset = SentenceDataUtil(
    path_num, root_idx, max_len, word2id_dict, has_label, label2id_dict, shuffle=True)
dataset_test = dataset.get_all_data()
data_loader_test = DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_worker)

# 加载模型
path_model = opts.path_model
text_cnn = torch.load(path_model)

# 测试
label2id_dict_rev = dict()
for k, v in label2id_dict.items():
    label2id_dict_rev[v] = k
file_result = codecs.open(path_result, 'w', encoding='utf-8')
current_count, total_count = 0, len(dataset_test)
for i_batch, sample_batched in enumerate(data_loader_test):
    current_count += sample_batched['data'].size()[0]
    sys.stdout.write('{0} / {1}\r'.format(current_count ,total_count))
    data = Variable(sample_batched['data'])
    if use_cuda:
        data = data.cuda()
    target = text_cnn(data)
    pred_labels = torch.max(target, dim=1)[1].data
    for label in pred_labels:
        file_result.write('{0}\n'.format(label2id_dict_rev[label]))
sys.stdout.write('{0} / {1}\n'.format(current_count ,total_count))
file_result.close()
print('done in {:.1f}s!'.format(time()-t0))
