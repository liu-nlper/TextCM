#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    预处理:
        1. 生成字典
        2. 将训练数据切分成小文件
"""
import os
import sys
import codecs
import pickle
import numpy as np
from optparse import OptionParser
from collections import Counter
from utils import build_word_voc, build_word_embed
from utils import is_interactive


def processing(path_data, has_label=True, root_data_idx=None, root_voc=None,
               root_embed=None, path_embed=None, percentile=98):
    """
    Args:
        path_data: str, 数据路径
        has_label: bool, 数据是否带有标签
        root_data_idx: str, 数据索引根目录
        root_voc: str, 词表根目录
        root_embed: str, embedding表根目录
        path_embed: str, embed文件的路径(txt or bin格式)
        percentile: int, 百分位, default is 98
    """
    data_idx = 0  # 数据编号
    if not os.path.exists(root_data_idx):
        os.mkdir(root_data_idx)
    file_data = codecs.open(path_data, 'r', encoding='utf-8')
    line = file_data.readline()
    if has_label:
        word_count_dict, label_set = Counter(), set()
    if not os.path.exists(root_data_idx):
        os.makedirs(root_data_idx)
    path_idx_num = os.path.join(root_data_idx, 'instance.csv')
    file_idx_num = codecs.open(path_idx_num, 'w', encoding='utf-8')
    sentence_lens = []  # 记录句子长度
    while line:
        line = line.strip()
        if not line:
            line = file_data.readline()
            continue
        if has_label:
            index = line.index('|')
            label, sentence = line[:index], line[index+1:]
            label_set.add(label)
            words = sentence.split(' ')
            sentence_lens.append(len(words))
            word_count_dict.update(words)
            file_idx_num.write('{0},{1}\n'.format(data_idx, label))
        else:
            sentence = line
            file_idx_num.write('{0}\n'.format(data_idx))

        # 句子写入文件
        path_sent = os.path.join(root_data_idx, '{0}.txt'.format(data_idx))
        file_sent = codecs.open(path_sent, 'w', encoding='utf-8')
        file_sent.write('{0}\n'.format(sentence))
        file_sent.close()

        line = file_data.readline()
        data_idx += 1
        sys.stdout.write('处理实例数: {0}\r'.format(data_idx))
        sys.stdout.flush()
    file_data.close()
    file_idx_num.close()
    print('处理实例数: {0}'.format(data_idx))

    if not has_label:
        print('done!')
        return

    # 构建vocs
    if not os.path.exists(root_voc):
        os.makedirs(root_voc)
    # 构建word voc
    print('构建word voc...')
    word2id_dict = build_word_voc(word_count_dict, percentile=percentile)
    path_word_voc = os.path.join(root_voc, 'word2id.pkl')
    file_word_voc = codecs.open(path_word_voc, 'wb')
    pickle.dump(word2id_dict, file_word_voc)
    file_word_voc.close()

    # 构建label voc
    print('构建label voc...')
    label2id_dict = dict()
    for label_idx, label in enumerate(sorted(label_set)):
        label2id_dict[label] = label_idx
    path_label_voc = os.path.join(root_voc, 'label2id.pkl')
    file_label_voc = codecs.open(path_label_voc, 'wb')
    pickle.dump(label2id_dict, file_label_voc)
    file_label_voc.close()

    # 从预训练的文件中构建embedding表
    if path_embed:
        if not os.path.exists(root_embed):
            os.makedirs(root_embed)
        print('构建word embedding表...')
        word_embed_table, unknow_count = build_word_embed(word2id_dict, path_embed)
        print('\t未登录词数量: {0}/{1}'.format(unknow_count, len(word2id_dict)))
        path_word_embed = os.path.join(root_embed, 'word2vec.pkl')
        file_word_embed = codecs.open(path_word_embed, 'wb')
        pickle.dump(word_embed_table, file_word_embed)
        file_word_embed.close()

    # 句子长度分布
    print('\n句子长度分布:')
    option_len_pt = [90, 95, 98, 100]
    for per in option_len_pt:
        tmp = int(np.percentile(sentence_lens, per))
        print('\t{0} percentile: {1}'.format(per, tmp))

    print('\n类别数: {0}\n'.format(len(label2id_dict)))

    print('done!')


op = OptionParser()
op.add_option(
    '-l', '--label', dest='label', action='store_true', default=False, help='数据是否带有标签(标志是否是训练集)')
op.add_option('--pd', dest='path_data', type='str', help='语料路径')
op.add_option('--ri', dest='root_idx', default='./data/train_idx', type='str', help='数据索引根目录')
op.add_option('--rv', dest='root_voc', default='./res/voc', type='str', help='字典根目录')
op.add_option('--re', dest='root_embed', default='./res/embed', type='str', help='embed根目录')
op.add_option('--pe', dest='path_embed', default=None, type='str', help='embed文件路径')
op.add_option('--pt', dest='pt', default=98, type='int', help='构建word voc的百分位值')
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if not opts.path_data:
    op.print_help()
    exit()

# 处理语料
if opts.label:  # 处理训练数据
    processing(
        opts.path_data, opts.label, opts.root_idx,
        opts.root_voc, opts.root_embed, opts.path_embed, opts.pt)
else:  # 处理测试数据
    processing(opts.path_data, opts.label, opts.root_idx)
