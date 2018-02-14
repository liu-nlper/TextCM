#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import codecs
import random
from . import read_csv, items2id_array

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset


class SentenceDataset(Dataset):

    def __init__(self, nums, root_idx, max_len, word2id_dict, labels=None,
                 label2id_dict=None):
        """
        Args:
            nums: list, 实例编号
            root_idx: str, 索引文件根目录
            max_len: int, 句子最大长度
            word2id_dict: dict, 词->id映射字典
            labels: None or list of labels, 标签
            label2id_dict: dict, label->id映射字典
        """
        self.nums = nums
        self.root_idx = root_idx
        self.max_len = max_len
        self.word2id_dict = word2id_dict
        self.labels = labels
        self.label2id_dict = label2id_dict

    def __len__(self):
        return len(self.nums)

    def __getitem__(self, idx):
        path_sentence = os.path.join(self.root_idx, '{0}.txt'.format(self.nums[idx]))
        words = codecs.open(path_sentence, 'r', encoding='utf-8').readline().strip().split()
        arr = items2id_array(words, self.word2id_dict, self.max_len)
        sample = {'data': arr}
        if self.labels is not None:
            label_id = self.label2id_dict[self.labels[idx]]
            sample.update({'label': label_id})
        return sample


class SentenceDataUtil():

    def __init__(self, path_num, root_idx, max_len, word2id_dict, has_label=True,
                 label2id_dict=None, shuffle=False, seed=1337):
        """
        Args:
            path_num: str, 实例编号文件，若带标签，则用逗号分隔
            root_idx: str, 索引文件根目录
            max_len: int, 句子最大长度
            word2id_dict: dict, 词->id映射字典
            has_label: bool, 数据中是否带标签
            label2id_dict: dict, label->id映射字典
            shuffle: 是否打乱数据集, default is False
            seed: int, 随机数种子, default is  1337
        """
        nums_and_labels = read_csv(path_num)
        self.nums = [int(term[0]) for term in nums_and_labels]
        self.has_label = has_label
        if self.has_label:
            self.labels = [term[1] for term in nums_and_labels]
        self.root_idx = root_idx
        self.max_len = max_len
        self.word2id_dict = word2id_dict
        self.label2id_dict = label2id_dict
        self.shuffle = shuffle
        self.seed = seed

    def shuffle_data(self):
        random.seed(self.seed)
        random.shuffle(self.nums)
        if self.has_label:
            random.seed(self.seed)
            random.shuffle(self.labels)

    def split_train_and_dev(self, dev_size=0.2):
        """
        划分训练集和开发集

        Args:
            dev_size: None, or a float value between 0 and 1

        Returns:
            dataset_train: torch.utils.data.Dataset
            dataset_dev: torch.utils.data.Dataset
        """
        if self.shuffle:
            self.shuffle_data()

        boundary = int(len(self.nums) * (1. - dev_size))
        nums_train, nums_dev = self.nums[:boundary], self.nums[boundary:]
        labels_train, labels_dev = None, None
        if self.has_label:
            labels_train, labels_dev = self.labels[:boundary], self.labels[boundary:]

        dataset_train = SentenceDataset(
            nums_train, self.root_idx, self.max_len,
            self.word2id_dict, labels_train, self.label2id_dict)
        dataset_dev = SentenceDataset(
            nums_dev, self.root_idx, self.max_len,
            self.word2id_dict, labels_dev, self.label2id_dict)

        return dataset_train, dataset_dev

    def get_all_data(self):
        """
        获取全部数据集

        Returns:
            dataset: torch.utils.data.Dataset
        """
        if self.shuffle:
            self.shuffle_data()
        if not hasattr(self, 'labels'):
            self.labels = None
        dataset = SentenceDataset(
            self.nums, self.root_idx, self.max_len,
            self.word2id_dict, self.labels, self.label2id_dict)
        return dataset
