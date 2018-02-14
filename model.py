#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
models:
    CNNs:
    LSTMs:
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self, args):
        """
        Args:
            nb_classes: int, 分类类别数
            vocab_size: int, 词表大小
            input_size: int, 输入dim
            filter_shape: list of tuple, 卷积核尺寸, default is [[3, 200], [4, 200], [5, 200]]
            pretrained_embed: np.array, default is None
            dropout_rate: float, dropout rate
        """
        super(TextCNN, self).__init__()
        for k, v in args.items():
            self.__setattr__(k, v)

        # embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.input_size)
        if hasattr(self, 'pretrained_embed') and self.pretrained_embed is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_embed))

        # conv layer
        if not hasattr(self, 'filter_shape'):
            shape = [[3, 200], [4, 200], [5, 200]]
            self.__setattr__('filter_shape', shape)
        self.conv_encoders = []
        for i, (filter_size, filter_num) in enumerate(self.filter_shape):
            conv_name = 'conv_encoder_{0}'.format(i)
            conv = nn.Conv2d(
                in_channels=1, out_channels=filter_num,
                kernel_size=(filter_size, self.input_size))
            self.__setattr__(conv_name, conv)
            self.conv_encoders.append(self.__getattr__(conv_name))

        # dropout layer
        if not hasattr(self, 'dropout_rate'):
            self.__setattr__('dropout_rate', '0.5')
        self.dropout = nn.Dropout(self.dropout_rate)

        # dense layer
        dense_input_dim = sum([f[1] for f in self.filter_shape])
        self.logistic = nn.Linear(dense_input_dim, self.nb_classes)

        self._init_weight()

    def forward(self, inputs):
        """
        Args:
            inputs: autograd.Variable, size=[batch_size, max_len]
        """
        inputs = self.embedding(inputs)  # size=[batch_size, max_len, input_size]
        # [batch_size, 1, max_len, input_size]
        inputs = torch.unsqueeze(inputs, 1)
        conv_outputs = []
        for conv_encoder in self.conv_encoders:
            enc = F.relu(conv_encoder(inputs))
            k_h = enc.size()[2]
            enc = torch.squeeze(F.max_pool2d(enc, kernel_size=(k_h, 1)))
            conv_outputs.append(enc)
        conv_output = self.dropout(torch.cat(conv_outputs, 1))

        return F.log_softmax(self.logistic(conv_output))

    def _init_weight(self, scope=.1):
        if hasattr(self, 'pretrained_embed'):
            self.embedding.weight.data.uniform_(-scope, scope)
        self.logistic.weight.data.uniform_(-scope, scope)


class TextLSTM(nn.Module):

    def __init__(self):
        super(TextLSTM, self).__init__()
        # TODO

    def formard(self, inputs):
        # TODO
        pass
