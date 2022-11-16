from abc import abstractmethod

import torch
import torch.nn as nn
from torch import optim

import torch.nn.functional as F

import config


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # Word embedding层
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_data, hidden):
        """
        前向计算的过程
        :param input_data: 输入的数据
        :param hidden: 当前隐藏层的参数
        :return: 输出的数据、当前模型的隐藏层参数
        """
        embedded = self.embedding(input_data).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=config.device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data, hidden):
        output = self.embedding(input_data).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=config.device)


class AttnDecoderRNN(nn.Module):
    """
    基于注意力机制的解码器
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=config.SEQ_MAX_LENGTH):
        """
        初始化使用了注意力机制的解码器
        :param hidden_size: 隐藏层大小
        :param output_size: 输出大小
        :param dropout_p: dropout rate
        :param max_length: 句子的最大长度
        """
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_data, hidden, encoder_outputs):
        embedded = self.embedding(input_data).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # 计算注意力权重
        attn_weights = F.softmax(
            self.attn(
                torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        # 将注意力权重与encoder的输出进行组合
        # 将两个权重在第0维度上增加1个维度，然后相乘
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=config.device)
