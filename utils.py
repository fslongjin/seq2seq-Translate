import os.path
import re
import time
import math
import unicodedata
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from config import WordIndex, SEQ_MAX_LENGTH
import torch
import config

plt.switch_backend('agg')


class Lang:
    def __init__(self, name):
        # 语言名
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {WordIndex.SOS_token: "SOS", WordIndex.EOS_token: "EOS"}

        # 由于存在SOS和EOS，因此n_words初始值设置为2
        self.n_words = 2

    def add_sentence(self, sentence: str):
        """
        将一个句子中的所有词语加到列表中
        :param sentence: 要添加的句子
        :return:
        """
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """
        将一个词语添加到字典
        :param word:
        :return:
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = s.lower().strip()
    if ' ' not in s:
        s = list(s)
        s = ' '.join(s)
    s = unicode_to_ascii(s)
    s = re.sub(r"([.。!！?？])", "", s)
    return s


def read_langs(lang1: str, lang2: str, reverse=True):
    print("正在读取数据...")

    lines = open("data/cmn.txt", encoding='utf-8').read().strip().split('\n')

    # 读取数据
    pairs = [[normalize_string(s) for s in l.split('\t')[:2]] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


# 保留以这些单词开头的英文句子
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_pair(p):
    """
    判断一个pair是否应当被放到训练集中
    :param p:
    :return:
    """
    return len(p[0].split(' ')) < SEQ_MAX_LENGTH and \
           len(p[1].split(' ')) < SEQ_MAX_LENGTH


def filter_pairs(pairs):
    """
    过滤所有的要被放到训练集中的pair
    :param pairs:
    :return:
    """
    return [pair for pair in pairs if filter_pair(pair)]


def as_minutes(s):
    """
    将秒转换为 "分 秒" 的形式
    :param s: 秒数
    :return:
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    """
    在给定当前时间和进度百分比的情况下返回 经过的时间和估计的剩余时间
    :param since: 开始时间
    :param percent: 完成的百分比
    :return: '经过的时间 (- 剩余时间)'
    """
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '已用时间：%s (剩余 %s)' % (as_minutes(s), as_minutes(rs))


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(os.path.join(config.result_path, "train_result.png"))


def indexes_from_sentence(lang, sentence):
    """
    为输入的句子，获取每个word的index
    :param lang: 语言对象
    :param sentence: 句子
    :return:
    """
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    """
    输入一个句子，为它创建一个tensor
    :param lang: 语言对象
    :param sentence: 句子
    :return:
    """
    indexes = indexes_from_sentence(lang, sentence)
    # 为每个句子末尾加上EOS token
    indexes.append(int(config.WordIndex.EOS_token))
    return torch.tensor(indexes, dtype=torch.long, device=config.device).view(-1, 1)


def tensors_from_pair(input_lang, target_lang, pair):
    """
    给定一个（源语言句子，目标语言句子）的二元组，为句子创建tensor
    :param input_lang: 源语言对象
    :param target_lang: 目标语言对象
    :param pair: 结构为（源语言句子，目标语言句子）的二元组
    :return:
    """
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(target_lang, pair[1])
    return input_tensor, target_tensor


def prepare_data(lang1, lang2, reverse=True):
    """
    数据预处理
    :param lang1: 数据第一列的语言名称
    :param lang2: 数据第二列的语言名称
    :param reverse: 是否交换第一第二语言
    :return:
    """
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)

    print("读取到 %s 个句对" % len(pairs))
    pairs = filter_pairs(pairs)
    print("经过过滤，最终剩下 %s 个句对" % len(pairs))
    print("正在对语句中的单词数目进行计算...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("总的单词数量：")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs
