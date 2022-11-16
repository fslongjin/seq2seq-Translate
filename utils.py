import re
import time
import math
import unicodedata
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from config import WordIndex, SEQ_MAX_LENGTH

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
    s = re.sub(r"([.!?])", r" \1", s)
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
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
