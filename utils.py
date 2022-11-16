import re
import unicodedata
from config import WordIndex, SEQ_MAX_LENGTH


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
           len(p[1].split(' ')) < SEQ_MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filter_pairs(pairs):
    """
    过滤所有的要被放到训练集中的pair
    :param pairs:
    :return:
    """
    return [pair for pair in pairs if filter_pair(pair)]
