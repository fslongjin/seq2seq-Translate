import random
from utils import read_langs, filter_pairs


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


def train():
    input_lang, output_lang, pairs = prepare_data("English", "Chinese", True)

    print(random.choice(pairs))

