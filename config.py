from enum import Enum


class WordIndex(Enum):
    # Start of sentence
    SOS_token = 0
    # End of sentence
    EOS_token = 1


# 句子的最大长度
SEQ_MAX_LENGTH = 12
