from enum import IntEnum
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WordIndex(IntEnum):
    # Start of sentence
    SOS_token = 0
    # End of sentence
    EOS_token = 1


# 句子的最大长度
SEQ_MAX_LENGTH = 30
# 隐藏层大小
hidden_size = 256

# ===== 与训练相关的设置 =====
# 启用teacher forcing加速训练的阈值. 当随机数大于这个值的时候将会启用。
teacher_forcing_threshold = 0.5
# 训练时的dropout rate
dropout_p = 0.1

# 训练的迭代次数
iters = 75000
# 训练多少轮，打印一次日志
print_every = 1000
# 训练多少轮，绘制一次图像
plot_every = 1000
learning_rate = 0.01

# 模型保存的路径
model_save_path = 'models/'
checkpoint_save_path = "checkpoints/"

# 训练多少轮保存一次checkpoint
checkpoint_steps = 1000
