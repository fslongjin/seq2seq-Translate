import os.path
import random
import time

import torch
import torch.nn as nn
from torch import optim
import config
import utils
from utils import read_langs, filter_pairs, tensors_from_pair, prepare_data
from model import EncoderRNN, AttnDecoderRNN


def check_if_use_teacher_forcing(current_iter, total_iters):
    """
    检查是否启用teacher forcing
    :param current_iter: 当前的迭代数
    :param total_iters: 总共需要的迭代数
    :return:
    """
    mode = config.teacher_forcing_mode
    if mode.lower() == 'const':
        threshold = config.teacher_forcing_threshold
    elif mode.lower() == 'linear':
        # 线性调整teacher forcing的阈值，使得在训练过程中，教师辅助逐渐变小，以提高模型的效果。
        threshold = config.teacher_forcing_threshold + \
                    (1-config.teacher_forcing_threshold)*(float(current_iter)/float(total_iters))
    else:
        print("Teaching forcing mode错误！")
        raise Exception("Teaching forcing mode错误")

    return True if random.random() > threshold else False


def do_train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
             criterion, max_length=config.SEQ_MAX_LENGTH, use_teacher_forcing=False):
    """
    训练一个epoch的函数
    :param input_tensor:
    :param target_tensor:
    :param encoder:
    :param decoder:
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param criterion:
    :param max_length: 句子的最大长度
    :param use_teacher_forcing 是否启用 teacher forcing
    :return: 本轮训练的平均损失
    """
    # 初始化encoder的隐藏层参数
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=config.device)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[int(config.WordIndex.SOS_token)]], device=config.device)
    # 将encoder的隐藏层参数作为decoder的隐藏层参数
    decoder_hidden = encoder_hidden

    if use_teacher_forcing:

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, target_tensor[di])

            # Teacher forcing
            decoder_input = target_tensor[di]
    else:
        # 不使用Teacher forcing: 使用decoder自己的输出作为下一次的输入
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 取输出的最大可能值
            topv, topi = decoder_output.topk(1)
            # 使用decoder自己的输出作为下一次的输入
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])

            # 遇到句子末尾，停下
            if decoder_input.item() == int(config.WordIndex.EOS_token):
                break

    # 反向传播
    loss.backward()
    # 更新优化器

    encoder_optimizer.step()
    decoder_optimizer.step()

    # 返回本轮训练的平均损失
    return loss.item() / target_length


def train_iters(encoder, decoder, input_lang, target_lang, pairs, n_iters, print_every=1000, plot_every=100,
                learning_rate=0.01):
    """
    训练多个迭代
    :param encoder: 编码器
    :param decoder: 解码器
    :param input_lang: 源语言对象
    :param target_lang: 目标语言对象
    :param pairs 句对
    :param n_iters: 训练的迭代数
    :param print_every: 每隔几个迭代打印一次日志
    :param plot_every: 每隔几个迭代画一个图
    :param learning_rate: 学习率
    :return: None
    """

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, weight_decay=0.001)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, weight_decay=0.001)

    # 为每个迭代选择要用于训练的数据
    training_pairs = [tensors_from_pair(input_lang, target_lang, random.choice(pairs)) for i in range(n_iters)]

    # 定义损失函数
    criterion = nn.NLLLoss()

    for i in range(1, n_iters + 1):
        training_pair = training_pairs[i - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        use_teacher_forcing = check_if_use_teacher_forcing(i, n_iters)
        loss = do_train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
                        criterion, use_teacher_forcing=use_teacher_forcing)

        print_loss_total += loss
        plot_loss_total += loss

        if i % config.checkpoint_steps == 0:
            # 保存checkpoint
            torch.save({'epoch': i, 'state_dict': encoder.state_dict(),
                        'optimizer': encoder_optimizer.state_dict()},
                       config.checkpoint_save_path + '/encoder-' + str(int(start)) + '-' + str(i) + '.pth.tar')

            torch.save({'epoch': i, 'state_dict': decoder.state_dict(),
                        'optimizer': decoder_optimizer.state_dict()},
                       config.checkpoint_save_path + '/decoder-' + str(int(start)) + '-' + str(i) + '.pth.tar')

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (utils.time_since(start, i / n_iters),
                                         i, i / n_iters * 100, print_loss_avg))

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    print("正在保存Encoder...")
    torch.save(encoder, os.path.join(config.model_save_path, "encoder-{}.model".format(str(int(start)))))
    print("正在保存Decoder...")
    torch.save(decoder, os.path.join(config.model_save_path, "decoder-{}.model".format(str(int(start)))))
    print("保存成功！")
    utils.show_plot(plot_losses)


def train():
    input_lang, target_lang, pairs = prepare_data("English", "Chinese", True)
    print(random.choice(pairs))

    encoder = EncoderRNN(input_lang.n_words, config.hidden_size).to(config.device)
    attn_decoder = AttnDecoderRNN(config.hidden_size, target_lang.n_words, dropout_p=config.dropout_p).to(config.device)

    train_iters(encoder, attn_decoder, input_lang, target_lang, pairs, config.iters, config.print_every,
                config.plot_every, config.learning_rate)


if __name__ == '__main__':
    print("启动训练...")
    train()
