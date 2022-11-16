import os.path
import random
import time

import torch
import torch.nn as nn
from torch import optim
import config
import utils
from utils import read_langs, filter_pairs
from model import EncoderRNN, AttnDecoderRNN


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
    indexes.append(config.WordIndex.EOS_token)
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


def do_train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
             criterion, max_length=config.SEQ_MAX_LENGTH):
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

    # Teacher Forcing: 使用目标语句直接作为下一次decoder的输入
    use_teacher_forcing = True if random.random() > config.teacher_forcing_threshold else False

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
            if decoder_input.item() == config.WordIndex.EOS_token:
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

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # 为每个迭代选择要用于训练的数据
    training_pairs = [tensors_from_pair(input_lang, target_lang, random.choice(pairs)) for i in range(n_iters)]

    # 定义损失函数
    criterion = nn.NLLLoss()

    for i in range(1, n_iters + 1):
        training_pair = training_pairs[i - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = do_train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
                        criterion)

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
