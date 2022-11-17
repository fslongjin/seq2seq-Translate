import torch
from matplotlib import pyplot as plt, ticker
from multiprocessing.dummy import Pool as ThreadPool
import config
import os
from model import EncoderRNN, AttnDecoderRNN
from utils import tensor_from_sentence, Lang, prepare_data, normalize_string

import threading

get_model_lock = threading.Lock()

__encoder = None
__decoder = None
__input_lang = None
__target_lang = None
__pairs = None


def get_model():
    get_model_lock.acquire()
    global __encoder, __decoder, __input_lang, __target_lang, __pairs

    if __encoder is not None and __decoder is not None:
        get_model_lock.release()
        return __encoder, __decoder, __input_lang, __target_lang, __pairs

    __input_lang, __target_lang, __pairs = prepare_data("English", "Chinese", True)

    # __encoder = EncoderRNN(__input_lang.n_words, config.hidden_size).to(config.device)
    # __decoder = AttnDecoderRNN(config.hidden_size, __target_lang.n_words, dropout_p=config.dropout_p).to(config.device)

    # 获取已经保存的模型的列表
    params = os.listdir(config.model_save_path)
    params.sort(reverse=True)
    encoder_path = None
    decoder_path = None
    for x in params:
        if encoder_path is None:
            if x.startswith('encoder'):
                encoder_path = os.path.join(config.model_save_path, x)

        if decoder_path is None:
            if x.startswith('decoder'):
                decoder_path = os.path.join(config.model_save_path, x)

    if encoder_path is None or decoder_path is None:
        print("模型参数文件不存在！encoder_path={}, decoder_path={}".format(encoder_path, decoder_path))
        get_model_lock.release()
        raise FileNotFoundError

    # 加载模型参数
    print("正在从文件：{} 加载encoder参数...".format(encoder_path))
    __encoder = torch.load(encoder_path, map_location=torch.device(config.device))
    print("正在从文件：{} 加载decoder参数...".format(decoder_path))
    __decoder = torch.load(decoder_path, map_location=torch.device(config.device))
    print("模型加载成功！")
    get_model_lock.release()

    return __encoder, __decoder, __input_lang, __target_lang, __pairs


def do_evaluate(encoder, decoder, input_lang, target_lang: Lang, sentence, max_length=config.SEQ_MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        # Encoder的输出
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=config.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[int(config.WordIndex.SOS_token)]], device=config.device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)

            if topi.item() == int(config.WordIndex.EOS_token):
                # decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(target_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluate(sentence: str):
    raw_str = sentence
    try:
        sentence = normalize_string(sentence)
        encoder, decoder, input_lang, target_lang, pairs = get_model()
        decoded_words, attentions = do_evaluate(encoder, decoder, input_lang, target_lang, sentence)

        # 拼接成完整的句子
        result = ''
        for x in decoded_words:
            result += x + ' '

        print("翻译完成, sentence={}, result={}".format(raw_str, result))
        return result
    except Exception as e:
        print("Err occurred, raw_str={}, msg={}".format(raw_str, e))
        raise e


def show_attention(save_path, input_sentence, output_words, attentions):
    """
    绘制模型的注意力图表
    :param save_path: 图片保存的目标位置
    :param input_sentence: 输入的句子（经过正则化的）
    :param output_words: 输出的结果
    :param attentions: 注意力tensor
    :return: None
    """

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(save_path)


def evaluate_and_show_attention(sentences: list):
    for sentence in sentences:
        raw_str = sentence
        try:
            sentence = normalize_string(sentence)
            encoder, decoder, input_lang, target_lang, pairs = get_model()
            decoded_words, attentions = do_evaluate(encoder, decoder, input_lang, target_lang, sentence)

            show_attention(os.path.join(config.result_path, raw_str) + ".png", sentence, decoded_words, attentions)

        except Exception as e:
            print("Err occurred, raw_str={}, msg={}".format(raw_str, e))


def evaluate_on_dataset():
    """
    计算在训练集上的bleu分数
    :return:
    """
    from nltk.translate.bleu_score import sentence_bleu
    from tqdm import tqdm
    from queue import Queue
    encoder, decoder, input_lang, target_lang, pairs = get_model()

    total_bleu_score = 0.0
    print("正在为{}条语句计算BLEU分数...".format(len(pairs)))
    with tqdm(total=len(pairs)) as pbar:

        if config.device.type == 'cpu':

            def worker(p):
                try:
                    _decoded_words, _attentions = do_evaluate(encoder, decoder, input_lang, target_lang, p[0])
                    # 拼接成完整的句子
                    _result = ''
                    for x in _decoded_words:
                        _result += x + ' '

                    q.put(sentence_bleu([p[1]], _result))
                except Exception as e:
                    print("在评估{}时发生错误".format(p))

                pbar.update(1)

            q = Queue()
            pool = ThreadPool(4)
            pool.map(worker, pairs)
            pool.close()

            while not q.empty():
                total_bleu_score += q.get()

        else:
            for pair in pairs:
                try:
                    decoded_words, attentions = do_evaluate(encoder, decoder, input_lang, target_lang, pair[0])
                    # print("sentence_bleu([pair[1]], decoded_words)={}".format(sentence_bleu([pair[1]], decoded_words)))
                    result = ''
                    for x in decoded_words:
                        result += x + ' '
                    total_bleu_score += sentence_bleu([pair[1]], result)
                except Exception as e:
                    print("在评估{}时发生错误".format(pair))

                pbar.update(1)

    print("已评估{}条语句，平均bleu score:".format(len(pairs)), total_bleu_score / len(pairs))


if __name__ == '__main__':
    data = ['我们非常需要食物',
            '他总是忘记事情',
            '他和他的邻居相处',
            '我肯定他会成功的',
            '她可以教英语',]
    evaluate_and_show_attention(data)
    evaluate_on_dataset()
