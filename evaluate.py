import torch
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
    print("正在加载encoder参数...")
    __encoder = torch.load(encoder_path, map_location=torch.device(config.device))
    print("正在加载decoder参数...")
    __decoder = torch.load(decoder_path, map_location=torch.device(config.device))
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
        return "Error occurred when translating."


if __name__ == '__main__':
    evaluate("我不想上学。")
