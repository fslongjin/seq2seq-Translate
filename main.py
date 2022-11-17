from __future__ import unicode_literals, print_function, division
import uvicorn
import json
from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse
import torch

import os
from multiprocessing import cpu_count
import train
from evaluate import evaluate
app = FastAPI()


@app.get('/seq2seq/{src_sentence}')
async def seq2seq_translate(src_sentence: str):
    """
    使用seq2seq模型进行翻译
    :param src_sentence: 中文字符串
    :return:
    """
    res_data = dict()
    try:
        res_data['result'] = evaluate(src_sentence)
        res_data['status'] = 0
        return JSONResponse(status_code=200, content=res_data)
    except Exception as e:
        res_data['status'] = -1
        res_data['result'] = "Error occurred when translating."
        return JSONResponse(status_code=500, content=res_data)


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=10086, reload=True)
