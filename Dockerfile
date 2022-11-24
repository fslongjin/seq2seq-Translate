FROM python:3.10-slim

# 拷贝本地所有数据到容器内
COPY . /seq2seq/

WORKDIR /seq2seq/

RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/  && \
    pip3 install torch==1.13.0+cpu torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

EXPOSE 10086

ENTRYPOINT [ "uvicorn", "main:app", "--port", "10086", "--host", "0.0.0.0"]