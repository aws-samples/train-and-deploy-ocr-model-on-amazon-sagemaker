FROM registry.baidubce.com/paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7

ENV LANG=en_US.utf8
ENV LANG=C.UTF-8

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

RUN pip3 install --upgrade pip

## install paddlepaddle framework
RUN pip3 install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
RUN pip3 install paddleocr==2.0.1

## clone PaddleOCR source code 
RUN git clone -b release/2.1 https://github.com/PaddlePaddle/PaddleOCR.git /opt/program/


#download pretrained model for finetunine
RUN mkdir /opt/program/pretrain/
RUN cd /opt/program/pretrain/
RUN wget -P /opt/program/pretrain/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar && tar -xf /opt/program/pretrain/ch_ppocr_mobile_v2.0_rec_train.tar -C /opt/program/pretrain/ && rm -rf /opt/program/pretrain/ch_ppocr_mobile_v2.0_rec_train.tar

# Set up the program in the image
COPY paddle-training-code/* /opt/program/
WORKDIR /opt/program

ENTRYPOINT ["python3", "train.py"]



