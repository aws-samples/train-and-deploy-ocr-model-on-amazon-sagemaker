#FROM paddlepaddle/paddle:2.0.0rc1-gpu-cuda10.1-cudnn7#
FROM registry.baidubce.com/paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7
#ARG REGISTRY_URI
#FROM ${REGISTRY_URI}/pytorch-training:1.3.1-cpu-py36-ubuntu16.04
#FROM ${REGISTRY_URI}/pytorch-training:1.5.1-gpu-py36-cu101-ubuntu16.04
#FROM ${REGISTRY_URI}/autogluon-training:0.2.1-gpu-py37-cu102-ubuntu18.04

ENV LANG=en_US.utf8
ENV LANG=C.UTF-8

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
#ENV LD_LIBRARY_PATH='/usr/local/cuda-10.1/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'

##########################################################################################
# SageMaker requirements
##########################################################################################
RUN pip3 install --upgrade pip

## install flask
#RUN pip3 install networkx==2.3 flask gevent gunicorn boto3 paddleocr>=2.0.1
#todo: need to test the paddleocr version
RUN pip3 install networkx==2.3 flask gevent gunicorn boto3 paddleocr==2.0.1

#RUN pip3 install paddlepaddle-gpu==2.0rc1 -i https://opentuna.cn/pypi/web/simple
RUN pip3 install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

#add folder
#RUN git clone https://github.com/jackie930/PaddleOCR.git /opt/program/
RUN git clone -b release/2.1 https://github.com/PaddlePaddle/PaddleOCR.git /opt/program/

#download pretrained model for finetunine
RUN mkdir /opt/program/pretrain/
RUN cd /opt/program/pretrain/
RUN wget -P /opt/program/pretrain/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar && tar -xf /opt/program/pretrain/ch_ppocr_mobile_v2.0_rec_train.tar -C /opt/program/pretrain/ && rm -rf /opt/program/pretrain/ch_ppocr_mobile_v2.0_rec_train.tar

#download model for inference
RUN mkdir /opt/program/inference/
RUN cd /opt/program/inference/
RUN wget -P /opt/program/inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar && tar -xf /opt/program/inference/ch_ppocr_server_v2.0_det_infer.tar -C /opt/program/inference/ && rm -rf /opt/program/inference/ch_ppocr_server_v2.0_det_infer.tar
RUN wget -P /opt/program/inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar && tar -xf /opt/program/inference/ch_ppocr_mobile_v2.0_cls_infer.tar -C /opt/program/inference/ && rm -rf /opt/program/inference/ch_ppocr_mobile_v2.0_cls_infer.tar
#RUN wget -P /opt/program/inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar && tar -xf /opt/program/inference/ch_ppocr_server_v2.0_rec_infer.tar -C /opt/program/inference/ && rm -rf /opt/program/inference/ch_ppocr_server_v2.0_rec_infer.tar

#RUN pip3.7 install -r /opt/program/requirements.txt -i https://opentuna.cn/pypi/web/simple

### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

# Set up the program in the image
COPY paddle/* /opt/program/
RUN chmod +x /opt/program/serve
WORKDIR /opt/program