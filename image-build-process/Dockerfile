FROM python:3.7


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# ENV LANG=en_US.utf8
# ENV LANG=C.UTF-8

# ENV PYTHONUNBUFFERED=TRUE
# ENV PYTHONDONTWRITEBYTECODE=TRUE
# ENV PATH="/opt/program:${PATH}"

# RUN pip3 install --upgrade pip

# ## install paddlepaddle framework
# RUN pip3 install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# RUN pip3 install paddleocr==2.0.1

# ## clone PaddleOCR source code 
# RUN git clone -b release/2.1 https://github.com/PaddlePaddle/PaddleOCR.git /opt/program/


# #download pretrained model for finetunine
# RUN mkdir /opt/program/pretrain/
# RUN cd /opt/program/pretrain/
# RUN wget -P /opt/program/pretrain/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar && tar -xf /opt/program/pretrain/ch_ppocr_mobile_v2.0_rec_train.tar -C /opt/program/pretrain/ && rm -rf /opt/program/pretrain/ch_ppocr_mobile_v2.0_rec_train.tar

# Set up the program in the image
RUN git clone https://github.com/catwhiskers/ocr_data_generator.git

#WORKDIR /opt/program
RUN cd ./ocr_data_generator && pip install -r requirements.txt && python generate_data.py seto_font train data/ZH_1.txt setofont 


# ENTRYPOINT ["python3", "train.py"]



