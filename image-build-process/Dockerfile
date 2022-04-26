FROM python:3.7


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ocr_data_generator/ /opt/program/ocr_data_generator

WORKDIR /opt/program
RUN cd ./ocr_data_generator && pip install -r requirements.txt 





