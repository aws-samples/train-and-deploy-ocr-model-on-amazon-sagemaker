# -*- coding: utf-8 -*-
import sys
import json
import boto3
import os
import warnings
import numpy as np
from paddleocr import PaddleOCR
from inference import *
import cv2

warnings.filterwarnings("ignore",category=FutureWarning)


import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass


with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    # from autogluon import ImageClassification as task

import flask

# The flask app for serving predictions
app = flask.Flask(__name__)

s3_client = boto3.client('s3')

#check and init models
# make sure the model parameters exist
for i in ['/opt/program/inference/ch_ppocr_server_v2.0_det_infer',
          '/opt/ml/model',
          '/opt/program/inference/ch_ppocr_mobile_v2.0_cls_infer']:
    if os.path.exists(i):
        print("<<<<pretrained model exists for :", i)
    else:
        print("<<< make sure the model parameters exist for: ", i)
        break

# list the files under opt/ml/model
print ("<<< files under opt/ml/model", os.listdir('/opt/ml/model/'))
print ("start!!!!")
ocr = PaddleOCR(det_model_dir='/opt/program/inference/ch_ppocr_server_v2.0_det_infer',
                rec_model_dir='/opt/ml/model',
                rec_char_dict_path='/opt/program/ppocr_keys_v1.txt',
                cls_model_dir='/opt/program/inference/ch_ppocr_mobile_v2.0_cls_infer',
                use_pdserving=False)  # need to run only once to download and load model into memory
print ("test!!!!")

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def bbox_main(type, imgpath, detect='paddle'):
    # 主函数：输入图像路径返回key-val匹配后的字典，字典key为名称，val为的bbox的四个顶点
    # input：str，图像路径
    # output：dict，dict，保存key-val

    #(filepath, tempfilename) = os.path.split(imgpath)
    #(filename, extension) = os.path.splitext(tempfilename)
    '''
    print ("start!!!!")
    ocr = PaddleOCR(det_model_dir='/opt/program/inference/ch_ppocr_server_v2.0_det_infer',
                    rec_model_dir='/opt/ml/model',
                    rec_char_dict_path='/opt/program/ppocr_keys_v1.txt',
                    cls_model_dir='/opt/program/inference/ch_ppocr_mobile_v2.0_cls_infer',
                    use_pdserving=False)  # need to run only once to download and load model into memory
    print ("test!!!!")
    '''

    if detect == 'paddle':
        if type == 'img_path':
            img = cv2.imread(imgpath)
            img_shape = img.shape
            print ("<<< img shape: ", img_shape)
            result = ocr.ocr(imgpath, rec=True)
            print(result)

        elif type == 'img':
            img_shape = imgpath.shape
            print ("<<< img shape: ", img_shape)
            result = ocr.ocr(imgpath, rec=True)
            print (result)

        # save results
        res2 = {}

        label = []
        confidence = []
        bbox = []
        for i in result:
            label.append(i[1][0])
            confidence.append(i[1][1])
            bbox.append(i[0])

        res2['label'] = label
        res2['confidence'] = confidence
        res2['bbox'] = bbox

        print ('<<<< res2', res2)
        return res2, img_shape
    else:
        return

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    #parse json in request
    print ("<<<< flask.request.content_type", flask.request.content_type)
    if flask.request.content_type == 'application/json':
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)

        bucket = data['bucket']
        image_uri = data['image_uri']

        download_file_name = image_uri.split('/')[-1]
        print ("<<<<download_file_name ", download_file_name)
        s3_client.download_file(bucket, image_uri, download_file_name)

        print('Download finished!')

        print('Start to inference:')

        # LOAD MODEL
        label = ''
        try:
            res, img_shape = bbox_main('img_path',download_file_name, detect='paddle')
        except Exception as exception:
            print(exception)

        print ("Done inference! ")
        inference_result = {
            'label': res['label'],
            'confidences': res['confidence'],
            'bbox': res['bbox'],
            'shape': img_shape
        }
        _payload = json.dumps(inference_result,ensure_ascii=False,cls=MyEncoder)

        ## remove file to release memory
        os.remove(download_file_name)

        return flask.Response(response=_payload, status=200, mimetype='application/json')

    elif flask.request.content_type == 'image/jpeg':
        data = flask.request.data
        print("len(data)={}".format(len(data)))
        data_np = np.fromstring(data, dtype=np.uint8)
        print("data_np.shape={}".format(str(data_np.shape)))
        print(' '.join(['{:x}'.format(d) for d in data_np[:20].tolist()]), flush=True)
        data_np = cv2.imdecode(data_np, cv2.IMREAD_UNCHANGED)
        print('data_np: ', data_np)
        data_np = cv2.cvtColor(data_np, cv2.COLOR_BGR2RGB)

        print('Start to inference:')

        # LOAD MODEL
        label = ''
        try:
            res, img_shape = bbox_main('img', data_np, detect='paddle')
        except Exception as exception:
            print(exception)

        print("Done inference! ")
        inference_result = {
            'label': res['label'],
            'confidences': res['confidence'],
            'bbox': res['bbox'],
            'shape': img_shape
        }
        _payload = json.dumps(inference_result, ensure_ascii=False, cls=MyEncoder)
        return flask.Response(response=_payload, status=200, mimetype='application/json')

    else:
        return flask.Response(response='This predictor only supports JSON data and JPEG image data',
                              status=415, mimetype='text/plain')

