"""Feature engineers the abalone dataset."""


import os 
import shutil

from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)
import logging
import argparse
import pathlib
import boto3 

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def get_strings(file_name): 
    f = open(file_name, 'r')
    results = []
    for l in f.readlines():
        if l and l.strip():
            results.append(l.strip())
    return results
        

def get_fonts(font_dir):
    onlyfiles = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f))]
    return onlyfiles
    


def get_training_data_img_and_labels(string_file, font_dir, output_folder, img_prefix, limit=1000): 
    strings = get_strings(string_file)
    fonts = get_fonts(font_dir)
    print(strings)
    print(fonts)
    generator = GeneratorFromStrings(
        strings,
        fonts = [f"{font_dir}/setofont.ttf"], 
#         blur=2,
#         random_blur=True
    )
    labels = [] 
    i = 0 
    for img, lbl in generator:
        if i<=limit: 
            file_name = os.path.join(output_folder, str(i)+".jpg")
            in_label_file_name = os.path.join(img_prefix, str(i)+".jpg")
            img.save(file_name)
            labels.append((in_label_file_name, lbl))
            i+=1 
        else: 
            break

    label_file = open(os.path.join(output_folder, "train.txt"), 'w')
    for l in labels: 
        line = '\t'.join(l)
        label_file.write(line)
        label_file.write('\n')


        
import sys 



if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    train_fn = f"{base_dir}/data/train.txt"
    test_fn = f"{base_dir}/data/test.txt"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key+"/train.txt", train_fn)
    s3.Bucket(bucket).download_file(key+"/test.txt", test_fn)
    font_dir = "/opt/program/ocr_data_generator/setofont"
    train_output_folder = f"{base_dir}/train"
    test_output_folder = f"{base_dir}/test"
    os.mkdir(train_output_folder)
    os.mkdir(test_output_folder)
    get_training_data_img_and_labels(train_fn, font_dir, train_output_folder, "train")    
    get_training_data_img_and_labels(test_fn, font_dir, test_output_folder, "test")    
    
