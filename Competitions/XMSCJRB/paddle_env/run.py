import os, sys
import cv2
import numpy as np
import pandas as pd
# import warnings
# warnings.filterwarnings('ignore')
from paddleocr import PaddleOCR

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Dense, Activation, Flatten, Dense,MaxPooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.models import load_model


# 压缩预测文件  此函数不要修改！！！
def parse_and_zip(result_save_path):
    os.system('zip -r submit.zip submit/')
    os.system(f'cp submit.zip {result_save_path}')


# text_identify
def textOCR(path):
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    result = ocr.ocr(path, cls=True)
    if result == [None]:
        result_txt = ''
    else:
        txt = []
        for line in result[0]:
            txt.append(line[-1][0])
        result_txt = ' '.join(txt)
    return result_txt


# 主函数，这个函数整体可根据选手自己模型的情况进行更改，在这里只是给出简单的示例
def main(to_pred_dir, result_save_path):
    dirpath = os.path.abspath(to_pred_dir)  # 待预测文件夹路径
    text_path = os.path.join(dirpath, 'text')  # 待预测文字图片文件夹路径
    num_path = os.path.join(dirpath, 'num')  # 待预测数字图片文件夹路径

    # 以下内容为示例，可以按照自己需求更改，主要是读入数据，并预测数据
    text_result = ['file_name,result']
    num_result = ['file_name,result']

    # 在这里用for循环进行每张图片的预测只是一个示例，选手可以根据模型一次读入多张图片并进行预测
    for text_filename in os.listdir(text_path):
        text_img_path = os.path.join(text_path, text_filename)  # 每一张文字图片路径
        text_pred = textOCR(text_img_path)
        text_result.append(text_filename + ',' + text_pred)

# 加载模型
    num_model = load_model('num_model.h5')  # task1的模型
    for num_filename in os.listdir(num_path):
        num_img_path = os.path.join(num_path, num_filename) # 每一张数字图片路径
        num_image = cv2.imread(num_img_path)
        
        num_pred = num_model.predict(num_image) # 这里是model预测结果，选手可以根据自己的情况调整
        # num_pred = '929071' # 而在这里为了给选手进行示例，我们直接指定了一个model预测结果
        num_result.append(num_filename+','+num_pred)

    # ！！！注意这里一定要新建一个submit文件夹，然后把你预测的结果存入这个文件夹中，这两个预测结果文件的命名不可修改，需要严格和下面写出的一致。
    os.mkdir('submit')
    with open('submit/text_sub.csv', 'w') as f:
        f.write('\n'.join(text_result))
    with open('submit/num_sub.csv', 'w') as f:
        f.write('\n'.join(num_result))
    parse_and_zip(result_save_path)


if __name__ == "__main__":
    # ！！！以下内容不允许修改，修改会导致评分出错
    to_pred_dir = sys.argv[1]  # 待预测数据存放路径
    result_save_path = sys.argv[2]  # 预测结果存放路径
    main(to_pred_dir, result_save_path)
