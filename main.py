# This is a sample Python script.
import json
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import random


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def cp_file():
    import pickle as cp
    import numpy as np
    import cv2
    col_file = '/data1/lily/SRNet-Datagen/Synthtext/data/colors.cp'

    with open(col_file, 'rb') as f:
        colorsRGB = cp.load(f, encoding='latin1')
    ncol = colorsRGB.shape[0]
    colorsLAB = np.r_[colorsRGB[:, 0:3], colorsRGB[:, 6:9]].astype(np.uint8)
    colorsLAB = np.squeeze(cv2.cvtColor(colorsLAB[None, :, :], cv2.COLOR_RGB2Lab))


def temp():
    import cv2
    import numpy as np

    def cv_draw(img_path, pos):
        # 1.读取图片
        image = cv2.imread(img_path)
        # 2.获取标签
        # 边框格式　bbox = [xl, yl, xr, yr]
        # bbox1 = [45, 35, 143, 73]
        bbox1 = pos
        # label1 = 'man'

        # bbox2 = [100, 80, 248, 334]
        # label2 = 'woman'

        # 设置字体格式及大小
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 获取label长宽
        # label_size1 = cv2.getTextSize(label1, font, 1, 2)
        # label_size2 = cv2.getTextSize(label2, font, 1, 2)

        # 设置label起点
        # text_origin1 = np.array([bbox1[0], bbox1[1] - label_size1[0][1]])
        # text_origin2 = np.array([bbox2[0], bbox2[1] - label_size2[0][1]])

        cv2.rectangle(image, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                      color=(0, 0, 255), thickness=1)
        # cv2.rectangle(image, tuple(text_origin1), tuple(text_origin1 + label_size1[0]),
        #               color=(0, 0, 255), thickness=-1)  # thickness=-1 表示矩形框内颜色填充
        # 1为字体缩放比例，2表示自体粗细
        # cv2.putText(image, label1, (bbox1[0], bbox1[1] - 5), font, 1, (255, 255, 255), 2)

        # cv2.rectangle(image, (bbox2[0], bbox2[1]), (bbox2[2], bbox2[3]),
        #               color=(0, 255, 0), thickness=2)
        # cv2.rectangle(image, tuple(text_origin2), tuple(text_origin2 + label_size2[0]),
        #               color=(0, 255, 0), thickness=-1)  # thickness=-1 表示矩形框内颜色填充
        # cv2.putText(image, label2, (bbox2[0], bbox2[1] - 5), font, 1, (255, 255, 255), 2)

        cv2.imwrite('cv_drwn.jpg', image)

    # f1 = open('bounding_box1.json', 'r')
    # f2 = open('bounding_box2.json', 'r')
    # content1 = f1.read()
    # content2 = f2.read()
    # a = json.loads(content1)
    # b = json.loads(content2)
    # f1.close()
    # f2.close()
    # cv_draw(img_path='./temp/temp3.png')
    cv_draw(img_path='/home/lily/SRNet-datagen/i_o/0002.png', pos=[122-1, 7-1, 151+1, 39+1])
    # print(b['00'][1])


def gen_chars():
    import numpy as np
    import string
    import re
    txt_root = 'Synthtext/data/newsgroup/'
    txt_path = txt_root + np.random.choice(os.listdir(txt_root))
    txt_content = open('Synthtext/data/newsgroup/猫咪分局.txt', 'r', encoding='utf-8').readlines()
    sum_content = ''
    for i in txt_content:
        # te = re.sub(r'[^\w\s]','',i.replace('\n', ''))
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        chinese = re.sub(pattern, '', i)
        sum_content += chinese

    sum_content = list(sum_content)
    num = random.randint(1, 4)
    choice_char = np.random.choice(np.array(sum_content), size=num, replace=False)
    choice_char = list(choice_char)
    choice_char = ''.join(choice_char)
    print(choice_char)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    temp()

    print_hi('PyCharm')

