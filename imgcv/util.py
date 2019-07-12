# coding:utf-8
''' 图像处理 '''

import sys
import os
import re
import base64
import numpy as np
import requests
import cv2
from six import PY3

def check_image_valid(im_source, im_search):
    '''Check if the input images valid or not.'''
    if im_source is not None and im_source.any() and im_search is not None and im_search.any():
        return True
    else:
        return False


def img_mat_rgb_2_gray(img_mat):
    """
    Turn img_mat into gray_scale, so that template match can figure the img data.
    "print(type(im_search[0][0])")  can check the pixel type.
    """
    assert isinstance(img_mat[0][0], np.ndarray), "input must be instance of np.ndarray"
    return cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)


def imread(filename):
    '''根据图片路径，将图片读取为cv2的图片处理格式.'''
    if isinstance(filename, str):
        if re.match(r"^https?://", filename):
            resp = requests.get(filename)
            im = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_UNCHANGED)
            return im
        else:
            if not os.path.isfile(filename):
                raise Exception("File not exist: " ,filename)
            if PY3:
                img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            else:
                filename = filename.encode(sys.getfilesystemencoding())
                img = cv2.imread(filename, 1)
            return img
    else:
        raise Exception("not a url or file path", filename)


def imwrite(filename, img):
    """
    写入图片到本地路径
    :param filename: 路径
    :param img: cv图片数据
    """
    imagetype = os.path.splitext(filename)[-1]
    if PY3:
        cv2.imencode('.'+ imagetype, img)[1].tofile(filename)
    else:
        filename = filename.encode(sys.getfilesystemencoding())
        cv2.imwrite(filename, img)


def image_block(Img, blocks=4):
    """
    图像分块
    :param Img: 待分块cv图像
    :param blocks: 分块数量
    :return: images[] 图像分块数组
    """
    #分块 size
    img_b_h = int(Img.shape[0] / blocks)
    size_block = np.array([img_b_h, img_b_h, 0, 0])
    size = np.array([0,img_b_h , 0, Img.shape[1]])
    # 图像分块
    Img_blocks = []
    while size[1] <= Img.shape[0]:
        # 仅分一块
        if blocks == 1:
            size[1] = Img.shape[0]
        # 资源图分块
        Img_blocks.append(cut_image_base(Img, size))
        # 更新分块 size
        size += size_block
        # 最后一个分块
        if (size[1] > Img.shape[0] - img_b_h
                and size[1] <= Img.shape[0]):
            size[1] = Img.shape[0]
    return Img_blocks


def cut_image(image,size):
    '''
    裁剪图像
        image:图片路径
        size：裁剪尺寸,相对坐标[y0,y1,x0,x1]
    '''
    img= imread(image)
    #读取图像
    if img is None or not img.any():
        #判断空图片
        raise Exception("空图像",image)
    shape = img.shape
    #图像尺寸,通道
    if size[0] >= size[1] or size[2] >= size[3]:
        #图像重置尺寸是否合理
        raise Exception("y1 应大于 y0 ，x1 应大于 x0",size)
    for i in size:
        if i < 0.0 or i > 1.0:
            #坐标参数合法检查
            raise Exception("Size shoud be 0~1",i,size)
    image = img[int(size[0]*shape[0]):int(size[1]*shape[0]), int(size[2]*shape[1]):int(size[3]*shape[1])]    #裁剪图像
    return image


def cut_image_base(img, size):
    '''
    裁剪图像根据实际尺寸
        image:opencv读取的图片
        size：裁剪尺寸,坐标[y0,y1,x0,x1]
    '''
    # 读取图像
    if img is None or not img.any():
        # 判断空图片
        raise Exception("空图像")
    image = img[size[0]:size[1], size[2]:size[3]]
    return image


def remove_status_bar(img1,img2):
    """
    去除手机状态栏
    :param img1: 图1
    :param img2: 图2
    :return: 图1，图2
    """
    size = [100, img1.shape[0], 0, img1.shape[1]]
    img1 = cut_image_base(img1, size)
    size = [100, img2.shape[0], 0, img2.shape[1]]
    img2 = cut_image_base(img2, size)
    return img1, img2

def Interval(Num,Min,Max):
    ''' 判断Num是否在Min~Max区间内 '''
    if Num < Min or Num >Max:
        return False
    else:
        return True


def cut_image_coo(img,size):
    '''
    裁剪图像
        image:图片
        size：裁剪尺寸,实际坐标[x,y,w,h]
    '''
    shape = img.shape   #图像尺寸,通道
    if not Interval(size[0],0,shape[1]):
        raise Exception("size[0] 坐标越界",size)
    if not Interval(size[1],0,shape[0]):
        raise Exception("size[1] 坐标越界",size)
    if not Interval(size[0]+size[2],0,shape[1]):
        raise Exception("size[0] + size[2] 坐标越界",size)
    if not Interval(size[1]+size[3],0,shape[0]):
        raise Exception("size[1] + size[3] 坐标越界",size)
    image = img[int(size[1]):int((size[1]+size[3])), int(size[0]):int((size[0]+size[2]))]    #裁剪图像
    return image


def show_image(img):
    '''
    显示图像
    '''
    # 窗口设置
    cv2.namedWindow("image", 0);
    cv2.resizeWindow("image", 540, 960);
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_add_circle(img, p):
    """
    根据坐标在图中标圈
    :param img: 图
    :param p: 坐标
    :return: img
    """
    cv2.circle(img, p , 50 , (0,0,255),4)
    return img



def Image_Base64(image,imagetype):
    '''
    base64转码
    '''
    img = cv2.imencode(imagetype,image)[1]
    img_code = str(base64.b64encode(img))[2:-1]
    return img_code