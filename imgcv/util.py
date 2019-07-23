# coding:utf-8
''' 图像处理 '''

import sys
import os
import re
import base64
import numpy as np
import requests
import cv2
from PIL import Image
from six import PY3


def size_mid_point(size):
    """
    ocr 'location' 转化成 [x,y]
    :param location:
    :return: [x,y]
    """
    x = int((size[2] + size[3]) / 2)
    y = int((size[0] + size[1]) / 2)
    if x > 0 and y > 0:
        return [x, y]
    else:
        raise Exception("location 解析结果不可信", [x,y])

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
    """
    根据图片路径，将图片读取为cv2的图片处理格式
    :param filename: 图片地址 url 或 本地文件路径
    :return: opencv img
    """
    imagetype = os.path.splitext(filename)[-1]
    if imagetype != '.jpg' and imagetype != '.png':
        # 检查图片格式是否支持
        raise Exception('write file must be jpg ort png',filename,imagetype)
    if isinstance(filename, str):
        if re.match(r"^https?://", filename):
            resp = requests.get(filename)
            im = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
            return im
        else:
            if not os.path.isfile(filename):
                raise Exception("File not exist: " ,filename)
            if PY3:
                img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                filename = filename.encode(sys.getfilesystemencoding())
                img = cv2.imread(filename, 1)
            return img
    else:
        print(filename)
        raise Exception("not a url or file path %s", filename)


def imwrite(filename, img):
    """
    写入图片
    :param filename: 路径（本地路径 或 服务器 go httpserver）
    :param img: cv图片数据
    """
    imagetype = os.path.splitext(filename)[-1]
    if imagetype != '.jpg' and imagetype != '.png':
        # 检查图片格式是否支持
        raise Exception('write file must be jpg ort png',filename,imagetype)
    if isinstance(filename, str):
        if re.match(r"https?://",filename):
            # filename为http链接则上传服务
            file = re.findall(r'[^\\/:*?"<>|\r\n]+$', filename)
            host = filename.split(file[-1])[0]
            r, buf = cv2.imencode('.' + imagetype, img)
            bytes_image = Image.fromarray(np.uint8(buf)).tobytes()
            # requst body
            files = {
                'file': (file[-1], bytes_image),
            }
            # post
            response = requests.post(host, files = files)
            # check response code
            if response.status_code != 200:
                raise Exception('upload fail', response)
        else:
            # 根据路径写到本地
            if PY3:
                cv2.imencode('.'+ imagetype, img)[1].tofile(filename)
            else:
                filename = filename.encode(sys.getfilesystemencoding())
                cv2.imwrite(filename, img)


def image_block(Img, blocks=4):
    """
    图像分块
    :param Img: 待分块cv图像 或 图像路径
    :param blocks: 分块数量 最多分16块
    :return: images[] 图像分块数组
    """
    if isinstance(Img, str):
        if os.path.exists(Img):
            Img = imread(Img)
        else:
            raise Exception('file does not exist', Img)
    #读取图像
    if Img is None or not Img.any():
        #判断空图片
        raise Exception("空图像",Img)
    #分块 size
    if blocks > 16:
        print("blocks should be less than 16 ")
        blocks = 16
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
        if (size[1] > Img.shape[0] - img_b_h) and \
                (size[1] <= Img.shape[0]):
            size[1] = Img.shape[0]
    return Img_blocks


def cut_image(image,size):
    """
    裁剪图像
    :param image: 图片路径 或 opencv img
    :param size: 裁剪尺寸,相对坐标 或 绝对坐标[y0,y1,x0,x1]
    :return: 裁剪后的图片 opencv img
    """
    if isinstance(image, str):
        if os.path.exists(image):
            image = imread(image)
        else:
            raise Exception('file does not exist', image)
    #读取图像
    if image is None or not image.any():
        #判断空图片
        raise Exception("空图像",image)
    shape = image.shape
    #图像尺寸,通道
    if size[0] >= size[1] or size[2] >= size[3]:
        #图像重置尺寸是否合理
        raise Exception("y1 应大于 y0 ，x1 应大于 x0",size)

    if (0 < size[0] < shape[0] and 0 < size[1] < shape[0]) and \
            (0 < size[2] < shape[1] and 0 < size[3] < shape[1]):
        if size[0] < 1:
            for i in size:
                if i < 0.0 or i > 1.0:
                    # 坐标参数合法检查
                    raise Exception("Size shoud be 0~1", i, size)
            image = image[int(size[0]*shape[0]):int(size[1]*shape[0]),
                    int(size[2]*shape[1]):int(size[3]*shape[1])]    #裁剪图像
        else:
            image = cut_image_base(image, size)
    else:
        raise Exception("size out of image size", size, shape)
    return image


def cut_image_base(img, size):
    """
    裁剪图像根据实际尺寸
    :param img: opencv读取的图片
    :param size: 裁剪尺寸,坐标[y0,y1,x0,x1]
    :return: 裁剪后的opencv图片
    """
    # 读取图像
    if img is None or not img.any():
        # 判断空图片
        raise Exception("空图像")

    image = img[size[0]:size[1], size[2]:size[3]]
    return image


def remove_status_bar(img1, img2, cut = 100):
    """
    图像去除部分顶部默认去除100px
    :param img1: 图1
    :param img2: 图2
    :return: img1，img2 类型：opencv img
    """
    if img1.shape[0] < cut or img2.shape[0] < cut:
        raise Exception("截除尺寸大于图像尺寸")
    size = [cut, img1.shape[0], 0, img1.shape[1]]
    img1 = cut_image_base(img1, size)
    size = [cut, img2.shape[0], 0, img2.shape[1]]
    img2 = cut_image_base(img2, size)
    return img1, img2


def cut_image_coo(img, size):
    """
    根据size裁剪图像
    :param img: 图片，类型 opencv img 或 图像地址
    :param size:裁剪尺寸,实际坐标[x,y,w,h]
    :return: 裁剪后的图片，类型 opencv img
    """
    if isinstance(img, str):
        print('open file'+ img)
        if os.path.exists(img):
            img = imread(img)
        else:
            raise Exception('file does not exist', img)
    if img is None and not img.any():
        raise Exception("empty image")

    shape = img.shape   #图像尺寸,通道
    if not 0 < size[0] < shape[1]:
        raise Exception("size[0] 坐标越界",size)
    if not 0 < size[1] < shape[0]:
        raise Exception("size[1] 坐标越界",size)
    if not 0 < size[0] + size[2] < shape[1]:
        raise Exception("size[0] + size[2] 坐标越界",size)
    if not 0 < size[1] + size[1] + size[3] < shape[0]:
        raise Exception("size[1] + size[3] 坐标越界",size)
    image = img[int(size[1]):int((size[1]+size[3])), int(size[0]):int((size[0]+size[2]))]    #裁剪图像
    return image

def imwrite_size_in_img(p, img, filename):
    """
    标记find位置在图片中
    :param p: [x,y]坐标点
    :param img: cv.img 或图片地址
    :param filename: 保存路经，仅支持本机路径
    :return: cv.img 标记后的图像
    """
    if isinstance(img, str):
        if os.path.exists(img):
            img = imread(img)
        else:
            raise Exception('file does not exist', img)
    if 0 < p[0] < img.shape[1] and 0 < p[1] < img.shape[0]:
        if p[0] < 1 and p[1] <1:
            # 相对坐标
            p = [p[0] * img.shape[1], p[1] * img.shape[0]]
        img = image_add_circle(img, tuple(p))
        imwrite(filename, img)
    else:
        raise Exception('pint out of img size')


################### show image #######################
def show_img(img, size = None, wait = 0):
    """
    显示坐标在图片中的位置
    :param size: [x,y]
    :param img: cv.img
    :param wait: 显示时间 单位s
    """
    if isinstance(img, str):
        if os.path.exists(img):
            img = imread(img)
        else:
            raise Exception('file does not exist', img)
    if size != None:
        img = image_add_circle(img, tuple(size))
    show_image(img, wait)


def show_image(img, wait = 0):
    '''
    显示图像
    '''
    if isinstance(img, str):
        if os.path.exists(img):
            img = imread(img)
        else:
            raise Exception('file does not exist', img)
    h, w = img.shape[:2]
    if img.shape[1] > 540 and img.shape[0] > 960:
        w = img.shape[1] / 2
        h = img.shape[0] / 2
    # 窗口设置
    cv2.namedWindow("image", 0);
    cv2.resizeWindow("image", int(w), int(h));
    cv2.imshow('image',img)
    cv2.waitKey(wait * 1000)
    cv2.destroyAllWindows()


def image_add_circle(img, p):
    """
    根据坐标在图中标圈
    :param img: 图
    :param p: 坐标
    :return: img
    """
    if img is None and not img.any():
        raise Exception("empty image")
    else:
        if  0 < p[0] < img.shape[1] and 0 < p[1] < img.shape[0]:
            cv2.circle(img, p , 50 , (0,0,255),4)
            return img
        else:
            raise Exception('pion not in image size',p, img.shape[:2])

############### image 2 base64 ########################
def Image_Base64(image,imagetype):
    '''
    image 转 base64
    '''
    img = cv2.imencode(imagetype,image)[1]
    if img is None and not img.any():
        raise Exception("empty image")
    else:
        img_code = str(base64.b64encode(img))[2:-1]
        return img_code