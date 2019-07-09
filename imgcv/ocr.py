# coding:utf-8
""" ocr识别 """

import os
import json
from .client import get_token_baiduAi, PostWithForm
from .util import imread, Image_Base64

AK = 'MAD2nYRkYot6NybAyFUHwRd7'
SK = 'EseGqDv1rd9wAcylLqq2B9gofOuQrmVH'


def ocr_general_basic(image, imagetype):
    """
    ocr 文字识别无坐标版(50000/天)
        image: opencv图片
        imagetype: 图片格式
    """
    Host_Base = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
    img_code = Image_Base64(image, imagetype)
    Param = {"image": img_code}
    Token = get_token_baiduAi(AK, SK)
    res = PostWithForm(Host_Base, Param, Token)
    Data = json.loads(res.text)
    return Data


def ocr_accurate_basic(image, imagetype):
    """
    ocr 文字识别高精度(500/天)
        image: opencv图片
        imagetype: 图片格式
    """
    Host = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
    img_code = Image_Base64(image, imagetype)
    Param = {"image": img_code, "probability": 'true'}
    Token = get_token_baiduAi(AK, SK)
    res = PostWithForm(Host, Param, Token)
    Data = json.loads(res.text)
    return Data


def ocr_general(image, imagetype):
    """
    ocr 文字识别有坐标版(500/天)
        image: opencv图片
        imagetype: 图片格式
    """
    Host = "https://aip.baidubce.com/rest/2.0/ocr/v1/general"
    img_code = Image_Base64(image, imagetype)
    Param = {"image": img_code, "probability": 'true'}
    Token = get_token_baiduAi(AK, SK)
    res = PostWithForm(Host, Param, Token)
    Data = json.loads(res.text)
    return Data


def ocr_accurate(image, imagetype):
    """
    ocr 文字识别高精度含位置信息(50/天)
        image: opencv图片
        imagetype: 图片格式
    """
    Host = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate"
    img_code = Image_Base64(image, imagetype)
    Param = {"image": img_code, "probability": 'true'}
    Token = get_token_baiduAi(AK, SK)
    res = PostWithForm(Host, Param, Token)
    Data = json.loads(res.text)
    return Data


def ocr_image(image, imagetype,algorithm = 'general'):
    """
    ocr 全图文字识别定位文字相对图像比例位置
        image :图像路径
            返回结果样例：'words':'xxx','size': [y0,y1,x0,x1]
    """
    if algorithm == 'general':
        # '文字位置信息'
        Data = ocr_general(image, imagetype)
    elif algorithm == 'general_basic':
        # '文字信息'
        Data = ocr_general_basic(image, imagetype)
    elif algorithm == 'accurate_basic':
        # '高精度文字信息'
        Data = ocr_accurate_basic(image, imagetype)
    elif algorithm == 'accurate':
        # '高精度文字位置信息'
        Data = ocr_accurate(image, imagetype)
    if 'words_result' in Data:
        Data = Data['words_result']
        return Data
    else:
        return None

def image_information(image, algorithm = 'general'):
    """
    文字位置信息提取
    :param Data: ocr response data
    :return: information data
    """
    imagetype = os.path.splitext(image)[-1]
    image = imread(image)
    # Data = ocr_general(image, imagetype)
    Data = ocr_image(image,imagetype,algorithm)
    ocr_data = []
    if algorithm == 'general' or algorithm == 'accurate':
        for words in Data:
            size = words['location']
            x0 = size['left']
            x1 = x0 + size['width']
            y0 = size['top']
            y1 = y0 + size['height']
            words_size = {'words': words['words'], 'size': [y0, y1, x0, x1]}
            ocr_data.append(words_size)
    else:
        ocr_data = Data
    return ocr_data


def image_information_change(image):
    """
    文字位置信息提取并转变坐标
    :param Data: ocr response data
    :return: information data
    """
    imagetype = os.path.splitext(image)[-1]
    image = imread(image)
    image_h , image_w = image.shape[:2]
    Data = ocr_general(image, imagetype)
    if 'words_result' in Data:
        Data = Data['words_result']
    else:
        return None
    ocr_data = []

    for words in Data:
        size = words['location']
        x0 = size['left'] / image_w
        x1 = x0 + size['width'] / image_w
        y0 = size['top'] / image_h
        y1 = y0 + size['height'] / image_h
        words_size = {'words': words['words'], 'size': [y0, y1, x0, x1]}
        ocr_data.append(words_size)
    return ocr_data


def word_in_data_all(data,find_words,find_type = 'in'):
    '''
    关键字是否在页面出现，存在则返回详细内容，坐标
    :param data: {}
    :param find_words: 'str'
    :param find_type: 'str' ('in': 字串， 'same': 完全相同)
    :return: words['words': 'str', 'location': {'width': int, 'top': int, 'left': int, 'height': int}]
    '''
    words = []
    for word in data:
        if find_type == 'in':
            if find_words in word['words']:
                words.append(word)
        elif find_type == 'same':
            if find_words == word['words']:
                words.append(word)
    return words

def word_in_image_all(image, find_word, find_type = 'in', withsize = True):
    """
    图片寻找文字,寻找所有
    :param image: 图像路径
    :param find_word: 寻找字段
    :param find_type: 匹配方式
    :param withsize: 是否需要坐标
    :return: 所有寻找到的信息
    """
    if withsize:
        data = image_information(image)
    else:
        data = image_information(image, 'general_basic')
    words = word_in_data_all(data, find_word,find_type)
    return words

def word_in_image(image,find_word,find_type = 'in', withsize = 'True'):
    """
    图片寻找文字,只返回第一个
    :param image:
    :param find_type:
    :param withsize:
    :return:
    """
    words = word_in_image_all(image,find_word,find_type,withsize)
    return words[0]
