# -*- coding: utf-8 -*-
'''
    引用库及版本：
        python  3.7.0
        opencv-contrib-python 3.4.2.16
        numpy   1.16.3
        scikit-image          0.15.0
        imutils               0.5.2
'''

import imutils
import cv2
import numpy as np
from skimage.measure import compare_ssim
from .util import image_block, imread, remove_status_bar
from .keypointMatching import find_surf,find_sift

'''
## Image_block_ssim
    参数：资源图路径，对比图路径，分块数量
    返回参数：相似判断，分块相似阈值数组，对比生成图（可使用imwrite存储）
    功能：结构相似对比算法，适用于尺寸相同的图片对比

## Image_block_compare
    参数: 资源图路径，对比图路径，分块数量，选择算法（1、sift；2、template；3、surf）
    返回参数：相似判断，分块相似阈值数组
    功能: 
        1、sift：特征对比算法，适用于不同尺寸图片对比,特征较少时匹配不佳
        2、template: 模板匹配算法，适用于相同尺寸图片对比
        3、surf：加速健壮特征算法，适用于不同尺寸图片对比，速度优于sift，特征较少时匹配不佳

## Imread 
    参数：图片路径
    返回参数：cv2图片格式
## Imwrite
    参数：图片存储路径，cv2图片
'''

# SIFT识别特征点匹配，参数设置:
FLANN_INDEX_KDTREE = 0
FLANN = cv2.FlannBasedMatcher(
    {'algorithm': FLANN_INDEX_KDTREE, 'trees': 5}, dict(checks=50))
# SIFT参数: FILTER_RATIO为SIFT优秀特征点过滤比例值(0-1范围，建议值0.4-0.6)
FILTER_RATIO = 0.6
# SIFT参数: SIFT识别时只找出一对相似特征点时的置信度(confidence)
ONE_POINT_CONFI = 0.5
# 图像相似阈值
THRESHOLD = 0.7
THRESHOLD_SURF = 0.60
# 汉明距离阈值
HANMING_DISTANCE = 5


# --------------------------ssim--------------------------#
def compare_ssim_diff(img_src, img_sch):
    '''
    结构相似算法
    :param img_src: 资源图
    :param img_sch: 对比图
    :return: score：相似阈值
             cnts:  不相似矩阵
    '''
    # convert the images to grayscale
    grayA = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img_sch, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    diff_rect = []
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        if w < 10 or h < 10:
            continue
        diff_rect.append([x, y, w, h])
    return score, diff_rect


def image_block_ssim(img_source, img_search, blocks=4):
    '''
    分块结构相似对比
    :param img_sources: 资源图文件路径
    :param img_searchs: 对比图文件路径
    :param blocks: 分块数量
    :return: 对比结果，分块对比阈值，对比结果图像
    '''
    block_similer = 0
    scores = []
    ssim_search = []
    img_sources = image_block(img_source, blocks)
    img_searchs = image_block(img_search, blocks)
    for b_source, b_search in zip(img_sources, img_searchs):
        score, diff_rect = compare_ssim_diff(b_source, b_search)
        scores.append(score)
        if score > 0.99:
            block_similer += 1
        for rect in diff_rect:
            b_search = cv2.rectangle(b_search, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]),
                                     (100, 100, 0), 8)
        ssim_search.append(b_search.copy())
    img_source = np.vstack(img_sources)
    img_search = np.vstack(ssim_search)
    img = np.hstack((img_source, img_search))
    if block_similer == blocks:
        find = True
    else:
        find = False
    return find, scores, img


# --------------------------phash-------------------------#
def classify_pHash(image1, image2):
    image1 = cv2.resize(image1, (32, 32))
    image2 = cv2.resize(image2, (32, 32))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct1 = cv2.dct(np.float32(gray1))
    dct2 = cv2.dct(np.float32(gray2))
    # 取左上角的8*8，这些代表图片的最低频率
    # 在python中进行掩码操作，可以直接这样取出图像矩阵的某一部分
    dct1_roi = dct1[0:8, 0:8]
    dct2_roi = dct2[0:8, 0:8]
    hash1 = getHash(dct1_roi)
    hash2 = getHash(dct2_roi)
    # 计算汉明距离
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num


# 输入灰度图，返回hash
def getHash(image):
    avreage = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


# -------------------------Template-----------------------#
def image_block_template(img_src, img_sch, blocks):
    """
    分块模板匹配
    :param img_src: 资源图
    :param img_sch: 搜索图
    :param blocks: 分块数
    :return: 对比结果，分块阈值，标注图像
    """
    img_schs = image_block(img_sch,blocks)
    img_sch_new = []
    find = blocks
    thresholds = []
    for tpl in img_schs:
        threshold = cal_rgb_confidence(img_src, tpl)
        thresholds.append(threshold)
        if threshold < THRESHOLD:
            p1 = (2,2)
            p2 = (tpl.shape[1]-2,tpl.shape[0]-2)
            cv2.rectangle(tpl, p1, p2 ,(0, 0, 255), 2)
            find -= 1
        img_sch_new.append(tpl)
    img_sch_new = np.vstack(img_sch_new)
    if find == blocks:
        f = True
    else:
        f = False
    return  f, thresholds,img_sch_new


#----------------------rgb-confidence---------------------#
def cal_rgb_confidence(img_src_rgb, img_sch_rgb):
    '''同大小彩图计算相似度.'''
    # BGR三通道心理学权重:
    weight = (0.114, 0.587, 0.299)
    src_bgr, sch_bgr = cv2.split(img_src_rgb), cv2.split(img_sch_rgb)

    # 计算BGR三通道的confidence，存入bgr_confidence:
    bgr_confidence = [0, 0, 0]
    for i in range(3):
        res_temp = cv2.matchTemplate(
            src_bgr[i], sch_bgr[i], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_temp)
        bgr_confidence[i] = max_val

    # 加权可信度
    weighted_confidence = bgr_confidence[0] * weight[0] + \
                          bgr_confidence[1] * weight[1] + bgr_confidence[2] * weight[2]
    return weighted_confidence


def image_block_compare(img_source, img_search, blocks=4, algorithm=1):
    '''
        图像分块对比相似判断
            image_source: 资源图像
            image_search: 对比图像
            blocks: 分块数目 (默认分4块)
            algorithm: 默认1、sift（1、sift ; 2、template; 3、surf）
    '''
    img_sources = image_block(img_source, blocks)
    img_searchs = image_block(img_search, blocks)
    find = 0
    find_data = []

    for b_source, b_search in zip(img_sources, img_searchs):
        # 算法选择
        if algorithm == 1:
            # 算法1
            sift_find = find_sift(b_source, b_search)
            find_data.append(sift_find)
            if sift_find > THRESHOLD:
                find += 1
        elif algorithm == 2:
            # 算法2
            template_find = cal_rgb_confidence(b_source, b_search)
            find_data.append(template_find)
            if template_find > THRESHOLD:
                find += 1
        elif algorithm == 3:
            # 算法3
            surf_find = find_surf(b_source, b_search)
            find_data.append(surf_find)
            if surf_find > THRESHOLD:
                find += 1
        elif algorithm == 4:
            # 算法4
            ssim_find, _ = compare_ssim_diff(b_source, b_search)
            find_data.append(ssim_find)
            if ssim_find > THRESHOLD:
                find += 1

    if find == blocks:
        return True, find_data
    else:
        return False, find_data


def similer(img_src, img_sch, blocks=8):
    """
    图像对比
    :param img_src: 资源图像
    :param img_sch: 对比图像
    :param blocks: 分块数量
    :return: bool(相似判断), dictionaty(对比描述), cv.image(对比图像)
    """
    img_src = imread(img_src)
    img_sch = imread(img_sch)
    img_src , img_sch = remove_status_bar(img_src, img_sch)
    des = {}
    des['src_size'] = img_src.shape[:2]
    des['sch_size'] = img_sch.shape[:2]
    if tuple(img_sch.shape[:2]) == tuple(img_src.shape[:2]):
        # 尺寸相同
        if classify_pHash(img_src, img_sch) > HANMING_DISTANCE:
            des['algorithm'] = 'phash'
            des['blocks'] = 1
            find = False
        else:
            des['algorithm'] = 'ssim'
            des['blocks'] = blocks
            find, data, img = image_block_ssim(img_src, img_sch, blocks)
            des['scores'] = data
    else:
        # 尺寸不同
        # 高比宽
        srch_w = img_src.shape[0] / img_src.shape[1]
        schh_w = img_sch.shape[0] / img_sch.shape[1]
        if srch_w == schh_w:
            # 比例相同，缩小大图
            des['algorithm'] = 'template'
            des['blocks'] = blocks
            if img_src.shape[0] > img_sch.shape[0]:
                rsize = img_sch.shape[:2]
                img_src = cv2.resize(img_src, rsize[::-1])
            else:
                rsize = img_src.shape[:2]
                img_sch = cv2.resize(img_sch, rsize[::-1])
            find, data, img = image_block_template(img_sch, img_src, blocks)
            des['scores'] = data
        else:
            des['algorithm'] = 'temlpate'
            des['blocks'] = blocks
            if img_src.shape[1] > img_sch.shape[1]:
                h = img_sch.shape[1] / img_src.shape[1] * img_src.shape[0]
                img_src = cv2.resize(img_src,(img_sch.shape[1],h) )
            elif img_src.shape[1] < img_sch.shape[1]:
                h = img_src.shape[1] / img_sch.shape[1] * img_sch.shape[0]
            find, data, img = image_block_template(img_sch, img_src,blocks)
            des['scores'] = data
        # else:
        #     des['algorithm'] = 'surf'
        #     des['blocks'] = 1
        #     if srch_w > schh_w:
        #         data.append(find_surf(img_src, img_sch))
        #         find = data[0] > THRESHOLD_SURF
        #     else:
        #         data.append(find_surf(img_sch, img_src))
        #         find = data[0] > THRESHOLD_SURF
        #     des['scores'] = data

    return find, des, img