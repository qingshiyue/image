# -*- coding: utf-8 -*-
import cv2


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
# --------------------------sift--------------------------#

def _init_sift():
    '''
    Make sure that there is SIFT module in OpenCV.
    '''
    if cv2.__version__.startswith("3."):
        # OpenCV3.x, sift is in contrib module;
        # you need to compile it seperately.
        try:
            sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=10)
        except sift:
            print("to use SIFT, you should build contrib with opencv3.0")
            raise Exception(
                "There is no SIFT module in your OpenCV environment !")
    else:
        # OpenCV2.x, just use it.
        sift = cv2.SIFT(edgeThreshold=10)

    return sift


def find_sift(img_source, img_search, ratio=FILTER_RATIO):
    '''
    基于sift进行图像识别
    :param img_source: 资源图
    :param img_search: 对比图
    :param ratio: 优秀特征筛选阈值
    :return: 相似阈值
    '''
    # 检测图片是否正常
    if not check_image_valid(img_source, img_search):
        raise Exception(img_source, img_search, "空图像")
    # 获取特征点集，匹配特征
    kp1, kp2, matches = _get_key_points(img_source, img_search, ratio)
    print(len(kp1), len(kp2), len(matches))
    # 关键点匹配数量，匹配掩码
    (matchNum, matchesMask) = getMatchNum(matches, ratio)
    print(matchNum, len(matchesMask))
    # 关键点匹配置信度
    matcheRatio = matchNum / len(matchesMask)
    if matcheRatio >= 0 and matcheRatio <= 1:
        return matcheRatio
    else:
        raise Exception("SIFT Score Error", matcheRatio)


# ---------------------------surf--------------------------#
def _init_surf():
    '''
    Make sure that there is SURF module in OpenCV.
    '''
    if cv2.__version__.startswith("3."):
        # OpenCV3.x, sift is in contrib module;
        # you need to compile it seperately.
        try:
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        except surf:
            print("to use SURF, you should build contrib with opencv3.0")
            raise Exception(
                "There is no SURF module in your OpenCV environment !")
    else:
        # OpenCV2.x, just use it.
        surf = cv2.SURF(hessianThreshold=100)

    return surf


def find_surf(img_source, img_search, ratio=FILTER_RATIO):
    '''
    基于surf进行图像识别
    :param img_source: 资源图
    :param img_search: 待对比图
    :param ratio: 优秀特征点过滤比例值
    :return: 相似阈值
    '''
    # 检测图片是否正常
    if not check_image_valid(img_source, img_search):
        raise Exception(img_source, img_search, "空图像")
    # 获取特征点集，匹配特征
    surf = _init_surf()
    kp1, des1 = surf.detectAndCompute(img_source, None)
    kp2, des2 = surf.detectAndCompute(img_search, None)
    # When apply knnmatch , make sure that number of features in both test and
    # query image is greater than or equal to number of
    #       nearest neighbors in knn match.
    if len(kp1) < 2 or len(kp2) < 2:
        raise Exception("Not enough feature points in input images !")
    # 匹配两个图片中的特征点集，k=2表示每个特征点取出2个最匹配的对应点:
    matches = FLANN.knnMatch(des1, des2, k=2)
    print(len(matches))
    # 关键点匹配数量，匹配掩码
    (matchNum, matchesMask) = getMatchNum(matches, ratio)
    print(matchNum, len(matchesMask))
    # 关键点匹配置信度
    matcheRatio = matchNum / len(matchesMask)
    if matcheRatio >= 0 and matcheRatio <= 1:
        return matcheRatio
    else:
        raise Exception("SURF Score Error", matcheRatio)


def _get_key_points(im_source, im_search, ratio):
    ''' 根据传入图像，计算所有的特征点并匹配特征点对 '''
    # 初始化sift算子
    sift = _init_sift()
    # 获取特征点集
    kp_sch, des_sch = sift.detectAndCompute(im_search, None)
    kp_src, des_src = sift.detectAndCompute(im_source, None)
    # When apply knnmatch , make sure that number of features in both test and
    # query image is greater than or equal to number of
    #       nearest neighbors in knn match.
    if len(kp_sch) < 2 or len(kp_src) < 2:
        raise Exception("Not enough feature points in input images !")

    # 匹配两个图片中的特征点集，k=2表示每个特征点取出2个最匹配的对应点:
    matches = FLANN.knnMatch(des_sch, des_src, k=2)
    return kp_sch, kp_src, matches


def check_image_valid(im_source, im_search):
    '''Check if the input images valid or not.'''
    if (im_source is not None and im_source.any() and
            im_search is not None and im_search.any()):
        return True
    else:
        return False


def getMatchNum(matches, ratio):
    '''返回特征点匹配数量和匹配掩码'''
    matchesMask = [[0, 0] for i in range(len(matches))]
    matchNum = 0
    for i, (m, n) in enumerate(matches):
        # 将距离比率小于ratio的匹配点删选出来
        if m.distance < ratio * n.distance:
            matchesMask[i] = [1, 0]
            matchNum += 1
    return (matchNum, matchesMask)