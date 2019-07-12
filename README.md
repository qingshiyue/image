# image

## imgcv.find

```
def find_tap_in_image(imsrc, imsch):
    """
    读取图像，搜寻位置
    :param imsrc: 大图
    :param imsch: 小图
    :return: 所有识别到的位置([x,y]...[x,y])
    """
    '''
    '''
    return tap
```

## imgcv.similar

```
def similer(img_src, img_sch, blocks=8):
    """
    图像对比
    :param img_src: 资源图像
    :param img_sch: 对比图像
    :param blocks: 分块数量
    :return: bool(相似判断), dictionaty(对比描述), cv.image(对比图像)
    """
    '''
    '''
    return find, des, img
```

## imgcv.ocr

```
def word_in_image(image,find_word,find_type = 'in', withsize = True):
    """
    图片寻找文字,只返回第一个
    :param image: 图像路径
    :param find_type:('in' or 'same')
    :param withsize: 是否需要坐标
    :return: 所有识别字段
    """
    '''
    '''
    return words[0]
```

