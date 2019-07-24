# image

## imgcv.find(模板匹配)

```
def find_tap_in_image(imsrc, imsch):
    """
    读取图像，搜寻位置
    :param imsrc: 大图
    :param imsch: 小图
    :return: 所有识别到的位置[x,y]
    """
    '''
    '''
    return [x,y]

def find_tap_in_image_all(imsrc, imsch):
    """
    读取图像，搜寻位置
    :param imsrc: 大图
    :param imsch: 小图
    :return: 位置 [(x,y),(x,y)]
    """
```

## imgcv.similar(图像对比)

```
def similer(img_src, img_sch, blocks=8):
    """
    图像对比
    :param img_src: 资源图像
    :param img_sch: 对比图像
    :param blocks: 分块数量
    :return: bool(相似判断), dictionaty(对比描述),opencv image(对比图像)
    """
    '''
    '''
    return find, des, img
```

## imgcv.ocr(图像文字识别)

```
def tap_in_image(image, find_word, find_type = 'in'):
    """
    图片中定位文字位置，只返回第一个搜索到的文字位置
    :param image: 图片
    :param find_word: 待搜素文字
    :param find_type: 搜索类型
    :param withsize: 是否需要坐标
    :return: 坐标 [x,y]
    """


def word_in_image(image,find_word,find_type = 'in', withsize = True):
    """
    图片寻找文字,只返回第一个
    :param image: 图像路径
    :param find_type:('in' or 'same')
    :param withsize: 是否需要坐标
    :return: 包含坐标的字典 {'words': 'str', 'size': [y0, y1, x0, x1]}
    """
    '''
    '''
    return words[0]


def word_in_image_all(image, find_word, find_type = 'in', withsize = True):
    """
    图片寻找文字,寻找所有信息
    :param image: 图像路径
    :param find_word: 寻找字段
    :param find_type: 匹配方式
    :param withsize: 是否需要坐标
    :return: 包含坐标的字典 [{'words': 'str', 'size': [y0, y1, x0, x1]}]
    """
```

## imgcv.util(图像处理工具)

```
def imread(filename):
    """
    根据图片路径，将图片读取为cv2的图片处理格式
    :param filename: 图片地址 url 或 本地文件路径
    :return: opencv img
    """


def imwrite(filename, img):
    """
    写入图片
    :param filename: 路径（本地路径 或 服务器 go httpserver）
    :param img: opencv img
    """


def imwrite_size_in_img(p, img, filename):
    """
    标记find位置在图片中
    :param p: [x,y]坐标点
    :param img: cv.img 或图片地址
    :param filename: 保存路经，仅支持本机路径
    :return: cv.img 标记后的图像
    """


def show_img(img, size = None, wait = 0):
    """
    显示坐标在图片中的位置
    :param size: [x,y] 非必选，不填显示原图
    :param img: cv.img
    :param wait: 显示时间 单位s
    """


def cut_image(image,size):
    """
    裁剪图像
    :param image: 图片路径 或 opencv img
    :param size: 裁剪尺寸,相对坐标 或 绝对坐标[y0,y1,x0,x1]
    :return: 裁剪后的图片 opencv img
    """
```