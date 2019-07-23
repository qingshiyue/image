from adb.adb_shell import *
from imgcv.find import find_tap_in_image
from imgcv.similar import similer
from imgcv.ocr import word_in_image,tap_in_image
from imgcv.util import *
from time import sleep
def shangxiang():

    host = r'http://10.6.53.25:8000/image/shangxiang/'
    activity = r'com.yiwang/com.yiwang.LoadingActivity'
    print("加载app")
    adb_start_activity(activity)
    sleep(12)

    print("点击搜索框")
    cut = 'cut.jpg'
    adb_screencap(cut)
    # icon = host + 'search_icon.jpg'
    # tap = find_tap_in_image(cut,icon)

    # 根据文字寻找搜索框，运行报错可更换文字
    tap = tap_in_image(cut,'感冒灵颗粒')
    print(tap)
    imwrite_size_in_img(tap, imread(cut), 'touch1.jpg')
    show_image('touch1.jpg', 2)

    print("输入文本974752")
    #运行报错，请按adb_input_text中描述安装虚拟键盘
    adb_tap(tap)
    adb_input_text('974752')

    print("点击搜索")
    adb_screencap(cut)
    word = word_in_image(cut,'搜索')
    tap = size_mid_point(word['size'])
    imwrite_size_in_img(tap, imread(cut), 'touch2.jpg')
    show_image('touch2.jpg', 2)
    adb_tap(tap)
    sleep(2)

    print("比较搜索页面")
    adb_screencap(cut)
    src = host + 'search_haipu.jpg'
    find, dsc , img = similer(src,cut)
    print(find,dsc)
    if dsc != None:
        imwrite('diff_test0.jpg', img)
        show_image(img,2)

    src = host + 'search_haipu_new.jpg'
    find_n, dsc, img = similer(src, cut)
    if dsc != None:
        imwrite('diff_test1.jpg', img)
        show_image(img, 2)

    src = host + 'search_haipu_xs.jpg'
    find_x, dsc, img = similer(src, cut)
    if dsc != None:
        imwrite('diff_test1.jpg', img)
        show_image(img, 2)
    assert find or find_n or find_x == True

    print("切换大图比较搜索页")
    word = word_in_image(cut, '大图')
    tap = size_mid_point(word['size'])
    imwrite_size_in_img(tap, imread(cut), 'touch3.jpg')
    show_image('touch3.jpg',2)
    assert len(tap) >= 1
    adb_tap(tap)
    adb_screencap(cut)

    print("点击商品")
    icon = host + 'ziyinghaipu.jpg'
    tap = find_tap_in_image(cut, icon)
    assert len(tap) >= 1
    imwrite_size_in_img(tap, imread(cut), 'touch4.jpg')
    show_image('touch4.jpg', 2)
    adb_tap(tap)


    sleep(2)
    os.system('adb shell am force-stop com.yiwang')




if __name__ == '__main__':
    shangxiang()