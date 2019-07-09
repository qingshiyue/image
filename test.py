from imgcv.ocr import word_in_image,word_in_image_all
from imgcv.find import find_tap_in_image,imread,find_all_template
from imgcv.similar import image_block_template,similer
from imgcv.util import imwrite,show_image,image_add_circle
import cv2

if __name__ == '__main__':
    # word = word_in_image_all('D:\\image\\app\\app_che_cut.png', 'good', withsize = False)
    # assert len(word) >= 1, '未搜寻到good'
    # print(word)
    im2 = 'D:\\image\\app\\cut.jpg'
    im1 = 'D:\\image\\app\\50193538.jpg'
    # im1 = imread(im1)
    # im2 = imread(im2)
    # find,des,img = similer(im1,im2,4)
    # print(find,des)
    # imwrite('D:\\image\\app\\app_che_1_find.png',img)
    # show_image(img)
    # template(im1,im2)
    # im2 = cv2.resize(im1, (1080,1772))
    # imwrite('D:\\image\\app\\app_che_3.png',im2)
    tap = find_tap_in_image(im1,im2)
    print(tap)
    if len(tap)!=0:
        im1 = imread(im1)
        image_add_circle(im1,tuple(tap))
        show_image(im1)