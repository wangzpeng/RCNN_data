#-*-coding=utf-8-*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
#from skimage.measure import label,regionprops
#全局阈值
def threshold_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
    #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    return binary
    #print("threshold value %s"%ret)
    #cv.namedWindow("binary0", cv.WINDOW_NORMAL)
    #cv.imshow("binary0", binary)


#局部阈值
def local_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
    #自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    binary =  cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
    #cv.namedWindow("binary1", cv.WINDOW_NORMAL)
    #cv.imshow("binary1", binary)
    cv2.imwrite('/home/wang/tf-faster-rcnn/roi_results/binary1.jpg',binary)

#用户自己计算阈值
def custom_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    print("mean:",mean)
    ret, binary =  cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    #cv.namedWindow("binary2", cv.WINDOW_NORMAL)
    #cv.imshow("binary2", binary)
    cv2.imwrite('/home/wang/tf-faster-rcnn/roi_results/binary2.jpg',binary)

src = cv2.imread('../demo_result/roiA_20170313123156_0.jpg')
#cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
#cv.imshow('input_image', src)
#rawimg = src - 246

fig = plt.figure()
#fig.add_subplot(2,3,1)
#plt.title("raw image")
#plt.imshow(src)
#local_threshold(src)
#custom_threshold(src)
binary_image=threshold_demo(src)
#cv2.imwrite('/home/wang/tf-faster-rcnn/roi_results/binary0.jpg',binary_image)
#fig.add_subplot(2,3,2)
#plt.title("binary image")
#grayscaleimg = cv2.cvtColor(rawimg,cv2.COLOR_BGR2GRAY)
#plt.imshow(binary_image)

# counting non-zero value by row , axis y
row_nz = []
for row in binary_image.tolist():
    row_nz.append(len(row) - row.count(0))
#fig.add_subplot(1,2,1)
#plt.title("non-zero values on y (by row)")
#plt.plot(row_nz)

# counting non-zero value by column, x axis
col_nz = []
for col in binary_image.T.tolist():
    col_nz.append(len(col) - col.count(0))
#fig.add_subplot(1,2,2)
#plt.title("non-zero values on x (by col)")
#plt.plot(col_nz)

##### start split
# first find left and right boundary of x (col)
#fig.add_subplot(2,3,5)
#plt.title("x boudary deleted")
left_x = 0
for i,x in enumerate(col_nz):
    if x != 0:
        left_x = i
        break
right_x = 0
for i,x in enumerate(col_nz[::-1]):
    if x!=0:
        right_x = len(col_nz) - i
        break
sliced_x_img = binary_image[:,left_x:right_x]
#plt.imshow(sliced_x_img)
#cv2.namedWindow("binary0", cv2.WINDOW_NORMAL)
#cv2.imshow("binary0", sliced_x_img)
#cv2.imwrite('./result/binary.jpg',sliced_x_img)
# then we find up and down boundary of every digital (y, on row)
#fig.add_subplot(2,3,6)
#plt.title("y boudary deleted")
row_boundary_list = []
record = False
for i,x in enumerate(row_nz[:-1]):
    if (row_nz[i] == 0 and row_nz[i+1] != 0) or row_nz[i] != 0 and row_nz[i+1] == 0:
        row_boundary_list.append(i+1)
img_list = []
xl = [ row_boundary_list[i:i+2] for i in range(0,len(row_boundary_list),2) ]
for x in xl:
    img_list.append( sliced_x_img[x[0]:x[1],:] )
# del invalid image
#img_list = [ x for x in img_list if x.shape[0] < 5 ]
# show image
fig = plt.figure()
for i,img in enumerate(img_list):
    #fig.add_subplot(3,4,i+1)
    #plt.imshow(img)
    #plt.imsave("/home/wang/tf-faster-rcnn/tools/result/%s.jpg"%i,img)
    cv2.imwrite('/home/wang/CNN/data/test/%s.bmp'%i,img)
#plt.show()


#cv.namedWindow("binary0", cv.WINDOW_NORMAL)
#cv.imshow("binary0", binary_image)
#label_image=label(binary_image,connectivity=2)
#print(label_image)

#cv2.waitKey(0)
#cv2.destroyAllWindows()


