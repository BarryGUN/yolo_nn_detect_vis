# _*_ coding:utf-8 _*_
# 计算iou

"""
bbox的数据结构为(xmin,ymin,xmax,ymax)--(x1,y1,x2,y2),
每个bounding box的左上角和右下角的坐标
输入：
    bbox1, bbox2: Single numpy bounding box, Shape: [4]
输出：
    iou值
"""
import numpy as np
import cv2

def iou(bbox1, bbox2):
    """
    计算两个bbox(两框的交并比)的iou值
    :param bbox1: (x1,y1,x2,y2), type: ndarray or list
    :param bbox2: (x1,y1,x2,y2), type: ndarray or list
    :return: iou, type float
    """
    if type(bbox1) or type(bbox2) != 'ndarray':
        bbox1 = np.array(bbox1)
        bbox2 = np.array(bbox2)

    assert bbox1.size == 4 and bbox2.size == 4, "bounding box coordinate size must be 4"
    xx1 = np.max((bbox1[0], bbox2[0]))
    yy1 = np.min((bbox1[1], bbox2[1]))
    xx2 = np.max((bbox1[2], bbox2[2]))
    yy2 = np.min((bbox1[3], bbox2[3]))
    bwidth = xx2 - xx1
    bheight = yy2 - yy1
    area = bwidth * bheight  # 求两个矩形框的交集
    union = (bbox1[2] - bbox1[0])*(bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0])*(bbox2[3] - bbox2[1]) - area  # 求两个矩形框的并集
    iou = area / union

    return iou


if __name__=='__main__':
    rect1 = (461, 97, 599, 237)
    # (top, left, bottom, right)
    rect2 = (522, 127, 702, 257)
    iou_ret = round(iou(rect1, rect2), 3) # 保留3位小数
    print(iou_ret)

    # Create a black image
    img=np.zeros((720,720,3), np.uint8)
    cv2.namedWindow('iou_rectangle')
    """
    cv2.rectangle 的 pt1 和 pt2 参数分别代表矩形的左上角和右下角两个点,
    coordinates for the bounding box vertices need to be integers if they are in a tuple,
    and they need to be in the order of (left, top) and (right, bottom). 
    Or, equivalently, (xmin, ymin) and (xmax, ymax).
    """
    cv2.rectangle(img,(461, 97),(599, 237),(0,255,0),3)
    cv2.rectangle(img,(522, 127),(702, 257),(0,255,0),3)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'IoU is ' + str(iou_ret), (341,400), font, 1,(255,255,255),1)
    cv2.imshow('iou_rectangle', img)
    cv2.waitKey(0)

