#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import cv2
# import numpy as np
# from PIL import Image
# from utils import compute_iou, plot_boxes_on_image,Kanchors, load_gt_boxes, compute_regression, decode_output


# In[3]:


import cv2
import numpy as np
import tensorflow as tf

grid_h = 45
grid_w = 60

# # original paper anchors
# anchors = np.array([[100.0, 100.0], [300.0, 300.0], [500.0, 500.0],
                   # [200.0, 100.0], [370.0, 185.0], [440.0, 220.0],
                   # [100.0, 200.0], [185.0, 370.0], [220.0, 440.0]])
"""
Kanchors 中包含着 9 个预测框的宽度和长度（这是经过 kmeans 算法计算过的结果）。カツ
"""
# k-means anchors (recommend)
Kanchors = np.array([[ 74., 149.],
                   [ 34., 149.],
                   [ 86.,  74.],
                   [109., 132.],
                   [172., 183.],
                   [103., 229.],
                   [149.,  91.],
                   [ 51., 132.],
                   [ 57., 200.]], dtype=np.float32)

def compute_iou(boxes1, boxes2):
    """(xmin, ymin, xmax, ymax)
    (xmin, ymin, xmax, ymax)
    ボックス1の形状: [-1, 4]、ボックス2の形状: [-1, 4]
    IOU の値を計算するには、compute_iou() 関数を使用します。
    すなわち、真の検出ボックスと予測検出ボックス（またはもちろん、任意の2つの検出ボックス）の連結された領域上の交点の領域。
    この値が高ければ高いほど、この予測箱は実物の箱に近くなります。 以下の画像で表現されています。
    ここに画像の説明を挿入してください] 
    boxes1 shape:  [-1, 4], boxes2 shape: [-1, 4]
    compute_iou() 函数用来计算 IOU 值，
    即真实检测框与预测检测框（当然也可以是任意两个检测框）的交集面积比上它们的并集面积，
    这个值越大，代表这个预测框与真实框的位置越接近。用下面这个图片表示：
    在这里插入图片描述カツ
               (xmin1, ymin1)
               ....................................                             
              .O                                   O                                    
              .O                                  .O                                    
              .O                                  .O                                    
              .O                                  .O                                    
              .O                                  .O                                    
              .O                                  .O                                    
              .O                                  .O                                    
              .O                                  .O                                    
              .O         (xmin2, ymin2)           .O                                    
              .O             0 ..................  Z................... ..             
              .O            .O                    .O                     .              
              .O            .O                    .O                     .              
              .O            .O                    .O                     .              
              .O            .O                    .O                     .              
              .O            .O                    .O                     .              
              .O            .O                    .O                     .              
              .O            .O                    .O                     .              
              . ............ 0.................... .                     .              
                            .O                      (xmax1, ymax1)       .              
                            .O                                           .              
                            .O                                           .              
                            .O                                           .              
                            .O                                           .              
                            .O                                           .              
                            .O                                           .              
                            .O                                           .              
                            .O                                           .              
                            .O                                           .              
                            .O                                           .              
                            .Z............................................  (xmax2, ymax2)
    那么，left_up=[xmin2, ymin2]；right_down=[xmax1, ymax1]。
    之后求出来这两个框的交集面积和并集面积，进而得到这两个框的 IOU 值。
    如果说得到的 IOU 值大于设置的正阈值，那么我们称这个预测框为正预测框（positive anchor），
    其中包含着检测目标；如果说得到的 IOU 值小于于设置的负阈值，
    那么我们称这个预测框为负预测框（negative anchor），其中包含着背景。
    カツ
    """
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2], )
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_wh = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union_area = boxes1_area + boxes2_area - inter_area
    ious = inter_area / union_area
    return ious

def plot_boxes_on_image(show_image_with_boxes, boxes, color=[0, 0, 255], thickness=2):
    """
    plot_boxes_on_image() 函数的输入有两个，
    分别是：需要被画上检测框的原始图片以及检测框的左上角和右下角的坐标。
    其输出为被画上检测框的图片。
    """
    for box in boxes:
        cv2.rectangle(show_image_with_boxes,
                pt1=(int(box[0]), int(box[1])),
                pt2=(int(box[2]), int(box[3])), color=color, thickness=thickness)
#         show_image_with_boxes = cv2.rectangle(show_image_with_boxes,
#                 pt1=(int(box[0]), int(box[1])),
#                 pt2=(int(box[2]), int(box[3])), color=color, thickness=thickness)
    show_image_with_boxes = cv2.cvtColor(show_image_with_boxes, cv2.COLOR_BGR2RGB)
    return show_image_with_boxes

def load_gt_boxes(path):
    """
    returns many ground truth boxes with the shape of [-1, 4].
    xmin, ymin, xmax, ymax
    load_gt_boxes()函数返回一个(-1, 4) 的数组，
    代表着多个检测物体的 ground truth boxes （即真实检测框）的左上角坐标和右下角坐标。
    其中，我们需要输入一个路径，此路径下的 .txt 文件中包含着真实框的 (x, y, w, h)，
    x 表示真实框左上角的横坐标；y 表示真实框左上角的纵坐标；w 表示真实框的宽度；h 表示真实框的高度。
    カツ
    """
    bbs = open(path).readlines()[1:]
    roi = np.zeros([len(bbs), 4])
    for iter_, bb in zip(range(len(bbs)), bbs):
        bb = bb.replace('\n', '').split(' ')
        bbtype = bb[0]
        bba = np.array([float(bb[i]) for i in range(1, 5)])
        # occ = float(bb[5])
        # bbv = np.array([float(bb[i]) for i in range(6, 10)])
        ignore = int(bb[10])

        ignore = ignore or (bbtype != 'person')
        ignore = ignore or (bba[3] < 40)
        bba[2] += bba[0]
        bba[3] += bba[1]

        roi[iter_, :4] = bba
    return roi

def compute_regression(box1, box2):
    """
    box1: ground-truth boxes
    box2: anchor boxes
    因为所有预测框的中心点位置（即特征图的每个块中心）以及尺寸（9个标准尺寸）都是固定的，
    那么这一定会导致获得的检测框很不准确。
    因此，我们希望创造一个映射，可以通过输入正预测框经过映射得到一个跟真实框更接近的回归框。
    假设正预测框的坐标为 ( A x , A y , A w , A h ) (A_x, A_y, A_w, A_h)
    即正预测框左上角坐标为.........(not end~
    カツ
    """
    target_reg = np.zeros(shape=[4,])
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]

    target_reg[0] = (box1[0] - box2[0]) / w2
    target_reg[1] = (box1[1] - box2[1]) / h2
    target_reg[2] = np.log(w1 / w2)
    target_reg[3] = np.log(h1 / h2)

    return target_reg

def decode_output(pred_bboxes, pred_scores, score_thresh=0.5):
    """
    pred_bboxes shape: [1, 45, 60, 9, 4]
    pred_scores shape: [1, 45, 60, 9, 2]
    decode_output 函数的作用是，将一张图片上的 45*60*9 个预测框的平移量与尺度因子以及每个框的得分输入，
    得到每个正预测框对应的回归框（其实所有表示同一个检测目标的回归框都是近似重合的）。
    对于输入：
    pred_bboxes：它的形状为 [1, 45, 60, 9, 4]，表示一共 45*60*9 个预测框，每个预测框都包含着两个平移量和两个尺度因子；
    pred_scores：它的形状为 [1, 45, 60, 9, 2]，表示在 45*60*9 个预测框中，[1, i, j, k, 0] 表示第 i 行第 j 列中的第 k 个预测框中包含的是背景的概率；
    [1, i, j, k, 1] 表示第 i 行第 j 列中的第 k 个预测框中包含的是检测物体的概率。
    其中，经过 meshgrid() 函数后，grid_x 的形状为 (45, 60)，grid_y 的性状也是 (45, 60)，
    它们的不同是：grid_x 由 45 行 range(60) 组成；grid_y 由 60 列 range(45) 组成。
    经过 stack() 函数后，grid_xy 包含着所有特征图中小块的左上角的坐标，如 (0, 0)，(1, 0)，……，(59, 0)，(0, 1)，……，(59, 44)。
    因为特征图中一个小块能表示原始图像中一块 16*16 的区域（也就是说，特征图中一个 1*1 的小块对应着原始图像上一个 16*16 的小块），所以计算原始图像上每个小块的中心 center_xy 时，只需要用 grid_xy 乘 16 加 8 即可。
    计算预测框的左上角坐标时，只需要用 center_xy 减去提前规定的预测框的宽度和长度（Kanchors）的一半即可。
    xy_min 和 xy_max 是回归框的左上角坐标和右下角坐标，它们的计算过程在 compute_regression() 函数那里已经讲过了，此处的 pred_bboxes 输入就是 compute_regression() 函数的输出，其中包含着每个框的平移量和尺度因子。然后将xy_min 和 xy_max 合并，得到新的 pred_bboxes，其中包含着回归框左上角坐标和右下角坐标。
    pred_scores[…, 1] 指的是每个框中含有检测目标的概率（称为得分），如果得分大于阈值，我们就认为这个框中检测到了目标，然后我们把这个框的坐标和得分提取出来，组成新的 pred_bboxes 和 pred_scores。
    经过 decode_output 函数的输出为：
    pred_score：其形状为 [-1, ]，表示每个检测框中的内容是检测物的概率。
    pred_bboxes：其形状为 [-1, 4]，表示每个检测框的左上角和右下角的坐标。
    カツ
    """
    grid_x, grid_y = tf.range(60, dtype=tf.int32), tf.range(45, dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    grid_x, grid_y = tf.expand_dims(grid_x, -1), tf.expand_dims(grid_y, -1)
    grid_xy = tf.stack([grid_x, grid_y], axis=-1)
#     中心xy座標を求めるからxyに+8する
    center_xy = grid_xy * 16 + 8
    center_xy = tf.cast(center_xy, tf.float32)
    anchor_xymin = center_xy - 0.5 * Kanchors

    xy_min = pred_bboxes[..., 0:2] * Kanchors[:, 0:2] + anchor_xymin
    xy_max = tf.exp(pred_bboxes[..., 2:4]) * Kanchors[:, 0:2] + xy_min

    pred_bboxes = tf.concat([xy_min, xy_max], axis=-1)
    pred_scores = pred_scores[..., 1]
    score_mask = pred_scores > score_thresh
    pred_bboxes = tf.reshape(pred_bboxes[score_mask], shape=[-1,4]).numpy()
    pred_scores = tf.reshape(pred_scores[score_mask], shape=[-1,]).numpy()
    return  pred_scores, pred_bboxes


def nms(pred_boxes, pred_score, iou_thresh):
    """
    pred_boxes shape: [-1, 4]
    pred_score shape: [-1,]
    非极大值抑制（Non-Maximum Suppression，NMS），顾名思义就是抑制不是极大值的元素，
    说白了就是去除掉那些重叠率较高但得分较低的预测框。
    
    nms() 函数的作用是从选出的正预测框中进一步选出最好的 n 个预测框，其中，n 指图片中检测物的个数。其流程为：
    取出所有预测框中得分最高的一个，并将这个预测框跟其他的预测框进行 IOU 计算；
    将 IOU 值大于 0.1 的预测框视为与刚取出的得分最高的预测框表示了同一个检测物，故去掉；
    重复以上操作，直到所有其他的预测框都被去掉为止。
    カツ
    """
    selected_boxes = []
    while len(pred_boxes) > 0:
        max_idx = np.argmax(pred_score)
        selected_box = pred_boxes[max_idx]
        selected_boxes.append(selected_box)
        pred_boxes = np.concatenate([pred_boxes[:max_idx], pred_boxes[max_idx+1:]])
        pred_score = np.concatenate([pred_score[:max_idx], pred_score[max_idx+1:]])
        ious = compute_iou(selected_box, pred_boxes)
        iou_mask = ious <= 0.1
        pred_boxes = pred_boxes[iou_mask]
        pred_score = pred_score[iou_mask]

    selected_boxes = np.array(selected_boxes)
    return selected_boxes


# In[ ]:




