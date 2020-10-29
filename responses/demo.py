#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[7]:



# import matplotlib.pyplot as plt
# %matplotlib inline
# plt.title('Test Graph')
# plt.plot([1,3,2,4])
# plt.show()
# plt.title('Test Graph')
# plt.plot([1,3,2,4])
# plt.show()


# In[5]:



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:

import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from utils import compute_iou, plot_boxes_on_image,Kanchors,load_gt_boxes, compute_regression, decode_output

pos_thresh = 0.5
neg_thresh = 0.1
iou_thresh = 0.5
grid_width = 16## 网格的长宽都是16，因为从原始图片到 feature map 经历了16倍的缩放，SSD和Mask要重写
grid_height = 16
image_height = 720
image_width = 960

image_path = "./Data/synthetic_dataset/image/1.jpg"
label_path ="./Data/synthetic_dataset/imageAno/1.txt"
gt_boxes = load_gt_boxes(label_path)## 把 ground truth boxes 的坐标读取出来
raw_image = cv2.imread(image_path) # 将图片读取出来 (高，宽，通道数)
image_with_gt_boxes = np.copy(raw_image) # 复制原始图片
plot_boxes_on_image(image_with_gt_boxes, gt_boxes)# 将 ground truth boxes 画在图片上
Image.fromarray(image_with_gt_boxes).show()# 展示画了 ground truth boxes 的图片然后，我们需要再此复制原始图片用来求解每个预测框的得分和回归变量（平移量与尺度因子）。
encoded_image = np.copy(raw_image)# 再复制原始图片
## 因为得到的 feature map 的长宽都是原始图片的 1/16，所以这里 45=720/16，60=960/16。
target_scores = np.zeros(shape=[45, 60, 9, 2]) # 0: background, 1: foreground, ,
target_bboxes = np.zeros(shape=[45, 60, 9, 4]) # t_x, t_y, t_w, t_h
target_masks  = np.zeros(shape=[45, 60, 9]) # negative_samples: -1, positive_samples: 1

################################### ENCODE INPUT #################################
## 将 feature map 分成 45*60 个小块
#decode_image = np.copy(raw_image)# 再复制原始图片
for i in range(45):
    for j in range(60):
        for k in range(9):
            center_x = j * grid_width + grid_width * 0.5
            center_y = i * grid_height + grid_height * 0.5
            xmin = center_x - Kanchors[k][0] * 0.5
            ymin = center_y - Kanchors[k][1] * 0.5
            xmax = center_x + Kanchors[k][0] * 0.5
            ymax = center_y + Kanchors[k][1] * 0.5
            # ignore cross-boundary anchors
            if (xmin > -5) & (ymin > -5) & (xmax < (image_width+5)) & (ymax < (image_height+5)):
                anchor_boxes = np.array([xmin, ymin, xmax, ymax])
                anchor_boxes = np.expand_dims(anchor_boxes, axis=0)
                # compute iou between this anchor and all ground-truth boxes in image.计算这个锚和真实所有框之间的iou
                ious = compute_iou(anchor_boxes, gt_boxes)
                positive_masks = ious > pos_thresh
                negative_masks = ious < neg_thresh

                if np.any(positive_masks):
                    plot_boxes_on_image(encoded_image, anchor_boxes, thickness=1)
                    print("=> Encoding positive sample: %d, %d, %d" %(i, j, k))
                    cv2.circle(encoded_image, center=(int(0.5*(xmin+xmax)), int(0.5*(ymin+ymax))),
                                    radius=1, color=[255,0,0], thickness=4) # 正预测框的中心点用红圆表示

                    target_scores[i, j, k, 1] = 1. # 表示检测到物体
                    target_masks[i, j, k] = 1 # labeled as a positive sample
                    # find out which ground-truth box matches this anchor找出哪些真实这个锚点匹配
                    max_iou_idx = np.argmax(ious)
                    selected_gt_boxes = gt_boxes[max_iou_idx]
                    target_bboxes[i, j, k] = compute_regression(selected_gt_boxes, anchor_boxes[0])

                if np.all(negative_masks):
                    target_scores[i, j, k, 0] = 1. # 表示是背景
                    target_masks[i, j, k] = -1 # labeled as a negative sample贴上一个负样本
                    cv2.circle(encoded_image, center=(int(0.5*(xmin+xmax)), int(0.5*(ymin+ymax))),
                                    radius=1, color=[0,0,0], thickness=4)#thickness厚度；负预测框的中心点用黑圆表示

Image.fromarray(encoded_image).show()

################################### DECODE OUTPUT #################################

decode_image = np.copy(raw_image)# 再复制原始图片
pred_boxes = []
pred_score = []

for i in range(45):
    for j in range(60):
        for k in range(9):
            # pred boxes coordinate
            center_x = j * grid_width + 0.5 * grid_width
            center_y = i * grid_height + 0.5 * grid_height
            anchor_xmin = center_x - 0.5 * Kanchors[k, 0]
            anchor_ymin = center_y - 0.5 * Kanchors[k, 1]

            xmin = target_bboxes[i, j, k, 0] * Kanchors[k, 0] + anchor_xmin
            ymin = target_bboxes[i, j, k, 1] * Kanchors[k, 1] + anchor_ymin
            xmax = tf.exp(target_bboxes[i, j, k, 2]) * Kanchors[k, 0] + xmin
            ymax = tf.exp(target_bboxes[i, j, k, 3]) * Kanchors[k, 1] + ymin

            if target_scores[i, j, k, 1] > 0: # it is a positive sample
                print("=> Decoding positive sample: %d, %d, %d" %(i, j, k))
                cv2.circle(decode_image, center=(int(0.5*(xmin+xmax)), int(0.5*(ymin+ymax))),
                                radius=1, color=[255,0,0], thickness=4)
                pred_boxes.append(np.array([xmin, ymin, xmax, ymax]))
                pred_score.append(target_scores[i, j, k, 1])

pred_boxes = np.array(pred_boxes)
plot_boxes_on_image(decode_image, pred_boxes, color=[0, 255, 0])
Image.fromarray(np.uint8(decode_image)).show()
#Image.fromarray(np.uint8(decode_image))
#plt.plot(np.uint8(decode_image))

############################## DECODE -FASTERKATU- OUTPUT ###############################

faster_decode_image = np.copy(raw_image)
pred_bboxes = np.expand_dims(target_bboxes, 0).astype(np.float32)
pred_scores = np.expand_dims(target_scores, 0).astype(np.float32)

pred_scores, pred_bboxes = decode_output(pred_bboxes, pred_scores)
plot_boxes_on_image(faster_decode_image, pred_bboxes, color=[255, 0, 0]) # red boundig box
Image.fromarray(np.uint8(faster_decode_image)).show()
#plt.plot(np.uint8(faster_decode_image))
#plt.show()


############################# DECODE OUTPUT ###############################
# faster_decode_image = np.copy(raw_image)
# pred_bboxes = np.expand_dims(target_bboxes, 0).astype(np.float32)
# pred_scores = np.expand_dims(target_scores, 0).astype(np.float32)
# pred_scores, pred_bboxes = decode_output(pred_bboxes, pred_scores)
# plot_boxes_on_image(faster_decode_image, pred_bboxes, color=[255, 0, 0]) # red boundig box
# Image.fromarray(np.uint8(faster_decode_image)).show()


# In[ ]:





# In[ ]:





# In[ ]:




