import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import math

class Config:
    def __init__(self):
        self.verbose = True
        
        self.network = 'resnet50'
        
        ##data augmentation
        ##水平翻转 垂直翻转 旋转90°
        self.use_horziontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False
        
        #anchor box scales
        #论文中设置了3种scales和三种ratios
        self.anchor_box_scales = [128,256,512]
        
        # anchor box ratios
        #anchor的长宽比 一共三种，为什么不用1：2
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
        
        #fix the size of smaller side of images
        self.im_size = 600
        
        #reduce channel-wise mean to reach 0-mean
        #应该是根据输入数据来计算的，这里为何固定 
        self.img_chanel_mean = [103.939, 116.779, 123.68]
        self.img_scaleing_factor = 255.0
        
        #num of ROI
        self.num_rois = 4
        
        #经过conv维度共下降16倍 与论文一致但是要结合实际conv网络
        self.rpn_stride = 16
        
        #这是什么
        self.balanced_classes = False
        
        #设置训练dev, 第二个是啥我也不懂，猜一个边框回归参数
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
        
        #IOU for rpn,与论文一致
        self.rpn_min_overlap = 0.5
        self.rpn_max_overlap = 0.75
        
        #IOU for classifiers 应该是fast rcnn 中的参数
        #应该是用于提取难例挖掘的负样本
        self.classifier_min_overlap = 0.25
        self.classifier_max_overlap = 0.6
        
        ## placeholder for the class mapping, automatically generated by the parser
        ## 一个类别数字映射表，每个类别有独一的数字对应
        self.class_mapping = None
        
        #location of pretrained weights for the base network 
        # weight files can be found at:
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
        self.model_path = ''
