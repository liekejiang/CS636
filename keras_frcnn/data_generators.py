from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
import threading
import itertools
##自定义文件
##self-defined
import sys
sys.path.append("..")
from keras_frcnn.data_augment import augment
import keras_frcnn.data_augment as data_augment

def union(a, b, area_intersection):
    """
    input format:(x1,y1,x2,y2)
    compute union area of ground truth box and anchor after bbox reg   
    """
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_union = area_a + area_b - area_intersection
    return area_union

def intersection(a, b):
    """
    compute intersection of ground truth box and anchor after bbox reg
    """
    x = max(a[0],b[0])
    y = max(a[1],b[1])
    w = min(a[2],b[2]) - x
    h = min(a[3],b[3]) - y
    
    if h<0 or w<0 :
        return 0
    
    return h*w
    
def iou(a,b):
    """
    input format:(x1,y1,x2,y2)
    compute iou of ground truth box and anchor after bbox reg
    """

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0


    area_intersect = intersection(a,b)
    area_union = union(a,b,area_intersect)
    ##avoid area_union to be zero
    return float(area_intersect)/float(area_union + 1e-6)

def get_new_img_size(w, h, img_min_side = 600):
    """
    make sure that the min side of img is 600
    and adjust with the ratio fixed
    according to the paper
    """
    if w <= h:
        f = float(img_min_side) / w
        resized_h = int(f * h)
        resized_w = img_min_side
    else:
        f = float(img_min_side) / h
        resized_w = int(f * w)
        resized_h = img_min_side
        
    return resized_w, resized_h

##class_count: 各个类别的数目, 以hashtable的形式存储
class SampleSelector:
    def __init__(self, class_count):
        """
        choose and build iterator to feed the network 
        """
        ## 把所有数量大于0的类都以list的形式存入 classes
        self.classes = [b for b in class_count.keys() if class_count[b] > 0]
        ## 创建一个以classes为基础的循环迭代器
        self.class_cycle = itertools.cycle(self.classes)
        ## 获取class_cycle迭代器的第一个值，不更新的话不会变 默认是'person'
        self.curr_class = next(self.class_cycle)
    
    def skip_sample_for_balanced_class(self, img_data):
        """
        input is a single element in all_imgs, which is a dict created in pascal_voc_parser
        all_imgs :[file_path, w, h, bbox]
        bbox: [class, x1,x2,y1,y2,difficulty]
        """
        class_in_img = False
        for bbox in img_data['bboxes']:
            cls_name = bbox['class']
            if cls_name == self.curr_class:
                class_in_img = True
                ## 更新一次，获取下一次的值
                self.curr_class = next(self.class_cycle)
                break    
        if class_in_img:
            return False
        else:
            return True
        
def calc_rpn(C, img_data, w, h, resized_w, resized_h,
             img_length_calc_function):
    """
    C = config.Config()
    img_data_aug = {file_path, h,w, bbox:{class, x1,x2,y1,y2,difficulty}}
    w, h = weight/ height
    resized_w/h = output of get_new_img_size()
    img_length_calc_function : return width//16, height//16
    """
    downscale = float(C.rpn_stride)  ## =16
    anchor_sizes = C.anchor_box_scales ## [128,256,512]
    anchor_ratios = C.anchor_box_ratios ## [1, 1], [1, 2], [2, 1]]
    num_anchors = len(anchor_sizes) * len(anchor_ratios)  ## 3*3 HERE
    
    ##compute the size of output
    ## DIVIDE 16 INDIVIDUALLY
    (output_w, output_h) = img_length_calc_function(resized_w, resized_h)
    
    # the number of ratios in config
    n_ratios = len(anchor_ratios)
    
    ##init the space for h/16 w/16 anchors
    y_rpn_overlap = np.zeros((output_h, output_w, num_anchors))
    y_is_box_valid = np.zeros((output_h, output_w, num_anchors))
    ## reg 是针对每一个像素为中心的9个anchorbox中IOU最大的box的tx,ty,tw,tz 也就是loss中使用的参数
    y_rpn_reg = np.zeros((output_h, output_w, num_anchors * 4))
    
    ## the number of bounding boxes in one image, usually one
    num_bboxes = len(img_data['bboxes'])
    ################################################################ Read Here #############################
    ## 每有一个bbox在当前的图里面，就要多分配一个计数位
    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int) ## num of pos case
    ## assign -1 for each element
    ## 大部分是（1，4）？
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes,4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes,4)).astype(np.float32)
    
    #ground truth box coordinatoes
    gtb_c = np.zeros((num_bboxes,4))
    ## resize to fit the size of image
    ## bbox_num 是键的序号，从0开始
    ## bbox是键
    ## 根据调整后的大小调整bbox的四坐标
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        gtb_c[bbox_num, 0] = bbox['x1'] * (resized_w / float(w))
        gtb_c[bbox_num, 1] = bbox['x2'] * (resized_w / float(w))
        gtb_c[bbox_num, 2] = bbox['y1'] * (resized_h / float(h))
        gtb_c[bbox_num, 3] = bbox['y2'] * (resized_h / float(h))

    ##以图象中每一个像素为中心建立九种长度不同，长宽比不同的anchor box 并且计算与之对应的ground truth的IOU 并且判断是正还是反例
    for anchor_size_index in range(len(anchor_sizes)): #len([128, 256, 512])=3
        for anchor_ratio_index in range(n_ratios):  #[[1, 1], [1, 2], [2, 1]]
            ##generate the size of weight and height for each anchor
            ## 计算当前对应格式的anchor box的长宽
            anchor_x = anchor_sizes[anchor_size_index] * anchor_ratios[anchor_ratio_index][0] ##[128, 128, 256, 256, 256, 512, 512, 512, 1024]
            anchor_y = anchor_sizes[anchor_size_index] * anchor_ratios[anchor_ratio_index][1] ##[128, 256, 128, 256, 512, 256, 512 ,1024, 512]

            ##每个像素来一遍
            for index_x in range(output_w):
                #downscale = 16 (stride)
                #downscale * index is the coord of center
                ## 以每个像素为中心 来九个anchor
                ## 分别是anchor box在原图中的的左右x轴坐标
                x1_anchor = downscale * (index_x + 0.5) - anchor_x / 2
                x2_anchor = downscale * (index_x + 0.5) + anchor_x / 2
                
                #get rid of the boxes which are across the boundaries
                #accoding to the setting in paper
                if x1_anchor < 0 or x2_anchor > resized_w:
                    continue
                
                for index_y in range(output_h):
                    ##same operation for y-coordinate
                    y1_anchor = downscale * (index_y + 0.5) - anchor_y / 2
                    y2_anchor = downscale * (index_y + 0.5) + anchor_y / 2 
                    
                    #same as before: boundaries exam
                    if y1_anchor < 0 or y2_anchor > resized_h:
                        continue
                    
                    #neg for dafault
                    bbox_type = 'neg'
                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):##一图中有几个bbox就有几个bbox_num
                        #call function iou to compute the overlap area of anchor box and ground truth bounding box
                        current_iou = iou([gtb_c[bbox_num,0],gtb_c[bbox_num,2],gtb_c[bbox_num,1],gtb_c[bbox_num,3]],
                                         [x1_anchor,y1_anchor,x2_anchor,y2_anchor])
                        #select for bounding box reg
                        if current_iou > best_iou_for_bbox[bbox_num] or current_iou > C.rpn_max_overlap:
                            center_gt_x = (gtb_c[bbox_num, 0] + gtb_c[bbox_num, 1]) / 2.0 #cx: center x for ground truth box
                            center_gt_y = (gtb_c[bbox_num, 2] + gtb_c[bbox_num, 3]) / 2.0 #cy: center y for ground truth box
                            center_x = (x1_anchor + x2_anchor) / 2.0 #cxa: center x for anchor box
                            center_y = (y1_anchor + y2_anchor) / 2.0 #cya: center y for anchor box

                            ## Ground truth t*x t*y t*w t*h
                            tx = (center_gt_x - center_x) / (x2_anchor - x1_anchor) ## （2） in paper
                            ty = (center_gt_y - center_y) / (y2_anchor - y1_anchor) ##
                            tw = np.log((gtb_c[bbox_num, 1] - gtb_c[bbox_num, 0]) / (x2_anchor - x1_anchor)) ## log(x2-x1)
                            th = np.log((gtb_c[bbox_num, 3] - gtb_c[bbox_num, 2]) / (y2_anchor - y1_anchor))## log(y2-y1)
                        
                        #background exam
                        if img_data['bboxes'][bbox_num]['class'] != 'bg':
                            #match every gt box to a predict box
                            if current_iou > best_iou_for_bbox[bbox_num]:
                                #update current optimal matched predicted box
                                ## Save best x1 y1 ratio length
                                best_anchor_for_bbox[bbox_num] = [index_x, index_y, anchor_ratio_index, anchor_size_index]
                                ## Save best IOU
                                best_iou_for_bbox[bbox_num] = current_iou
                                ## Save anchor box coor
                                best_x_for_bbox[bbox_num, :] = [x1_anchor, x2_anchor, y1_anchor, y2_anchor]
                                ## Save the number used in L for this anchor box
                                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]
                                
                            #set pos label if iou>0.7    
                            if current_iou > C.rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1
                                
                                if current_iou > best_iou_for_loc:
                                    best_iou_for_loc = current_iou
                                    best_reg = (tx, ty, tw, th)
                            
                            #neutral example
                            if C.rpn_min_overlap < current_iou and current_iou < C.rpn_max_overlap:
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    ## 如果是负样本， y_is_box_valid 为1 y_rpn_overlap为0， 通过第三维来确定对应的anchor长度和长宽比
                    ## 正负样本都是有效的，中立不考虑所以y_is_box_valid在正负都是1， 只有正样本的IOU有考虑价值，所以y_rpn_reg标记为1而且y_rpn_reg需要保存数据
                    ## 对最后输出图上的每一个像素点的每一个种格式提出一种且只有一种anchor box 这与图种有多少GT box无关
                    ## 因为无论GT box目标有多少，当前关注的有且只有在这个相速度以某种宽度和长宽比存在的anchor box
                    ## 若当前anchor box 与GT box的重合度大于阈值，那么它就是正样本 反之亦然
                    ## 负样本和正样本有用，用来训练二分类，valid数组用于记录
                    ## y_rpn_overlap 用于记录iou是否大于阈值，标记正样本
                    ## 正样本还需要记录
                    if bbox_type == 'neg':
                        y_is_box_valid[index_y, index_x, anchor_ratio_index +
                                       n_ratios * anchor_size_index] = 1
                        y_rpn_overlap[index_y, index_x, anchor_ratio_index + 
                                     n_ratios * anchor_size_index] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[index_y, index_x, anchor_ratio_index +
                                       n_ratios * anchor_size_index] = 0
                        y_rpn_overlap[index_y, index_x, anchor_ratio_index + 
                                     n_ratios * anchor_size_index] = 0                        
                    elif bbox_type == 'pos':
                        y_is_box_valid[index_y, index_x, anchor_ratio_index +
                                       n_ratios * anchor_size_index] = 1
                        y_rpn_overlap[index_y, index_x, anchor_ratio_index + 
                                     n_ratios * anchor_size_index] = 1
                        ## start 是当前对应的第几种情况 一共九种
                        start = 4 * (anchor_ratio_index + n_ratios * anchor_size_index)
                        y_rpn_reg[index_y, index_x, start:start+4] = best_reg
    
    
    #make sure that each bounding box has at least one pos rpn region
    ## 在最大IOU不超过0.7的情况下，需要选一个最大值作为对应的最优anchor box
    ## 遍历该图片中的所有目标类别，若该类别没有与之对应的IOU大于阈值的的anchor box
    for index in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[index] == 0:
            #跳过bg类
            if best_anchor_for_bbox[index, 0] == -1:
                continue
            ## 强制选择一个正样本
            y_is_box_valid[best_anchor_for_bbox[index, 0], best_anchor_for_bbox[index,1],
                          best_anchor_for_bbox[index,2] + n_ratios * best_anchor_for_bbox[index,3]] = 1
            y_rpn_overlap[best_anchor_for_bbox[index, 0], best_anchor_for_bbox[index,1],
                         best_anchor_for_bbox[index,2] + n_ratios * best_anchor_for_bbox[index,3]]=1
            start = 4 * (best_anchor_for_bbox[index,2] + n_ratios * best_anchor_for_bbox[index,3])
            y_rpn_reg[best_anchor_for_bbox[index,0], best_anchor_for_bbox[index,1],start:start+4]= best_dx_for_bbox[index, :]

    ## after trans : num_anchor类别, height, weigh
    ## 为很么要扩充第一维
    y_rpn_overlap = np.transpose(y_rpn_overlap, (2,0,1))                    
    y_rpn_overlap = np.expand_dims(y_rpn_overlap,axis = 0)

    ## after trans : num_anchor类别, height, weigh
    y_is_box_valid = np.transpose(y_is_box_valid,(2,0,1))
    y_is_box_valid = np.expand_dims(y_is_box_valid,axis = 0)
    
    y_rpn_reg = np.transpose(y_rpn_reg,(2,0,1))
    y_rpn_reg = np.expand_dims(y_rpn_reg,axis=0)
                         
    pos_locs = np.where(np.logical_and(y_rpn_overlap[0,:,:,:] == 1, y_is_box_valid[0,:,:,:] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0,:,:,:] == 0, y_is_box_valid[0,:,:,:] == 1))                     
    
    #choose only 256 example for each img 
    #according to the paper
    ## 会得到第二维的数量
    num_pos = len(pos_locs[0])
    
    num_regions = 256
    # 2 ways to make sure 256 examples are selected
    if len(pos_locs[0]) > num_regions/2:
        ##从0 到 len(pos_locs[0])中间随机选取 len(pos_locs[0]) -128 个数， 将其有效位置零，变成中立样本
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions/2                 
                         
    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    ## 最令我不解的就是这里的拼接，
    ## 1, num_anchor*2, height, weight
    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    ## 1, num_anchor*4*2, height, weight
    y_rpn_reg = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_reg], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_reg)
     
class threadsafe_iter:
    """
    make sure the safe of SampleSelect input data generator
    lock the next() to make it occupy the enough resources
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
        
    def __iter__(self):
        return self
    
    def next(self):
        with self.lock:
            return next(self.it)
        
def threadsafe_generator(f):
    """
    call the class threadsafe_iter to instantize a threadsafe iterator 
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

##输入参数为 train_imgs: all_imgs中的子集，包含图片路径，大小，bbox的类别，坐标
##          class_count: 各个类别的数目
##          C : 训练配置
##          nn.get_img_output_length: return width//16, height//16 表示使用vgg16最后会缩小到原来的16倍
##          mode = ‘train'  表示训练集，’val‘同理
def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, mode='train'):
    #original code doen not compatible with py3.5
    #all_img_data = sorted(all_img_data)
    sample_selector = SampleSelector(class_count)


    if mode == 'train':
        #randomlized the data
        np.random.shuffle(all_img_data)
    #print("start")

    for img_data in all_img_data:
        try:
            ## C.balanced_classes = False for dafault
            ## bool for skip_
            ## some kinds of threshold
            if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                continue

            #read img and augment
            ## if train: hor/ver flip or rotate according to setting
            ## else: just read img and return img and its size
            ## img_data_aug 是哈希表，存放各种和当前图象有关的信息
            ## img 是cv2读进来的图片
            #print("read")
            if mode == 'train' :
                img_data_aug, x_img = data_augment.augment(img_data, C, augment = True)
            else:
                img_data_aug, x_img = data_augment.augment(img_data, C, augment = False)
            ## check the information on side
            (width, height) = (img_data_aug['width'], img_data_aug['height'])
            (rows,cols, _ ) = x_img.shape
            assert cols == width
            assert rows == height

            # get image dim
            # C.im_size = 600
            # resize and make the shorter side fixing at 600
            (resized_w, resized_h) = get_new_img_size(width, height,C.im_size)
            x_img = cv2.resize(x_img, (resized_w, resized_h), interpolation = cv2.INTER_CUBIC)

            try:
                y_rpn_cls, y_rpn_reg = calc_rpn(C, img_data_aug, width, height, resized_w, resized_h, img_length_calc_function)
            except:
                continue

            ############################################### 4.1 ################################################################

            ## Zero-mean preprocess
            #BGR -> RGB due to cv2
            x_img = x_img[:,:,(2,1,0)]
            x_img = x_img.astype(np.float32)

            #C.img_channel_mean should calculated by ourself
            ## 减去均值达成均值归一
#             x_img[:,:,0] -= C.img_chanel_mean[0]
#             x_img[:,:,1] -= C.img_chanel_mean[1]
#             x_img[:,:,2] -= C.img_chanel_mean[2]
            x_img[:,:,0] -= np.mean(x_img[:,:,0])
            x_img[:,:,1] -= np.mean(x_img[:,:,1])
            x_img[:,:,2] -= np.mean(x_img[:,:,2])
            # C.img_scaling_factor = 1
            x_img /=  C.img_scaleing_factor


            ## 1， channel, height , weight
            x_img = np.transpose(x_img, (2,0,1))
            x_img = np.expand_dims(x_img, axis = 0)

            C.std_scaling = 4
            ## 1, num_anchor*4*2, height, weight
            ## y_rpn_reg.shape[1] // 2 = 9
            ## y_rpn_overlap 前面9个
            ## y_rpn_reg     后面9个
            y_rpn_reg[:, y_rpn_reg.shape[1] // 2:, :, :] *= C.std_scaling

            ## 1，height width channel for tensorflow form
            x_img = np.transpose(x_img, (0,2,3,1))
            ## 1，height, width, num_anchor*x for tensorflow form
            y_rpn_cls = np.transpose(y_rpn_cls, (0,2,3,1))
            y_rpn_reg = np.transpose(y_rpn_reg, (0,2,3,1))

            #build a generator
            yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_reg)], img_data_aug

        #keep moving
        except Exception as e:
            print(e)
            continue





