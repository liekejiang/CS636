from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar

from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers

import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

## max recursion number
sys.setrecursionlimit(40000)

## default command: python train_frcnn.py -p /path/to/pascalvoc/
##
parser = OptionParser()
#-p is the most important one
parser.add_option("-p", "--path", dest = "train_path", help = "Path to training data",default = "C:/Users/sunzh/CS636/keras_frcnn/TF2.0_version")
## use the function def in simple_parser
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc", default = "pascal_voc")
## setting the num of roi
parser.add_option("-n", "--num_rois", type = "int",dest = "num_rois", help = "Num of RoIs to process at once", default = 32 )
parser.add_option("--network", dest = "network", help="Base network to use. Support vgg or resnet50.", default = 'vgg')
## 3kinds of data_aug methods
parser.add_option("--hf", dest = "horizontal_flips", help = "Augment data with method of horizontal flips of image",action="store_true", default = False)
parser.add_option("--vf", dest = "vertical_flips", help = "Augment data with method of vertical flips of image",action="store_true", default = False)
parser.add_option("--rot", dest = "rot_90", help = "Augment data with the method ofke rotation 90 degrees of image", action = "store_true", default = False)

parser.add_option("--num_epochs", type = "int", dest = "num_epochs", help = "Number of epochs.", default = 20)
parser.add_option("--config_filename", dest = "config_filename", help = "Location to store all the metadata related to the training", default = "config.pickle")
parser.add_option("--output_weight_path", dest = "out_put", help = "Output path for weights.", default = "/model_frcnn.hdf5")
parser.add_option("--input_weight_path", dest = "input_weight_path", help = "Input path for weights.",default = "C:/Users/sunzh/CS636/keras_frcnn/TF2.0_version/vgg16_weights_tf_dim_ordering_tf_kernels.h5")

(options, args) = parser.parse_args()

if not options.train_path:
    parser.error('Error: path to training data must be specified. Pass --path to command line')
if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError("must have one dataset to train 'pascal_voc' or 'simple'")
##########################################################################################################################################################
#change config file according to the given paras
C = config.Config()
## first part, data augment setting
## all default values are False
C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

## second part, the path to save model
#C.model_path = options.output_weight_path
C.model_path = "\model_frcnn.hdf5"
## third part, set number of region of interests
C.num_rois = int(options.num_rois)

## fourth part, set base net for training
if options.network == 'vgg':
    C.network = 'vgg'
    from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
    from keras_frcnn import resnet as nn
    C.network = 'resnet50'
else:
    print('Not a valid model')
    raise ValueError("must have one network, 'vgg' or 'resnet50'")

# fifth part, load the pretrained weight
if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
else:
    C.base_net_weights = nn.get_weight_path()
# sixth part, load the data from pascal
all_imgs, classes_count, class_mapping = get_data(options.train_path)
## all_imgs 是一个由哈希表组成的列表，每个哈希元素包括了该图片文件的路径，图片的高度和宽度和一个名为bboxes的子哈希表，bboxes中包括了被选中的图片的类别，4坐标和是否为难例
## classes_count 记录了一共有多少种类别和相关数目
## class_mapping 为每个类记录了一种数字映射


## bg 这是什么
## bg 应该是一个图片类别，指代background， 用于进行反例训练
## 若无bg类，为了与接下来的代码保持一致，创建bg类 数量为0
## seventh part, bg check
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

## eighth part, class_mapping operation
## default = None
## 应该是是否使用class_mapping的开关
C.class_mapping = class_mapping
##key to value value to key
#inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
## pprint排版更好，会换行
## 这里应该是显示所有的类别及其相关信息，每个类一行
pprint.pprint(classes_count)
print('Num class (including bg) = {}'.format(len(classes_count)))

## ninth part, write and save config file for test
## default = config.pickle
config_output_filename = options.config_filename
with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C,config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))
#######################################################################
	##Upper part about config, training start here
#######################################################################

## 随机打乱数据
random.shuffle(all_imgs)
## length of image dataset
num_imgs = len(all_imgs)

## load train&validation set
train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']
print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

#cancel the K.image_dim_ordering() arg
## nn.
##输入参数为 train_imgs: all_imgs中的子集，包含图片路径，大小，bbox的类别，坐标
##          class_count: 各个类别的数目
##          C : 训练配置
##          nn.get_img_output_length: return width//16, height//16 表示使用vgg16最后会缩小到原来的16倍
##          mode = ‘train'  表示训练集，’val‘同理
## 理论返回四个值，但是这里只有一个，是这四个值的一个元组(x_img, y_rpn_cls, y_rpn_reg, img_data_aug)
## 分别是(x_img: 图片， y_rpn_cls: 包含y_rpn_valid, y_rpn_overlap 前者表示该像素位的某种格式的anchor是否有效，后者表示是pos还是neg )
## y_rpn_reg包含：y_rpn_overlap 和 原本的y_rpn_reg 后者表示若有超过阈值的anchor的 四个loss里用到的值
## img_data_aug 表示图片的一系列相关信息)
data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, mode = 'train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, mode = 'val')

######################################## 4.1 ###########################################################################

## input for nn
input_shape_img = (None, None, 3)
## Input用来初始化一个keras tensor
img_input = Input(shape = input_shape_img)
roi_input = Input(shape = (None,4))
## set the bone net only with the shape of input
## shared_layers is the output of VGG16 bone net
shared_layers = nn.nn_base(img_input, trainable = True)

# ## build division shared model of VGG16 ####################
# VGG_model = Model(img_input, shared_layers)               ##
#
# img_input_rpn = Input(shape = input_shape_img)            ##
# img_input_classifier = Input(shape = input_shape_img)     ##
# img_input_all = Input(shape = input_shape_img)            ##
#                                                           ##
# VGG16_output_rpn = VGG_model(img_input_rpn)               ##
# VGG16_output_classifier = VGG_model(img_input_classifier) ##
# VGG16_output_all = VGG_model(img_input_all)               ##
#
# num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
# rpn = nn.rpn(VGG16_output_rpn, num_anchors)
#
# classifier = nn.classifier(VGG16_output_classifier, roi_input, C.num_rois, nb_classes = len(classes_count), trainable = True)
#
# model_rpn = Model(img_input_rpn, rpn[:2])  ## because rpn[2] is base_layers(input)
# #                                     ##
# model_classifier = Model([img_input_classifier, roi_input], classifier)                                                          ##
# model_all = Model([img_input_all, roi_input], rpn[:2]+classifier)
############################################################


################origin code ############################################################################################
## num of anchor                                                                                                      ##
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)                                                     ##
## rpn network (x_class, x_reg, shared_layers)                                                                        ##
rpn = nn.rpn(shared_layers, num_anchors)                                                                              ##
## [out_class, out_reg]  ## num_rois = 4                                                                              ##
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes = len(classes_count), trainable = True)   ##
## build model for each network Model(input, output)                                                                  ##
#######################debug here 4.8####################################
model_rpn = Model(img_input, rpn[:2])  ## because rpn[2] is base_layers(input)                                        ##
model_classifier = Model([img_input, roi_input], classifier)                                                          ##
model_all = Model([img_input, roi_input], rpn[:2]+classifier)                                                         ##
########################################################################################################################
try:
    ##base_net_weight not in config.py but defined above
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name = True)
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
        https://github.com/fchollet/keras/tree/master/keras/applications')

######################################## Loss need to be reviewed ###########################################################################
## optimizer for each network
optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss = [losses.rpn_loss_cls(num_anchors), losses.rpn_loss_reg(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss = [losses.class_loss_cls, losses.class_loss_reg(len(classes_count)-1)],
                        metrics = {'dense_class_{}'.format(len(classes_count)):'accuracy'})
model_all.compile(optimizer = 'RMSprop', loss = 'mae')
## train 1000 epoch
epoch_length = 1000
num_epochs = int(options.num_epochs)  ## default = 2000
iter_num = 0

losses = np.zeros((epoch_length,5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()
best_loss = np.Inf

class_mapping_inv = {v: k for k,v in class_mapping.items()}
print('Starting Training')

vis = True

for epoch_num in range(num_epochs):
    progbar = Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num+1, num_epochs))

    while True:
        try:
            ##在第1k个epoch 检查一次rpn重合面积
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:  ## verbose = True for default
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = [] ##清空
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes.')
                    print('Check RPN settings or keep training.')
######################################## 4.2 ###########################################################################
            ## X = x_img; Y = [y_rpn_cls, y_rpn_reg]; img_data_aug
            ## Load data to train
            X, Y, img_data = next(data_gen_train)

            ## Y is label of class and position reg
            loss_rpn = model_rpn.train_on_batch(X,Y)
            ## 返回值的格式在model初始化的时候就确定，定义在上面
            ## :x_class [9, h/16, w/16]  = P_rpn[0]
            ## :x_reg [9*4, h/16, w/16]  = P_rpn[1]
            P_rpn = model_rpn.predict_on_batch(X)
            ## 选出最多300个被分为正样本的predict boxes
            ## 返回的是预测输出 P_rpn  中这最多300个对应的正负样预测值和边框预测值（x1,x2,y1,y2）而不是计算值tx
            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, use_reg = True, overlap_thresh = 0.7, max_boxes = 300)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            ## X2 是预测坐标x1, y1, w, h
            ## Y1 是class标签[None(expanded),num of pos box , len of class ]
            ## Y2 是坐标标签
            ## IouS 用于debug
            X2,Y1,Y2,IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)
######################################## 4.6 ###########################################################################
            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue
            ## 最后一列是bg 因此可以判断是否是正样本
            neg_samples = np.where(Y1[0,:,-1] == 1)
            pos_samples = np.where(Y1[0,:,-1] == 0)
            ## 获取正负样本的数量
            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []
            ##
            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append(len(pos_samples))
            ## 只选择2个 正例子转化成list格式， 负样本也是2个
            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois//2:
                    ##from numpy array to list
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace = False).tolist()

                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace = False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace = True).tolist()

                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                #in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0,2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            ## 训练代码在这里
            ##  y_rpn_cls: 包含y_rpn_valid, y_rpn_overlap 前者表示该像素位的某种格式的anchor是否有效，后者表示是pos还是neg )
            loss_class = model_classifier.train_on_batch([X,X2[:,sel_samples,:]], [Y1[:, sel_samples,:],Y2[:, sel_samples,:]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            progbar.update(iter_num+1, [('rpn_cls', losses[iter_num, 0]), ('rpn_reg', losses[iter_num,1]),
                                        ('detector_cls', losses[iter_num,2]), ('detector_reg', losses[iter_num,3])])
            iter_num += 1

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_reg = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_reg = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_reg))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_reg))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_reg + loss_class_cls + loss_class_reg
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(C.model_path)

                break
        except Exception as e:
            print('Exception: {}').format(e)
            continue
##怎么判断训练结束
print('Training complete ,exiting.')
