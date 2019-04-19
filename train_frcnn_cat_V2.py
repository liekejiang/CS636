from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
import pandas as pd
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
import itertools
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
parser.add_option("-p", "--path", dest = "train_path", help = "Path to training data")
## use the function def in simple_parser
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc", default = "cat")
## setting the num of roi
parser.add_option("-n", "--num_rois", type = "int",dest = "num_rois", help = "Num of RoIs to process at once", default = 6 )
parser.add_option("--network", dest = "network", default = 'vgg')
## 3kinds of data_aug methods
parser.add_option("--hf", dest = "horizontal_flips", help = "Augment data with method of horizontal flips of image",action="store_true", default = False)
parser.add_option("--vf", dest = "vertical_flips", help = "Augment data with method of vertical flips of image",action="store_true", default = False)
parser.add_option("--rot", dest = "rot_90", help = "Augment data with the method ofke rotation 90 degrees of image", action = "store_true", default = False)
parser.add_option("--num_epochs", type = "int", dest = "num_epochs", help = "Number of epochs.", default = 50)
parser.add_option("--config_filename", dest = "config_filename", help = "Location to store all the metadata related to the training", default = "config.pickle")
parser.add_option("--input_weight_path", dest = "input_weight_path", help = "Input path for weights.",default = "/home/zheyongsun/TF2.0_version/vgg16_weights_tf_dim_ordering_tf_kernels.h5")

(options, args) = parser.parse_args()

if not options.train_path:
    parser.error('Error: path to training data must be specified. Pass --path to command line')

if options.parser == 'cat':
    from keras_frcnn.cat_voc_parser import get_cat
elif options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
else:
    raise ValueError("must have one dataset to train 'pascal_voc' or 'cat'")
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
## third part, set number of region of interests
C.num_rois = int(options.num_rois)

## fourth part, set base net for training
if options.network == 'vgg':
    C.network = 'vgg'
    from keras_frcnn import vgg as nn
else:
    print('Not a valid model')
    raise ValueError("must have one network, 'vgg' or something else")

# fifth part, load the pretrained weight
if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
else:
    C.base_net_weights = nn.get_weight_path()
# sixth part, load the data from pascal
all_imgs, classes_count, class_mapping = get_cat(options.train_path)
## all_imgs : a hash tabel contains the inforamtion of img path, class, width, height, and note for ground truth boxes
## classes_count : how many class in total
## class_mapping :a number mapping for each classes

## seventh part, bg check
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

## eighth part, class_mapping operation save for writing
C.class_mapping = class_mapping
##key to value value to key
#inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
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
random.shuffle(all_imgs)
## length of image dataset
num_imgs = len(all_imgs)

## save train&validation set for data generator
train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']
print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))


## return (x_img, y_rpn_cls, y_rpn_reg, img_data_aug)
## (x_img: input image， y_rpn_cls: includeing [y_rpn_valid, y_rpn_overlap] first one for neg or pos, second one only for pos
## y_rpn_reg：incldeing [y_rpn_overlap ,y_rpn_reg] first one mark for pos example, second one for 4 coordinates
## img_data_aug: the information hash tabel about image)
data_gen_train = itertools.cycle(data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, mode = 'train'))
data_gen_val = itertools.cycle(data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, mode = 'val'))

## input for nn
input_shape_img = (None, None, 3)
img_input = Input(shape = input_shape_img)
roi_input = Input(shape = (None,4))
## set the bone net only with the shape of input
## shared_layers is the output of VGG16 bone net
shared_layers = nn.nn_base(img_input, trainable = True)
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
## rpn network (x_class, x_reg, shared_layers)
rpn = nn.rpn(shared_layers, num_anchors)
## [out_class, out_reg]  ## num_rois = 4
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes = len(classes_count), trainable = True)
## build model for each network Model(input, output)
model_rpn = Model(img_input, rpn[:2])  ## because rpn[2] is base_layers(input)
model_classifier = Model([img_input, roi_input], classifier)
model_all = Model([img_input, roi_input], rpn[:2]+classifier)

try:
    ##base_net_weight not in config.py but defined above
#     print('loading weights from {}'.format('finalmodel_frcnn_cat.h5'))
#     model_rpn.load_weights('finalmodel_frcnn_cat.h5', by_name=True)
#     model_classifier.load_weights('finalmodel_frcnn_cat.h5', by_name = True)
    print('loading weights from {}'.format('vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    model_rpn.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
    model_classifier.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name = True)
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
        https://github.com/fchollet/keras/tree/master/keras/applications')

## optimizer for each network
optimizer = Adam(lr=1e-6)
optimizer_classifier = Adam(lr=1e-6)
model_rpn.compile(optimizer=optimizer, loss = [losses.rpn_loss_cls(num_anchors), losses.rpn_loss_reg(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss = [losses.class_loss_cls, losses.class_loss_reg(len(classes_count)-1)],
                        metrics = {'dense_class_{}'.format(len(classes_count)):'accuracy'})
model_all.compile(optimizer = 'RMSprop', loss = 'mae')
## train 1000 epoch
epoch_length = 250
#num_epochs = int(options.num_epochs)  ## default = 2000
num_epochs = 50
iter_num = 0

losses = np.zeros((epoch_length,5))
#Valid_losses = np.zeros((epoch_length,5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()
best_loss = np.Inf
curr_loss_record = []
valid_curr_loss_record = []
class_mapping_inv = {v: k for k,v in class_mapping.items()}
print('Starting Training')

vis = True

for epoch_num in range(num_epochs):
    Valid_losses = np.zeros((epoch_length,5))
    progbar = Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num+1, num_epochs))

    while True:
        try:
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:  ## verbose = True for default
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes.')
                    print('Check RPN settings or keep training.')
######################################## 4.2 ###########################################################################
            ## X = x_img; Y = [y_rpn_cls, y_rpn_reg]; img_data_aug
            ## Load data to train
            X, Y, img_data = next(data_gen_train)
            X_valid, Y_valid, img_data_valid = next(data_gen_val)
            ## Y is label of class and position reg
            loss_rpn = model_rpn.train_on_batch(X,Y)

            ##  loss_rpn = [1,1]
            ## :x_class [9, h/16, w/16]  = P_rpn[0]
            ## :x_reg [9*4, h/16, w/16]  = P_rpn[1]
            P_rpn = model_rpn.predict_on_batch(X)
            ## get loss for validation and anchor boxes for loss of Classification of Validation
            Valid_rpn_loss = model_rpn.test_on_batch(X_valid, Y_valid)
            Valid_rpn = model_rpn.predict_on_batch(X_valid)
            ## get most 350 predicted anchor boxes for training Classification
            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, use_reg = True, overlap_thresh = 0.7, max_boxes = 350)
            R_valid = roi_helpers.rpn_to_roi(Valid_rpn[0], Valid_rpn[1], C, use_reg = True, overlap_thresh = 0.7, max_boxes = 350)
            ## calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2,Y1,Y2,IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)
            X2_valid,Y1_valid,Y2_valid,IouS_valid = roi_helpers.calc_iou(R_valid, img_data_valid, C, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue
            ## last one is 'bg', so check neg or pos
            neg_samples = np.where(Y1[0,:,-1] == 1)
            pos_samples = np.where(Y1[0,:,-1] == 0)

            pos_samples_valid = np.where(Y1_valid[0,:,-1] == 0)
            neg_samples_valid = np.where(Y1_valid[0,:,-1] == 1)
            
            num_pos_valid = len(pos_samples_valid[0])
            num_pos = len(pos_samples[0])
            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []
 #############################################Valid setting#########################
            if len(neg_samples_valid) > 0:
                neg_samples_valid = neg_samples_valid[0]
            else:
                neg_samples_valid = []
            if len(pos_samples_valid) > 0:
                pos_samples_valid = pos_samples_valid[0]
            else:
                pos_samples_valid = []

            if C.num_rois > 1:
                if len(pos_samples_valid) < C.num_rois:
                    ##from numpy array to list
                    selected_pos_samples_valid = pos_samples_valid.tolist()
                else:
                    selected_pos_samples_valid = np.random.choice(pos_samples_valid, C.num_rois, replace = False).tolist()

                try:
                    selected_neg_samples_valid = np.random.choice(neg_samples_valid, C.num_rois - len(selected_pos_samples_valid), replace = False).tolist()
                except:
                    selected_neg_samples_valid = np.random.choice(neg_samples_valid, C.num_rois - len(selected_pos_samples_valid), replace = True).tolist()

                sel_samples_valid = selected_pos_samples_valid + selected_neg_samples_valid
#############################################Valid setting#########################
            ##
            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append(len(pos_samples))
            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois:
                    ##from numpy array to list
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois, replace = False).tolist()

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

            loss_class = model_classifier.train_on_batch([X,X2[:,sel_samples,:]], [Y1[:, sel_samples,:],Y2[:, sel_samples,:]])
            try:
                Valid_loss_class = model_classifier.test_on_batch([X_valid,X2_valid[:,sel_samples_valid,:]], [Y1_valid[:, sel_samples_valid,:],Y2_valid[:, sel_samples_valid,:]])
                Valid_losses[iter_num, 2] = Valid_loss_class[1]
                Valid_losses[iter_num, 3] = Valid_loss_class[2]
                Valid_losses[iter_num, 4] = Valid_loss_class[3]
            except:
                print('No pos sample for validation')

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]
            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            Valid_losses[iter_num, 0] = Valid_rpn_loss[1]
            Valid_losses[iter_num, 1] = Valid_rpn_loss[2]


            progbar.update(iter_num+1, [('rpn_cls', losses[iter_num, 0]), ('rpn_reg', losses[iter_num,1]),
                                        ('detector_cls', losses[iter_num,2]), ('detector_reg', losses[iter_num,3]),('num of pos',num_pos),('num of pos valid',num_pos_valid)])

            iter_num += 1

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_reg = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_reg = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                Valid_loss_rpn_cls = np.mean(Valid_losses[:, 0])
                Valid_loss_rpn_reg = np.mean(Valid_losses[:, 1])
                Valid_loss_class_cls = np.mean(Valid_losses[:, 2])
                Valid_loss_class_reg = np.mean(Valid_losses[:, 3])
                Valid_class_acc = np.mean(Valid_losses[:, 4])
                valid_curr_loss_record.append([Valid_loss_rpn_reg , Valid_loss_class_cls , Valid_loss_class_reg , Valid_class_acc])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_reg))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_reg))

                    print('*****************Validation')
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(Valid_loss_rpn_cls))
                    print('Loss RPN classifier: {}'.format(Valid_loss_rpn_reg))
                    print('Loss RPN regression: {}'.format(Valid_loss_class_cls))
                    print('Loss Detector classifier: {}'.format(Valid_loss_class_reg))
                    print('Loss Detector regression: {}'.format(Valid_class_acc))
                    print('Elapsed time: {}'.format(time.time() - start_time))


                curr_loss = loss_rpn_cls + loss_rpn_reg + loss_class_cls + loss_class_reg
                curr_loss_record.append([loss_rpn_cls , loss_rpn_reg , loss_class_cls , loss_class_reg])

                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                    best_loss = curr_loss
                    model_all.save("model_frcnn_cat.h5")

                break
        except Exception as e:
            print('Exception: {}'.format(e))
            continue

print('Training complete ,exiting.')
model_all.save("finalmodel_frcnn_cat.h5")

curr_loss_record=pd.DataFrame(columns=['loss_rpn_cls', 'loss_rpn_reg' , 'loss_class_cls' , 'loss_class_reg'],data=curr_loss_record)
valid_curr_loss_record = pd.DataFrame(columns=['loss_rpn_cls' , 'loss_rpn_reg' , 'loss_class_cls' , 'loss_class_reg'],data=valid_curr_loss_record)
curr_loss_record.to_csv('curr_loss_record.csv',encoding='gbk')
valid_curr_loss_record.to_csv('valid_curr_loss_record.csv',encoding='gbk')


