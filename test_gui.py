from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras_frcnn import roi_helpers
import keras_frcnn.vgg as nn

import tensorflow as tf
import random as rn
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
tf.keras.backend.set_session(sess)




parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.",default = 'C:/Users/sunzh/CS636/keras_frcnn/TF2.0_version/cat')
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)

(options, args) = parser.parse_args([])

img_path = options.test_path


if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')

with open('config (1).pickle', 'rb') as f_in:
	C = pickle.load(f_in)

# No need for data augment here
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape

	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio

# def format_img_channels(img, C):
# 	""" formats the image channels based on config """
# 	img = img[:, :, (2, 1, 0)]
# 	img = img.astype(np.float32)
# 	img[:, :, 0] -= C.img_chanel_mean[0]
# 	img[:, :, 1] -= C.img_chanel_mean[1]
# 	img[:, :, 2] -= C.img_chanel_mean[2]
# 	img /= C.img_scaleing_factor

# 	img = np.expand_dims(img, axis=0)
# 	return img


def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= np.mean(img[:,:,0])
	img[:, :, 1] -= np.mean(img[:,:,1])
	img[:, :, 2] -= np.mean(img[:,:,2])
	img /= C.img_scaleing_factor

	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

def test(image_path,mode = 'test'):
	## get the dict for labels
	class_mapping = C.class_mapping
	if 'bg' not in class_mapping:
		class_mapping['bg'] = len(class_mapping)

	class_mapping = {v: k for k, v in class_mapping.items()}
	print(class_mapping)
	class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
	C.num_rois = int(options.num_rois) # 32 for default

	model_path = 'model_frcnn_cat.h5'
	# model_path = 'model_frcnn_cat_tested.h5'
	# model_path = 'model_frcnn (1).h5'

	## shape for input
	input_shape_img = (None, None, 3)
	# input_shape_features = (None, None, num_features)
	## rebuild the model in train_frcnn
	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(C.num_rois, 4))
	## Bone network of Vgg16
	shared_layers = nn.nn_base(img_input, trainable=True)
	## network of rpn and classifcation
	num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)                                                                      ##
	rpn = nn.rpn(shared_layers, num_anchors)                                                                              ##
	## [out_class, out_reg]  ## num_rois = 4                                                                              ##
	classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes = len(class_mapping), trainable = True)   ##
	## build model for each network Model(input, output)                                                                  ##
	model_rpn = Model(img_input, rpn[:2])  ## because rpn[2] is base_layers(input)                                        ##
	model_classifier = Model([img_input, roi_input], classifier)                                                          ##
	model_all = Model([img_input, roi_input], rpn[:2]+classifier)
	##
	print('Loading weights from {}'.format(model_path))
	model_rpn.load_weights(model_path, by_name=True)
	model_classifier.load_weights(model_path, by_name=True)
	model_rpn.compile(optimizer='sgd', loss='mse')
	model_classifier.compile(optimizer='sgd', loss='mse')


	bbox_threshold = 0.7
	image = cv2.imread(image_path)
	## resize make the shorter side to be 600
	## and get the resize ratio
	X, ratio = format_img(image, C)
	## make predict
	[Y1, Y2] = model_rpn.predict(X)
	##
	R = roi_helpers.rpn_to_roi(Y1, Y2, C, overlap_thresh=0.7)
	#X2,Y1,Y2,IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)
	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]
	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}


	for jk in range(R.shape[0]//C.num_rois + 1):
		## take 4 ROIs each time
	    ## 1, 32, 4
		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)

		if ROIs.shape[1] == 0:
			break

		## when it comes to the last time
		if jk == R.shape[0]//C.num_rois:
			#pad R
			curr_shape = ROIs.shape # 1,4,4
			## 1 4 4
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_reg] = model_classifier.predict([X, ROIs])

		for ii in range(P_cls.shape[1]): ##32

			print('Max value')
			print(np.max(P_cls[0, ii, :]))
			print('label map')
			print(np.argmax(P_cls[0, ii, :]))

			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			## we get the predict truth
			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_reg[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			## rpn_stride = 16
			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]) )

	all_dets = []
	#print(bboxes)

	## show time !
	for key in bboxes:
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.35)

		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]

			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

			cv2.rectangle(image,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

			textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
			all_dets.append((key,100*new_probs[jk]))

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x1, real_y1-0)

			cv2.rectangle(image, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(image, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(image, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

	#print('Elapsed time = {}'.format(time.time() - st))

	return (image)
