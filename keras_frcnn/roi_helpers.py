import numpy as np
import pdb
import math
import sys
sys.path.append("..")
import keras_frcnn.data_generators as data_generators
import copy


def calc_iou(R, img_data, C, class_mapping):
	"""
	本函数读入图片数据和经过非极大值抑制的rpn预测的坐标数据，对每个提出的predict box进行label的制作
	label包括GT 类别， GT坐标
	一个box四坐标只能对应一个类别，所以即使一张图里面有很多GT，只能选择iou最大的用于匹配
	:param R:
	:param img_data:
	:param C:
	:param class_mapping:
	:return:
	"""
    ## img_data_aug 是哈希表，存放各种和当前图象有关的信息
	bboxes = img_data['bboxes']

	(width, height) = (img_data['width'], img_data['height'])
	# get image dimensions for resizing
	(resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)
	## 有多少标注的ground truth boxes 就有多少gta
	gta = np.zeros((len(bboxes), 4))

	for bbox_num, bbox in enumerate(bboxes):
		# get the GT box coordinates, and resize to account for image resizing
		## 计算调整后且映射到输出图大小上的ground truth boxes
		gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/C.rpn_stride))
		gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/C.rpn_stride))
		gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/C.rpn_stride))
		gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/C.rpn_stride))

	x_roi = []
	y_class_num = []
	y_class_regr_coords = []
	y_class_regr_label = []
	IoUs = [] # for debugging only
	## R 只有 [bbox, prob] 中的 bbox 对应坐标值
    ## A: layerkinds*w*h, coor， 这里[0]会小于layerkinds*w*h 因为最多300
	for ix in range(R.shape[0]):
		(x1, y1, x2, y2) = R[ix, :]
		x1 = int(round(x1))
		y1 = int(round(y1))
		x2 = int(round(x2))
		y2 = int(round(y2))

		best_iou = 0.0
		best_bbox = -1
		for bbox_num in range(len(bboxes)):
			curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])
			if curr_iou > best_iou:
				best_iou = curr_iou
				best_bbox = bbox_num
		## 选择可以用来最终分类的predict box
		if best_iou < C.classifier_min_overlap: #if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap: 
				continue
		else:
			w = x2 - x1
			h = y2 - y1
			x_roi.append([x1, y1, w, h])
			IoUs.append(best_iou)
			## 难例挖掘，与目标有部分重合但不够判断，分类成背景
			## 一张图里面有再多的类
			if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap: #if best_iou < C.classifier_min_overlap:
				# hard negative example
				cls_name = 'bg'
			elif C.classifier_max_overlap <= best_iou:
				cls_name = bboxes[best_bbox]['class']
				## ground truth box的中心点
				cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
				cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0
				## predict box的中心值
				cx = x1 + w / 2.0
				cy = y1 + h / 2.0

				tx = (cxg - cx) / float(w)
				ty = (cyg - cy) / float(h)
				tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
				th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
			else:
				print('roi = {}'.format(best_iou))
				raise RuntimeError

		class_num = class_mapping[cls_name]
		class_label = len(class_mapping) * [0]
		class_label[class_num] = 1
		y_class_num.append(copy.deepcopy(class_label))
		## -1 是因为背景
		coords = [0] * 4 * (len(class_mapping) - 1)
		labels = [0] * 4 * (len(class_mapping) - 1)
		if cls_name != 'bg':
			label_pos = 4 * class_num
			sx, sy, sw, sh = C.classifier_regr_std
			coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
			labels[label_pos:4+label_pos] = [1, 1, 1, 1]
			y_class_regr_coords.append(copy.deepcopy(coords))
			y_class_regr_label.append(copy.deepcopy(labels))
		else:
			y_class_regr_coords.append(copy.deepcopy(coords))
			y_class_regr_label.append(copy.deepcopy(labels))

	if len(x_roi) == 0:
		return None, None, None, None
	## 感觉像创建label
	X = np.array(x_roi)
	Y1 = np.array(y_class_num)
    ## axis= 1 这样同属于同一个像素的位置的同一个格式anchorbox的label coord 标签在同一行
	Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

	return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs

def apply_regr(x, y, w, h, tx, ty, tw, th):
	try:
		cx = x + w/2.
		cy = y + h/2.
		cx1 = tx * w + cx
		cy1 = ty * h + cy
		w1 = math.exp(tw) * w
		h1 = math.exp(th) * h
		x1 = cx1 - w1/2.
		y1 = cy1 - h1/2.
		x1 = int(round(x1))
		y1 = int(round(y1))
		w1 = int(round(w1))
		h1 = int(round(h1))

		return x1, y1, w1, h1

	except ValueError:
		return x, y, w, h
	except OverflowError:
		return x, y, w, h
	except Exception as e:
		print(e)
		return x, y, w, h

def apply_regr_np(X, T):
	"""

	:param X: [4, h, w]  A
	:param T: [4, h, w] reg
	:return:
	"""
	try:
		x = X[0, :, :]
		y = X[1, :, :]
		w = X[2, :, :]
		h = X[3, :, :]

		tx = T[0, :, :]
		ty = T[1, :, :]
		tw = T[2, :, :]
		th = T[3, :, :]

		## 右边界x坐标
		cx = x + w/2.
		## 下边界y坐标
		cy = y + h/2.
		## 还原得到预测的x,y,w,h
		cx1 = tx * w + cx
		cy1 = ty * h + cy
		##
		w1 = np.exp(tw.astype(np.float64)) * w
		h1 = np.exp(th.astype(np.float64)) * h
		## 得到左x轴 上y轴
		x1 = cx1 - w1/2.
		y1 = cy1 - h1/2.

		x1 = np.round(x1)
		y1 = np.round(y1)
		w1 = np.round(w1)
		h1 = np.round(h1)
		return np.stack([x1, y1, w1, h1])
	except Exception as e:
		print(e)
		return X

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
	# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# calculate the areas
	area = (x2 - x1) * (y2 - y1)

	# sort the bounding boxes 
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		## 选取正样本概率最大的anchor probs就是rpn_layer
		pick.append(i)

		# find the intersection
		## 选取最大的左侧x轴与最上的y轴， 但是会得到len = len（idex）-1 的vector，值全为最大值
		## 实质是把选出的概率最大predict box的坐标与剩下的作比较
		## 若选出box的值比剩下的大，左x上y保留大的，右x下y保留小的
		## 目的在于计算出重合面积
		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
		## 选取最小的右侧x轴与最下的y轴，
		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
		yy2_int = np.minimum(y2[i], y2[idxs[:last]])
		## 选出不符合实际的例子，也计算宽度和高
		ww_int = np.maximum(0, xx2_int - xx1_int)
		hh_int = np.maximum(0, yy2_int - yy1_int)
		## 计算面积
		area_int = ww_int * hh_int

		# find the union
		area_union = area[i] + area[idxs[:last]] - area_int

		# compute the ratio of overlap
		overlap = area_int/(area_union + 1e-6)

		# delete all indexes from the index list that have
		## 首先删除已经被选中的box对应的概率项，已经不需要了
		## 然后删除所有重合面积大于0.9的其他box，因为重复性太高
		## 然后开启下一次的遍历，选出剩余的概率最大的
		idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlap_thresh)[0])))
		## 如果选出了超过300个pos项，则停止
		if len(pick) >= max_boxes:
			break

	# return only the bounding boxes that were picked using the integer data type
	boxes = boxes[pick].astype("int")
	probs = probs[pick]
	return boxes, probs

import time
def rpn_to_roi(rpn_layer, reg_layer, C, use_reg = True, max_boxes=300,overlap_thresh=0.9):
	"""

    :x_class [None, h/16, w/16,9]
    :x_reg [None, h/16, w/16, 4*9]
	"""
	reg_layer = reg_layer / C.std_scaling  #4

	anchor_sizes = C.anchor_box_scales  #[128,256,512]
	anchor_ratios = C.anchor_box_ratios #[[1, 1], [1, 2], [2, 1]]

	assert rpn_layer.shape[0] == 1
	## size of output image
	(rows, cols) = rpn_layer.shape[1:3]

	curr_layer = 0

	## 4， height, width, channel for tensorflow form
	A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))


	for anchor_size in anchor_sizes:
		for anchor_ratio in anchor_ratios:
			## size和ratios的组合一共九种，因此curr_layer也应该是0-8 对应输入 rpn_layer, reg_layer
			## 相应位置的信息
			anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride ##16
			anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride ##16
			## 取出当前图片对应anchor格式的所有位置上的预测reg值
			reg = reg_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
			##  [height, weight, reg_value]
			## =>[4, h, w]
			reg = np.transpose(reg, (2, 0, 1))

			##可以构建坐标值，[0,1]  [0,0]
			##              [0,1]  [1,1]
			X, Y = np.meshgrid(np.arange(cols),np. arange(rows))
			## 构建当前对应位置的anchor box图象中点 对应anchor box的坐标
			## [reg_value（x,y,w,h）,height, weight, curr_layer]
			A[0, :, :, curr_layer] = X - anchor_x/2
			A[1, :, :, curr_layer] = Y - anchor_y/2
			## 保存当前对应位置的anchor box的边长和宽
			A[2, :, :, curr_layer] = anchor_x
			A[3, :, :, curr_layer] = anchor_y

			if use_reg:
				##A 还原了anchor box的坐标，而reg里面都是tx格式的，因此要解回去
				A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], reg)##[reg_value,height, weight]
			## 因为最低1像素，所以必须维持最小值
			A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
			A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
			## ## 得到右x轴 下y轴
			A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
			A[3, :, :, curr_layer] += A[1, :, :, curr_layer]
			## x, y 不能小于边界？
			A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
			A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
			## w，h不能超出图片最大值
			A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
			A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

			curr_layer += 1
	## before A： coor， w,h, layer kinds
	## after A: coor layer kinds w h
	## after after A : coor, layerkinds*w*h
	## after after after A: layerkinds*w*h, coor
	all_boxes = np.reshape(A.transpose((0, 3, 1,2)), (4, -1)).transpose((1, 0))
	## after: [None, 9,h/16, w/16]
	## after after: 压缩成一维的，重复 每个anchor格式对应的w*h
	all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))
	## 提取每个坐标对应的所以anchor上的值
	x1 = all_boxes[:, 0]
	y1 = all_boxes[:, 1]
	x2 = all_boxes[:, 2]
	y2 = all_boxes[:, 3]
	## 选出所有非法值
	idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
	## 删除这些非法值
	all_boxes = np.delete(all_boxes, idxs, 0)
	all_probs = np.delete(all_probs, idxs, 0)

	result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

	return result
