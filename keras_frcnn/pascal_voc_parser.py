import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
def get_data(input_path):
	all_imgs = []

	classes_count = {}

	class_mapping = {}

	visualise = False
	###同时处理2007和2012？
	#data_paths = [os.path.join(input_path,s) for s in ['VOC2007', 'VOC2012']]
	data_paths = [os.path.join(input_path,s) for s in ['VOC2012']]

	print('Parsing annotation files')
	### path+VOC2007, path+VOC2012
	for data_path in data_paths:
		##Annotations 存放bounding box信息
		annot_path = os.path.join(data_path, 'Annotations')
		##JPEG 存放图片
		imgs_path = os.path.join(data_path, 'JPEGImages')
		##ImageSets存放分类信息
		imgsets_path_trainval = os.path.join(data_path, 'ImageSets','Main','trainval.txt')
		## test原本不存在 是要自己建立吗 懂了 大部分没有test.file
		imgsets_path_test = os.path.join(data_path, 'ImageSets','Main','test.txt')

		trainval_files = []
		test_files = []

		try:
			with open(imgsets_path_trainval) as f:
				for line in f:
					trainval_files.append(line.strip() + '.jpg')
		except Exception as e:
			print(e)

		try:
			with open(imgsets_path_test) as f:
				for line in f:
					test_files.append(line.strip() + '.jpg')
		except Exception as e:
			if data_path[-7:] == 'VOC2012':
				# this is expected, most pascal voc distibutions dont have the test.txt file
				pass
			else:
				print(e)

		##读取annotations 里面的每一个文件名到annots里面
		annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
		##读取数量计数器
		idx = 0
		##
		for annot in annots:
			try:
				idx += 1
				## read the information in the xml
				et = ET.parse(annot)
				element = et.getroot()
				element_objs = element.findall('object')
				element_filename = element.find('filename').text
				element_width = int(element.find('size').find('width').text)
				element_height = int(element.find('size').find('height').text)

				if len(element_objs) > 0:
					annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
									   'height': element_height, 'bboxes': []}
					if element_filename in trainval_files:
						annotation_data['imageset'] = 'trainval'
					elif element_filename in test_files:
						annotation_data['imageset'] = 'test'
					else:
						annotation_data['imageset'] = 'trainval'


				for element_obj in element_objs:
					class_name = element_obj.find('name').text

					## 记录每个类出现了多少次
					if class_name not in classes_count:
						classes_count[class_name] = 1
					else:
						classes_count[class_name] += 1
					## 一种哈希映射？？？
					## 这里不懂
					if class_name not in class_mapping:
						class_mapping[class_name] = len(class_mapping)

					## 得到bbox坐标
					obj_bbox = element_obj.find('bndbox')
					x1 = int(round(float(obj_bbox.find('xmin').text)))
					y1 = int(round(float(obj_bbox.find('ymin').text)))
					x2 = int(round(float(obj_bbox.find('xmax').text)))
					y2 = int(round(float(obj_bbox.find('ymax').text)))
					## 判断是不是难例挖掘？
					difficulty = int(element_obj.find('difficult').text) == 1
					annotation_data['bboxes'].append(
						{'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
				all_imgs.append(annotation_data)

				##视觉显示bbox
				if visualise:
					img = cv2.imread(annotation_data['filepath'])
					for bbox in annotation_data['bboxes']:
						cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
									  'x2'], bbox['y2']), (0, 0, 255))
					cv2.imshow('img', img)
					cv2.waitKey(0)

			except Exception as e:
				print(e)
				continue
	return all_imgs, classes_count, class_mapping

##最疑惑的还是class_mapping， 每增加一个key，len（dict）会同样增加，因此每个类都有一一对应的数字，另一种hash

## all_imgs 是一个由哈希表组成的列表，每个哈希元素包括了该图片文件的路径，图片的高度和宽度和一个名为bboxes的子哈希表，bboxes中包括了被选中的图片的类别，4坐标和是否为难例
## classes_count 记录了一共有多少种类别和相关数目
## class_mapping 为每个类记录了一种数字映射
