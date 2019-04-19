import cv2
import numpy as np
import copy

def augment(img_data, config, augment = True):
    """
    input is a single element in all_imgs, which is a dict created in pascal_voc_parser
    all_imgs :[file_path, w, h, bbox]
    bbox: [class, x1,x2,y1,y2,difficulty]

    train: augment = True else False
    """
    assert 'filepath' in img_data
    assert 'bboxes'in img_data
    assert 'width' in img_data
    assert 'height' in img_data
    
    #深复制，和python机制有关
    img_data_aug = copy.deepcopy(img_data)
    img = cv2.imread(img_data_aug['filepath'])
    
    if augment:
        rows, cols = img.shape[:2]
        #第二个判断条件是随机生成的，也即随机对数据进行各种操作 以增强数据集
        #第二个判断条件生成1的时候不进行任何操作
        if config.use_horizontal_flips and np.random.randint(0,2) == 0:
            img = cv2.flip(img,1)
            #图片水平翻转了 box也得跟着转
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2
                
        if config.use_vertical_flips and np.random.randint(0,2) == 1:
            img = cv2.flip(img,0)
            #图片水平翻转了 box也得跟着转
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2            
            
        #顺时针旋转九十°        
        if config.rot_90:
            #在这4个数里面选随机选择一个
            angle = np.random.choice([0,90,270,180],1)[0]
            if angle == 270:
                #转置矩阵，但是是镜像转置，需要翻转才能是270°
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img,0)
            elif angle == 90:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img,1)        
            elif angle == 180:
                img = cv2.flip(img, -1)
                
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = cols - x2
                    bbox['y2'] = cols - x1        
                elif angle == 90:
                    bbox['x1'] = rows - y2
                    bbox['x2'] = rows - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2       
                elif angle == 180:
                    bbox['x1'] = cols - x2
                    bbox['x2'] = cols - x1
                    bbox['y1'] = rows - y2
                    bbox['y2'] = rows - y1
    
    #宽和高颠倒是cv2的设置，有毛病
    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]
    
    #可能更新了图片的形状 所以要更新
    return img_data_aug, img
