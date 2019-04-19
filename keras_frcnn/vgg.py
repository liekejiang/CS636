import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, TimeDistributed, Flatten,Dense,Dropout,MaxPooling2D
import tensorflow as tf
import sys
sys.path.append("..")
from keras_frcnn.RoiPoolingConv import RoiPoolingConv



def get_weight_path():
    return 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'

def get_img_output_length(width, height):
    
    return width//16, height//16

### used in train_frcnn 158
### input_tensor 是Input（）定义的占位tensor
### trainable = True
def nn_base(input_tensor = None, trainable = False):
    
    # input_shape = (None, None, 3)
    # bn_axis = 3
    #
    # if input_tensor is None:
    #     img_input = keras.layers.Input(shape=input_shape)
    # else:
    #     #if not tf.is_tensor(input_tensor):  for TF2.0
    #     if tf.contrib.framework.is_tensor(input_tensor):
    #         img_input = keras.layers.Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    
    #NN block
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return x

def rpn(base_layers, num_anchors):
    """

    :param base_layers:
    :param num_anchors:

    :return:
    :x_class [None, h/16, w/16,9]
    :x_reg [None, h/16, w/16, 4*9]
    """
    #base_layers maybe the return of nn_base
    x = Conv2D(512, (3,3), padding = 'same', activation = 'relu'
               , kernel_initializer='normal',name='rpn_conv1')(base_layers)
    ## 图象计算到这里经过了4个maxpooling，图象w，h是原图的16分之一，与设定一致，因此与data_generator里的output size相同
    ## 因为这里只是判断每个anchor对应的单个像素点位置（别忘了维度）是否是正样本还是负样本，因此每一个维度的与一种格式的anchor box相一致
    x_class = Conv2D(num_anchors, (1,1), activation='sigmoid', kernel_initializer='uniform', name = 'rpn_out_class')(x)
    ## 这里要判断anchor box的边框回归，因此每个格式的anchor对应4个不同的维度，因此要乘4
    x_reg = Conv2D(num_anchors * 4, (1,1), activation='linear', kernel_initializer='zero', name = 'rpn_out_regress')(x)


    return [x_class, x_reg, base_layers]

def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable = False):
    """
    :param base_layers: output of nn_base
    :param input_rois:  Input(shape = (None,4))
    :param num_rois:  4
    :param nb_classes:
    :param trainable: True
    :return:
    """
    pooling_regions = 7
    input_shape  = (num_rois, 7, 7, 512)
    ## (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels )
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    ### Flatten layer
    out = TimeDistributed(Flatten(name = 'flatten'))(out_roi_pool) ## 1 * num_rois * pool_size^2 * nb_channels
    out = TimeDistributed(Dense(4096, activation = 'relu', name = 'fc1'))(out) ### 4096 output
    out = TimeDistributed(Dropout(0.5))(out)   ##
    out = TimeDistributed(Dense(4096, activation = 'relu', name = 'fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    ###
    out_class = TimeDistributed(Dense(nb_classes, activation = 'softmax', kernel_initializer = 'zero'),
                               name = 'dense_class_{}'.format(nb_classes))(out)
    print(out_class)
    out_reg = TimeDistributed(Dense(4 * (nb_classes-1) , activation = 'linear', kernel_initializer='zero')
                              , name = 'dense_regress_{}'.format(nb_classes))(out)
    print(out_reg)
    
    return [out_class, out_reg]             
