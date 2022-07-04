from typing import Concatenate
import tensorflow as tf
from tensorflow.keras.layers import *

def conv_bn_relu(inp, kernels, kernel_size = 3, stride = 1, data='2d', bn_relu = True):
    x = inp
    if data == '2d':
        x = Conv2D(kernels, 
                    kernel_size = kernel_size,
                    padding = 'same'
                    strides = stride)(x)
    else:
        x = Conv3D(kernels, 
                    kernel_size = kernel_size,
                    padding = 'same'
                    strides = stride)(x)
    if bn_relu:
        x = BatchNormalization()(x)
        x = ReLU()(x)
    return x
''
def conv_block(inp, kernels, data='2d', downsampling=False):
    x = inp
    if downsampling:
        x = conv_bn_relu(x, kernels, 1, 1, data)
    x = conv_bn_relu(x, kernels, data=data)
    x = conv_bn_relu(x, kernels, data=data)
    return x

def residual_conv_block(inp, kernels, repeat = 2, data='2d', downsampling=False):
    '''
        Residual convolution block
    '''
    x = inp
    for i in range(repeat):
        stride = 1
        if i == 0 and downsampling:
            stride = 2

        skip_conn = conv_bn_relu(x, kernels, 1, stride, data, False)

        x = conv_bn_relu(x, kernels, 3, stride, data)
        x = conv_bn_relu(x, kernels, data, bn_relu=False)
        
        # Residual connection
        x = Add()([x, skip_conn])
        x = BatchNormalization()(x)
        x = ReLU()(x)
    return x

def squeeze_exite(inp, kernels, ratio = 16, data='2d'):
    x = inp
    if data == '2d':
        x = GlobalAveragePooling2D(keepdims=True)(x)
    else:
        x = GlobalAveragePooling3D(keepdims=True)(x)
    x = Dense(kernels // ratio)(x)
    x = ReLU()(x)
    x = Dense(kernels)(x)
    x = Sigmoid()(x)
    return Multiply()([inp, x])

def seresnet_block(inp, kernels, repeat=2, data='2d', downsampling=True):
    '''
        Residual convolution block with squeeze and exitation
    '''
    x = inp
    for i in range(repeat):
        stride = 1
        if i == 0 and downsampling:
            stride = 2

        skip_conn = conv_bn_relu(x, kernels, 1, stride, data, False)

        x = conv_bn_relu(x, kernels, 3, stride, data)
        x = conv_bn_relu(x, kernels, data, bn_relu=False)
        x = squeeze_exite(x, kernels, data=data)
        
        # Residual connection
        x = Add()([x, skip_conn])
        x = BatchNormalization()(x)
        x = ReLU()(x)

    return x

def up_conv_block(inp, kernels, connect, data='2d'):
    x = inp
    if data == '2d':
        x = UpSampling2D()(x)
    else:
        x = UpSampling3D()(x)
    x = Concatenate(axis=-1)([x, connect])
    x = conv_block(x, kernels, data=data, downsampling=False)
    return x

def up_residual_conv_block(inp, kernels, connect, repeat=2, data='2d'):
    x = inp
    if data == '2d':
        x = UpSampling2D()(x)
    else:
        x = UpSampling3D()(x)
    x = Concatenate(axis=-1)([x, connect])
    x = residual_conv_block(x, kernels, repeat, data=data, downsampling=False)
    return x

def up_seresnet_conv_block(inp, kernels, connect, repeat=2, data='2d'):
    x = inp
    if data == '2d':
        x = UpSampling2D()(x)
    else:
        x = UpSampling3D()(x)
    x = Concatenate(axis=-1)([x, connect])
    x = seresnet_block(x, kernels, repeat, data=data, downsampling=False)
    return x
