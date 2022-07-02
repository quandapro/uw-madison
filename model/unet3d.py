import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
import gc
'''
    3D-UNET
'''
class Unet3D:
    def __init__(self, num_classes = 3, 
                 input_shape = (None, None, None, 1),
                 conv_settings = [16, 32, 64, 128, 256], 
                 deep_supervision = True,
                 activation = 'sigmoid'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_settings = conv_settings
        self.deep_supervision = deep_supervision
        self.activation = activation
    
    def conv_in_relu(self, inp, kernels, kernel_size = 3, stride = 1, bn_relu = True):
        x = inp
        x = Conv3D(kernels, 
                    kernel_size = kernel_size,
                    padding = 'same',
                    strides = stride)(x)
        if bn_relu:
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.01)(x)
        return x
    
    def conv_block(self, inp, kernels, downsample = False):
        '''
            Basic convolution block vgg styles
        '''
        x = inp
        if downsample:
            x = self.conv_in_relu(x, kernels, kernel_size=1, stride=2)
        x = self.conv_in_relu(x, kernels, kernel_size=3, stride=1)
        x = self.conv_in_relu(x, kernels, kernel_size=3, stride=1)
        return x

    def up_conv_block(self, inp, kernels, connect):
        x = UpSampling3D()(inp)
        x = Concatenate(axis=-1)([x, connect]) # Skip connection
        x = self.conv_in_relu(x, kernels)
        x = self.conv_in_relu(x, kernels)
        return x

    def __call__(self):
        conv_settings = self.conv_settings
        
        num_blocks = len(conv_settings)
        
        inp = Input(self.input_shape)

        outputs = []

        encoder_blocks = []

        # Encoder
        conv = BatchNormalization()(inp)
        
        for i in range(0, num_blocks - 1):
            if i == 0:
                conv = self.conv_block(conv, conv_settings[i])
                encoder_blocks.append(conv)
            else:
                conv = self.conv_block(conv, conv_settings[i], downsample = True)
                encoder_blocks.append(conv)
                
        out = self.conv_block(conv, conv_settings[-1], downsample = True)

        # Decoder
        for i in range(num_blocks - 1, 0, -1):
            out = self.up_conv_block(out, conv_settings[i - 1], encoder_blocks[i - 1])
            if self.deep_supervision and 1 < i < 4:
                pool_size = 2**(i - 1)
                pred = Conv3D(self.num_classes, 
                            kernel_size = (1, 1, 1), 
                            padding = 'same')(out)
                pred = UpSampling3D(pool_size)(pred)
                pred = Activation(self.activation, name = f'output_{i}')(pred)
                outputs.append(pred)

        out = Conv3D(self.num_classes, 
                    kernel_size = (1, 1, 1), 
                    padding = 'same')(out)
        out = Activation(self.activation)(out)
        outputs.append(out)

        model = Model(inputs = inp, outputs = outputs)
        return model

if __name__ == '__main__':
    K.clear_session()
    gc.collect()
    model = Unet3D()()
    model.summary(line_length=150)