import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
import gc
'''
    3D-UNET
'''
class ResUnet3D:
    def __init__(self, num_classes = 3, 
                 input_shape = (None, None, None, 1),
                 conv_settings = [16, 32, 64, 128, 256], 
                 repeat = [2, 2, 2, 2, 2],
                 deep_supervision = False,
                 activation = 'sigmoid'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_settings = conv_settings
        self.repeat = repeat
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
            x = ReLU()(x)
        return x
    
    def residual_conv_block(self, inp, kernels, repeat = 2, downsample = False):
        '''
            Residual convolution block
        '''
        x = inp
        for i in range(repeat):
            stride = 1
            if i == 0 and downsample:
                stride = 2

            skip_conn = self.conv_in_relu(x, kernels, 1, stride, False)

            x = self.conv_in_relu(x, kernels, 3, stride)
            x = self.conv_in_relu(x, kernels, bn_relu=False)
            
            # Residual connection
            x = Add()([x, skip_conn])
            x = BatchNormalization()(x)
            x = ReLU()(x)
        return x

    def up_conv_block(self, inp, kernels, connect):
        '''
            Decoder block
        '''
        x = UpSampling3D()(inp)
        x = Concatenate(axis=-1)([x, connect]) # Skip connection
        x = self.residual_conv_block(x, kernels)
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
                conv = self.residual_conv_block(conv, conv_settings[i], self.repeat[i])
                encoder_blocks.append(conv)
            else:
                conv = self.residual_conv_block(conv, conv_settings[i], self.repeat[i], downsample = True)
                encoder_blocks.append(conv)
                
        out = self.residual_conv_block(conv, conv_settings[-1], self.repeat[-1], downsample = True)

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
        out = Activation(self.activation, name=f'output_final')(out)
        outputs.append(out)
        
        model = Model(inputs = inp, outputs = outputs)
        return model

if __name__ == '__main__':
    K.clear_session()
    gc.collect()
    model = Unet3D()()
    model.summary(line_length=150)