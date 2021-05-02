import tensorflow as tf
import numpy as np
class Convolution():
    
    def __init__(self,filters,kernels,padding,stride,activation) :
       self.filters = filters
       self.kernels = kernels
       self.padding = padding
       self.stride = stride
       self.bias = np.random.normal(size=(3,3))
       self.activation = activation

    def doConvolution(self,inputs,kernels, bias):
        
        (inputs_prev_height, inputs,prev_width) = inputs.shape
        (curr_height, curr_width) = kernels.shape

        output_height = int(int(inputs_prev_height + 2 * self.padding - curr_height) / self.stride + 1)
        output_width = int(int(inputs_prev_height + 2 * self.padding - curr_width) / self.stride + 1)
        
        output = tf.zeros(shape=(output_height,output_width)).numpy()
        padded_inputs = self.zeroPadding(inputs)
        slice_padded_inputs = tf.zeros(shape=(padded_inputs.shape))
        for height in range(output_height):
            vertical_start = self.stride * height
            vertical_end = vertical_start + curr_height
            for width in range(output_width):
                horiz_start = self.stride*width
                horiz_end = horiz_start + curr_width

                slice_padded_inputs = padded_inputs[vertical_start:vertical_end,horiz_start:horiz_end]
                output[height,width] = self.convComputation(slice_padded_inputs,self.kernels,self.bias)

        return output

    def convComputation(self,inputs,Weights, Bias):
        i_w =  tf.multiply(inputs, Weights)
        i_w =  tf.reduce_sum(i_w, axis= None)
        i_w =  i_w + Bias
        return i_w


    def zeroPadding(self,image):
        "Only Works With 2D Image"
        "Padd the whole image "

        return tf.pad(image,( (self.padding, self.padding), (self.padding,self.padding) ))

if __name__ == "__main__":
    tf.random.set_seed(5)
    inputs = tf.random.normal(shape=(5,5))

    weights = tf.random.normal(shape=(3,3))
    bias = tf.random.normal(shape=(1,1))

        