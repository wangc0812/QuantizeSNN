# read configuration file as global settings
import configparser
import os

QuantizationMode  =   "LSQ"


import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np

if QuantizationMode == "WAGE" :
    from Quantizer.WAGEQuantizer import WAGEQuantizer as Quantizer
if QuantizationMode == "WAGEV2" :
    from Quantizer.WAGEV2Quantizer import WAGEV2Quantizer as Quantizer    
if QuantizationMode == "DynamicFixedPoint":
    from Quantizer.DFQuantizer import DFQuantizer as Quantizer
if QuantizationMode == "DynamicAffine":
    from Quantizer.DAQuantizer import DAQuantizer as Quantizer
if QuantizationMode == "LSQ":
    from Quantizer.LSQuantizer import LSQuantizer as Quantizer
if QuantizationMode == "LSQv2":
    from Quantizer.LSQv2Quantizer import LSQPlusuantizer as Quantizer       




class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,bias=False,
                 quantize_weight=True, quantize_input= True, quantize_error=True,
                 name ='Qconv2d'):

        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.name = name
        self.quantizer = Quantizer() 
        self.scale = self.quantizer.weight_init(self.weight, factor=1.0)
        self.quantize_weight = quantize_weight
        self.quantize_input = quantize_input
        self.quantize_error = quantize_error

    def forward(self, input):
        
        
        if self.quantize_input:
            if self.training:
                self.quantizer.update_range(input) 
            input = self.quantizer.input_clamp(input)
        output= CIM_Conv.apply(input, self.weight, self.bias, self.stride[0], self.padding[0], self.dilation, self.groups, self.scale,
                              self.quantize_weight,self.quantize_input,self.quantize_error, self.name, self.quantizer, self.training)
        return output

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'kernel_size={}, in_channels={}, out_channels={}, stride={}, bias={}, quantize_weight={}, quantize_input={}, quantize_error={}'.format(
            self.kernel_size,self.in_channels, self.out_channels,self.stride,self.bias is not None,self.quantize_weight , self.quantize_input, self.quantize_error
        )


class CIM_Conv(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, stride = 1, padding = 1,dilation=1, groups=1, fixed_scale=1,
                quantize_weight =True,quantize_input =True,quantize_error =True, name = 'CONV',quantizer=None,training=None):
        ctx.save_for_backward(input, weight, bias)
        ctx.padding = padding
        ctx.stride = stride
        ctx.quantizer = quantizer
        ctx.quantize_weight = quantize_weight
        ctx.quantize_input = quantize_input
        ctx.quantize_error = quantize_error
        ctx.fixed_scale = fixed_scale
        ctx.name = name
        ctx.training = training
        inputscale  = 1
        weightscale = 1
        inputshift  = 0
        weightshift = 0
        
        if quantize_weight:
            #mapping quantized weight to the code book
            weight, weightscale, weightrange, weightshift = quantizer.QuantizeWeight(weight=weight, Wsigned=True,train=training)

        if quantize_input:
            #mapping quantized weight to the code book
            input,  inputscale, inputrange, inputshift =  quantizer.QuantizeInput(input=input, Isigned=True,train=training)

       
        output = INTConv(input, inputshift, weight, weightshift, bias, stride, padding, dilation, groups)

        return output*weightscale*inputscale/fixed_scale
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias= ctx.saved_tensors
        padding = ctx.padding
        quantizer = ctx.quantizer
        inputscale = 1
        weightscale = 1
        grad_outputscale = 1
        grad_bias = grad_input = grad_weight = None
        pad_m = nn.ConstantPad2d(padding, 0)
        grad_input = torch.zeros_like(input)
        grad_input_dummy = torch.zeros_like(input)
        dummy_grad_output = torch.zeros_like(grad_output)
        dummy_weight = torch.zeros_like(weight)
        dummy_input = torch.zeros_like(input)
        grad_weight_dummy1 = torch.zeros_like(weight)
        grad_weight_dummy2 = torch.zeros_like(weight)
        grad_weight_dummy3 = torch.zeros_like(weight)
        inputshift = 0
        grad_outputshift = 0
        weightshift = 0

        if ctx.stride != 1:
            grad_output = pad_within(grad_output,ctx.stride)

        dummy_grad_output = torch.zeros_like(grad_output)
        if ctx.quantize_weight:
            #TODO: Pay attention to the range
            weight, weightscale, weightrange, weightshift = quantizer.QuantizeWeight(weight=weight,Wsigned=True,train=ctx.training)

            dummy_weight = torch.ones_like(weight)
        if ctx.quantize_input:
            #TODO: Pay attention to the range
            input,  inputscale, inputrange, inputshift = quantizer.QuantizeInput(input=input,Isigned=True,train=ctx.training) 

            dummy_input = torch.ones_like(input)

        if ctx.quantize_error:
            #np.save(Logdir+'/'+ctx.name+'_error.npy', input.cpu().numpy().dataq
            # grad_output = torch.clamp(grad_output/grad_output.abs().max(), -1, 1)
            #TODO: Pay attention to the range
            
            grad_output, grad_outputscale, grad_outputrange, grad_outputshift = quantizer.QuantizeError(error=grad_output,Esigned=True)
        
            dummy_grad_output = torch.ones_like(grad_output)

        weight_t = weight.transpose(0, 1)
        dummy_weight_t = dummy_weight.transpose(0, 1)
        filter_size = weight_t.size()

        input_size = input.size()
        grad_output_size = grad_output.size()

        if ctx.needs_input_grad[0]:
            for i in range(filter_size[2]):
                for j in range(filter_size[3]):
                    current_filter = weight_t[:,:,filter_size[2]-1-i,filter_size[3]-1-j]
                    current_filter_dummy = dummy_weight_t[:,:,filter_size[2]-1-i,filter_size[3]-1-j]

                    output = torch.mm(current_filter,grad_output.transpose(0,1).reshape(grad_output_size[1],-1)).reshape(input_size[1],input_size[0],input_size[2],input_size[3]).transpose(0,1)
                    output_dummy1 = torch.mm(current_filter,
                                      dummy_grad_output.transpose(0, 1).reshape(grad_output_size[1], -1)).reshape(
                        input_size[1], input_size[0], input_size[2], input_size[3]).transpose(0, 1) * grad_outputshift
                    output_dummy2 = torch.mm(current_filter_dummy,
                                      grad_output.transpose(0, 1).reshape(grad_output_size[1], -1)).reshape(
                        input_size[1], input_size[0], input_size[2], input_size[3]).transpose(0, 1) * weightshift
                    output_dummy3 = torch.mm(current_filter_dummy,
                                             dummy_grad_output.transpose(0, 1).reshape(grad_output_size[1], -1)).reshape(
                        input_size[1], input_size[0], input_size[2], input_size[3]).transpose(0, 1) * weightshift * grad_outputshift


                    partial_output_dec = pad_m(output)
                    partial_output_dec_dummy1 = pad_m(output_dummy1)
                    partial_output_dec_dummy2 = pad_m(output_dummy2)
                    partial_output_dec_dummy3 = pad_m(output_dummy3)

                    grad_input = grad_input + partial_output_dec[:, :, i:i + input_size[2], j:j + input_size[3]]
                    grad_input_dummy = grad_input_dummy + partial_output_dec_dummy1[:, :, i:i + input_size[2], j:j + input_size[3]]
                    grad_input_dummy = grad_input_dummy + partial_output_dec_dummy2[:, :, i:i + input_size[2], j:j + input_size[3]]
                    grad_input_dummy = grad_input_dummy + partial_output_dec_dummy3[:, :, i:i + input_size[2], j:j + input_size[3]]

        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)
            padded_input = pad_m(input)
            padded_input_dummy = pad_m(dummy_input)
            grad_output_w = (grad_output).transpose(0,1).reshape(grad_output_size[1], -1)
            grad_output_w_dummy = (dummy_grad_output).transpose(0,1).reshape(grad_output_size[1], -1)

            for i in range(filter_size[2]):
                for j in range(filter_size[3]):
                    input_window = padded_input[:, :, i:i + input_size[2], j:j + input_size[3]].transpose(0,1).reshape(input_size[1], -1).transpose(0, 1)
                    grad_weight[:, :, i, j] = torch.mm(grad_output_w,input_window)
                    #input_window = padded_input[:, :, i:i + input_size[2], j:j + input_size[3]].transpose(0,1).reshape(input_size[1], -1).transpose(0, 1)
                    grad_weight_dummy1[:, :, i, j] = torch.mm(grad_output_w_dummy,input_window)*grad_outputshift

                    input_window = padded_input_dummy[:, :, i:i + input_size[2], j:j + input_size[3]].transpose(0,1).reshape(input_size[1], -1).transpose(0, 1)
                    grad_weight_dummy2[:, :, i, j] = torch.mm(grad_output_w,input_window)*inputshift
                    #input_window = padded_input[:, :, i:i + input_size[2], j:j + input_size[3]].transpose(0,1).reshape(input_size[1], -1).transpose(0, 1)
                    grad_weight_dummy3[:, :, i, j] = torch.mm(grad_output_w_dummy,input_window)*grad_outputshift*inputshift
        # return (grad_input+grad_input_dummy)*weightscale*grad_outputscale , (grad_weight+grad_weight_dummy1+grad_weight_dummy2+grad_weight_dummy3)*inputscale*grad_outputscale, grad_bias, None, None, None, None, None, None, None, None, None, None, None
        # print("wg=",(grad_weight+grad_weight_dummy1+grad_weight_dummy2+grad_weight_dummy3)*inputscale*grad_outputscale)
        return (grad_input+grad_input_dummy)*weightscale*grad_outputscale/ctx.fixed_scale , (grad_weight+grad_weight_dummy1+grad_weight_dummy2+grad_weight_dummy3)*inputscale*grad_outputscale, grad_bias, None, None, None, None, None, None, None, None, None, None, None


def pad_within(x, stride=2):
    w = x.new_zeros(stride, stride)
    w[0, 0] = 1
    return F.conv_transpose2d(x, w.expand(x.size(1), 1, stride, stride), stride=stride, groups=x.size(1))

def INTConv(input, inputshift, weight, weightshift, bias, stride, padding, dilation, groups):
     output = F.conv2d(input, weight, bias, stride, padding, dilation, groups).detach()
     if inputshift != 0:
          dummy_input = torch.ones_like(input)
          output += inputshift * F.conv2d(dummy_input, weight, bias, stride, padding, dilation, groups).detach()
     if weightshift != 0:
          dummy_weight = torch.ones_like(weight)
          output += weightshift * F.conv2d(input, dummy_weight, bias, stride, padding, dilation, groups).detach()
     if inputshift != 0 and weightshift != 0 :
          dummy_input = torch.ones_like(input)
          dummy_weight = torch.ones_like(weight)
          output += inputshift * weightshift * F.conv2d(dummy_input, dummy_weight, bias, stride, padding, dilation,
                                                        groups).detach()
     return output
 
 