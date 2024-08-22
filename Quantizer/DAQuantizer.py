from .Quantizer import Quantizer
import torch
import torch.nn as nn
import math

class DAQuantizer(Quantizer):
    
    def __init__(self):
        super(DAQuantizer, self).__init__()
        self.InputMin  = nn.Parameter(torch.tensor(-1.0, requires_grad=False))
        self.InputMax  = nn.Parameter(torch.tensor(1.0, requires_grad=False))
        self.inputMomentum = 0.1

    def weight_init(self, weight, bits_W=None,factor=2.0, mode="fan_in"):
        if bits_W is None:
            bits_W = self.weight_precision
        
        if mode != "fan_in":
            raise NotImplementedError("support only wage normal")
        dimensions = weight.ndimension()
        if dimensions < 2: raise ValueError("weight at least is 2d")
        elif dimensions == 2: fan_in = weight.size(1)
        elif dimensions > 2:
            num_input_fmaps = weight.size(1)
            receptive_field_size = 1
            if weight.dim() > 2:
                receptive_field_size = weight[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
        # This is a magic number, copied
        float_limit = math.sqrt( 3 * factor / fan_in)
        float_std =  math.sqrt( 2 / fan_in)
        quant_limit,scale = self.scale_limit(float_std, bits_W)
        weight.data.uniform_(-quant_limit, quant_limit)

        print("fan_in {:6d}, float_limit {:.6f}, float std {:.6f}, quant limit {}, scale {}".format(fan_in, float_limit, float_std, quant_limit, scale))
        return scale    
    
    def update_range(self, input):
        self.InputMin.data = self.inputMomentum * self.InputMin.data + (1-self.inputMomentum) * input.min().item()
        self.InputMax.data = self.inputMomentum * self.InputMax.data + (1-self.inputMomentum) * input.max().item()
        return None
    
    def input_clamp(self, input):
        input = torch.clamp(input, self.InputMin.data ,self.InputMax.data )
        return input   
                      
    def QuantizeWeight(self, weight, bits=None, Wsigned=True):
        if bits is  None:
            bits= self.weight_precision
            
        range = [-1+1/2**(self.weight_precision-1),1-1/2**(self.weight_precision-1)]
               
        weight, weightscale, weightrange, weightshift = self.Q(weight,self.weight_precision,signed=Wsigned, 
                                                          fixed_range=range,
                                                          odd_stage=True)
        
        return weight, weightscale, weightrange, weightshift 

    def QuantizeInput(self, input, bits=None, Isigned=True):
        if bits is  None:
            bits= self.input_precision
            
        range = [self.InputMin.data.item(), self.InputMax.data.item()]     
          
        input, inputscale, inputrange,inputshift = self.Q(input,self.input_precision,signed=Isigned, 
                                                          fixed_range=range,
                                                          odd_stage=True)
        
        return input, inputscale, inputrange, inputshift
    
    def QuantizeError(self, error, bits=None, Esigned=True):
        if bits is  None:
            bits= self.error_precision
               
        error, errorscale, errorrange,errorshift = self.Q(error,self.error_precision,signed=Esigned, 
                                                          fixed_range=range,
                                                          odd_stage=True)
        
        return error, errorscale, errorrange, errorshift 

    def quantize_grad(self, x): 
        raise NotImplementedError("use QSGD")
