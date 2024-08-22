from .Quantizer import Quantizer
import torch
import torch.nn as nn
import math
from torch.autograd import Function

# LSQ
class LSQ(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        # w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp)) 
        w_q = (torch.div(weight, alpha).round().clamp(Qn, Qp))
        # w_q = w_q * alpha
        return w_q, alpha

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        smaller = (q_w < Qn).float() #bool值转浮点值，1.0或者0.0
        bigger = (q_w > Qp).float() #bool值转浮点值，1.0或者0.0
        between = 1.0 - smaller -bigger #得到位于量化区间的index
        # grad_alpha = ((smaller * Qn + bigger * Qp + 
        #         between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
        # remove grad_weight:
        # grad_alpha = ((smaller * Qn + bigger * Qp + 
        #         between * (q_w.round()) - between * q_w)* g).sum().unsqueeze(dim=0)
        grad_alpha = ((smaller * Qn + bigger * Qp + 
                between * (q_w.round()) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
        #在量化区间之外的值都是常数，故导数也是0
        grad_weight = between * grad_weight
        print(grad_weight)
        return grad_weight, grad_alpha, None, None, None, None

# activation quantizer
class LSQActivationQuantizer(nn.Module):
    def __init__(self, a_bits, Isigned=False):
        super(LSQActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.Isigned = Isigned
        # self.batch_init = batch_init
        if self.Isigned == False:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** self.a_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (self.a_bits - 1)
            self.Qp = 2 ** (self.a_bits - 1) - 1
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.init_state = 0

    # 量化/反量化
    def forward(self, activation):
        '''
        For this work, each layer of weights and each layer of activations has a distinct step size, represented
as an fp32 value, initialized to 2h|v|i/√OP , computed on either the initial weights values or the first
batch of activations, respectively
        '''
        self.g = 1.0/math.sqrt(activation.numel() * self.Qp)
        self.s.data = torch.mean(torch.abs(activation.detach()))*2/(math.sqrt(self.Qp))
        q_a,s_a = LSQ.apply(activation, self.s, self.g, self.Qn, self.Qp)
        # alpha = grad_scale(self.s, g)
        # q_a = Round.apply((activation/alpha).clamp(Qn, Qp)) * alpha
        return q_a,s_a

# weight quantizer  
class LSQWeightQuantizer(nn.Module):
    def __init__(self, w_bits, Wsigned=False):
        super(LSQWeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self.Wsigned = Wsigned
        if self.Wsigned == False:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** w_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (w_bits - 1)
            self.Qp = 2 ** (w_bits - 1) - 1
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    # quantize
    def forward(self, weight):
        self.g = 1.0/math.sqrt(weight.numel() * self.Qp)
        self.s.data = torch.mean(torch.abs(weight.detach()))*2/(math.sqrt(self.Qp))
        w_q,w_a = LSQ.apply(weight, self.s, self.g, self.Qn, self.Qp)
        # alpha = grad_scale(self.s, g)
        # w_q = Round.apply((weight/alpha).clamp(Qn, Qp)) * alpha
        return w_q,w_a


class LSQuantizer(Quantizer):
    
    def __init__(self):
        super(LSQuantizer, self).__init__()
        self.activation_quantizer = LSQActivationQuantizer(a_bits=self.input_precision, Isigned=self.Isigned)
        self.weight_quantizer = LSQWeightQuantizer(w_bits=self.weight_precision, Wsigned=self.Wsigned)

    def weight_init(self, weight, bits_W=None,factor=2.0, mode="fan_in"):
        scale = 1.0
        return scale    
    
    def update_range(self, input):
        pass
    
    def input_clamp(self, input):
        return input  
                      
    def QuantizeWeight(self, weight, bits=None, Wsigned=True):
        
        weight,weightscale = self.weight_quantizer(weight)
        weightrange = []
        weightshift = 0.0
        
        return weight, weightscale, weightrange, weightshift 

    def QuantizeInput(self, input, bits=None, Isigned=True):
        
        input,inputscale = self.activation_quantizer(input) 
        inputrange = []
        inputshift = 0.0    
        
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
