import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy as np
import time

class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)
        
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'
        
class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size, X_mean, num_class = 2, output_last = False):
        """
        GRU-D: Adaptation of standard GRU to deal with irregularly spaced multivariate time series data
        
        Implemented based on the paper: 
        @article{che2018recurrent,
          title={Recurrent neural networks for multivariate time series with missing values},
          author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
          journal={Scientific reports},
          volume={8},
          number={1},
          pages={6085},
          year={2018},
          publisher={Nature Publishing Group}
        }
        
        GRU-D:
            input_size: variable dimension of each time
            hidden_size: dimension of hidden_state
            mask_size: dimension of masking vector
            X_mean: the mean of the training data
        """
        
        super(GRUD, self).__init__()
        
        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        self.num_class = num_class
        
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zeros = Variable(torch.zeros(input_size).cuda())
            self.X_mean = Variable(torch.Tensor(X_mean).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(input_size))
            self.X_mean = Variable(torch.Tensor(X_mean))
        
        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
  
        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity)
        self.gamma_h_l = nn.Linear(self.delta_size, hidden_size)
        self.binary_final_fcl =  nn.Linear(hidden_size, 1)
        self.multi_final_fcl =  nn.Linear(hidden_size, num_class)
        self.sigmoid = nn.Sigmoid()
        self.output_last = output_last

    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        
        batch_size = x.shape[0]
        dim_size = x.shape[1]
        
        # Set of weights - Delta of the measured weights and delta of the Hidden state
        
        delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta)))
      
        delta_h = torch.exp(-torch.max(torch.zeros(self.hidden_size), self.gamma_h_l(delta)))
        
        # Deterimines what measurement to present to the step
        # If there is a measurement present mask (1) * x will present the measurement if present
        #   OR if not present then the opposite will happen (1-mask) activates 2nd half of equation  
        # 2nd half computes the difference Last_obs and its mean. If Delta close to one then closer to last_obs 
        
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        
        # Decay term to for the 'missingness' for the hidden state
      
        h = delta_h * h
        combined = torch.cat((x, h, mask), 1).to(torch.float32)
        z = F.sigmoid(self.zl(combined))
        r = F.sigmoid(self.rl(combined))
        # Reset gate
        combined_r = torch.cat((x, r * h, mask), 1).to(torch.float32) 
      
        # Candidate hidden state
        h_tilde = F.tanh(self.hl(combined_r)) 
      
        # Update gate
        h = (1 - z) * h + z * h_tilde 
        
        return h
    
    def forward(self, input):
        batch_size = input.size(0)
        type_size = input.size(1)
        step_size = input.size(2)
        spatial_size = input.size(3)
        
        Hidden_State = self.initHidden(batch_size)
        # Extract from Concated X,X_last_obvs, mask, Delta
        X = input[:,0,:,:]
        X_last_obsv = input[:,1,:,:]
        Mask = input[:,2,:,:]
        Delta = input[:,3,:,:]
        
        outputs = None
        for i in range(step_size):
            # Make sure you squeeze right dimention (if batch = 1, then it will also squeeze batch dim)
            Hidden_State = self.step(torch.squeeze(X[:,i:i+1,:], dim = 1)\
                                    , torch.squeeze(X_last_obsv[:,i:i+1,:], dim = 1)\
                                    , torch.squeeze(self.X_mean[:,i:i+1,:], dim = 1)\
                                    , Hidden_State\
                                    , torch.squeeze(Mask[:,i:i+1,:], dim = 1)\
                                    , torch.squeeze(Delta[:,i:i+1,:], dim = 1))
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
        if self.num_class == 2:        #i.e binary classification
            if self.output_last:
                # Depends if this is just last recurrent
                # Add FCL with Sigmoid activation
                out = self.binary_final_fcl(outputs[:,-1,:])
                return self.sigmoid(out).to(torch.float32)
            else:
                out = self.binary_final_fcl(outputs)
                return  self.sigmoid(out).to(torch.float32)
            
        else: # Return logits for cross_entropy
            if self.output_last:
                return self.multi_final_fcl(outputs[:,-1,:])
            else:
                return  self.multi_final_fcl(outputs)
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State
        
