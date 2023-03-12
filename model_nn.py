"""
Augmented Physics-Based Models

This file is part of GNSSjamLoc. 

Additional information can be found at https://doi.org/10.48550/arXiv.2212.08097

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Copyright (C) 2023 Andrea Nardin <andrea.nardin@polito.it>
Navigation, Signal Analysis and Simulation (NavSAS) group,
Politecnico di Torino 
Copyright (C) 2023 Peng Wu, Tales Imbiriba, Pau Closas
Signal Processing Imaging Reasoning and Learning (SPIRAL) Lab
Northeastern University
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
import numpy as np
import functions as f


class Net(nn.Module):
    
    def __init__(self, input_dim, layer_wid, nonlinearity):
        """
        """
        super(Net, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = layer_wid[-1]
        
        self.fc_layers = nn.ModuleList()
        
        self.fc_layers.append(nn.Linear(in_features=input_dim, 
                                        out_features=layer_wid[0]))
        
        for i in range(len(layer_wid) - 1):
            self.fc_layers.append(nn.Linear(in_features=layer_wid[i], 
                                            out_features=layer_wid[i + 1]))

        # Set the nonlinearity
        if nonlinearity == "sigmoid":
            self.nonlinearity = lambda x: torch.sigmoid(x)
        elif nonlinearity == "relu":
            self.nonlinearity = lambda x: F.relu(x)
        elif nonlinearity == "softplus":
            self.nonlinearity = lambda x: F.softplus(x)
        elif nonlinearity == 'tanh':
            self.nonlinearity = lambda x: torch.tanh(x) #F.tanh(x)
        elif nonlinearity == 'leaky_relu':
            self.nonlinearity = lambda x: F.leaky_relu(x)
        elif nonlinearity == 'softplus':
            self.nonlinearity = lambda x: F.softplus(x)
        elif nonlinearity == 'leaky_relu':
            self.nonlinearity = lambda x: F.leaky_relu(x)

    def forward(self, x):
        """
        :param x: input with dimensions N x input_dim where N is number of
            inputs in the batch.
        """
        
        for fc_layer in self.fc_layers[:-1]:
            x = self.nonlinearity(fc_layer(x))
            
        return self.fc_layers[-1](x)
    
    def get_layers(self):
        L = len(self.fc_layers)
        layers = (L+1)*[0]
        layers[0] = self.fc_layers[0].in_features
        for i in range(L):
            layers[i+1] = self.fc_layers[i].out_features
        return layers
    
    def get_param(self):
        """
        Returns a tensor of all the parameters
        """
        P = torch.tensor([])
        for p in self.parameters():
            a = p.clone().detach().requires_grad_(False).reshape(-1)
            P = torch.cat((P,a))
        return P
    
    
class Polynomial3(torch.nn.Module):
    def __init__(self,gamma=2,theta0=None,data_max=None,data_min=None):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        if theta0 == None:
            self.theta = nn.Parameter(torch.zeros((2))) # 2-dimensional jammer position. Param to be learned
        else:
            self.theta = nn.Parameter(torch.tensor(theta0))
                        
        self.gamma = gamma
        #self.gamma = nn.Parameter(torch.randn(())) #try to learn it if it can converge
        
        self.P0 = nn.Parameter(torch.randn(())) # tx power. It should be learned
        # self.P0 = 10
        
        self.data_max = data_max
        self.data_min = data_min
        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        L = self.gamma*10*torch.log10(torch.norm(x-self.theta,p=2,dim=1))
        
        # when points are too close L<0 are found. Workaround set L=0
        # It works for -inf as well
        nearfield_loss =  self.gamma*10*np.log10(np.pi)
        if torch.sum(L<nearfield_loss):
            #L[L<0] = 10  
            # try to smooth singularities
            i = (L<nearfield_loss).nonzero()
            L[i] = nearfield_loss
        
        fobs = self.P0 - L.unsqueeze(1)
        
        #fobs = f.normalize_maxmin(fobs,self.data_max,self.data_min)
        
        return fobs

    def get_theta(self):
        """
        Get the theta parameter leanred by the PL model
        """
        return self.theta
    
    
# Build a augmented model class    
class Net_augmented(torch.nn.Module):
    def __init__(self,input_dim, layer_wid, nonlinearity,gamma=2,theta0=None,
                 data_max=None,data_min=None):
        super().__init__()
        #super(Net, self).__init__()

        # build the pathloss model
        self.model_PL = Polynomial3(gamma,theta0,data_max,data_min)

        # build the NN network
        self.model_NN = Net(input_dim, layer_wid, nonlinearity)  
    
        
    def forward(self,x):
    
        y = self.model_PL(x) + self.model_NN(x)
        # y = self.model_PL(x)
        # y = self.model_NN(x)

        
        return y
    def get_NN_param(self):
        """
        Returns a tensor of parameter of NN layers
        """
        return self.model_NN.get_param()
    
    def get_theta(self):
        """
        Get the theta parameter leanred by the PL model
        """
        return self.model_PL.get_theta()
    
# define main function, return the train loss and test loss
def main(train_x,train_y,test_x,test_y,layer_wid,nonlinearity,epochs,lr,data_max,data_min):
    """
    layer_wid: NN structure. 
    nonlinearity: activation function. 
    epochs: training epochs
    lr: learning rate of neural network
    """
    # prepare the train data and test data
    train_data = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(dataset=train_data, batch_size=400, shuffle=True)
    # test_data = TensorDataset(test_x, test_y)
    # test_dataloader = DataLoader(dataset=test_data, batch_size=400, shuffle=True)
    # get the input dimension
    input_dim = train_x.shape[1]
    
    # initialize theta in the hybrid model
    idx = train_y.detach().clone().argmax()
    theta0 = train_x.detach().clone()[idx,:] + torch.randn(2) # avoid singularities
    theta0 = theta0.tolist()
    
    # build the Augmented NN model
    gamma = 2  
    model_aug = Net_augmented(input_dim, layer_wid, nonlinearity,gamma,theta0,data_max,data_min)
    
    # define the loss funciton and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model_aug.parameters(), lr=lr)

    train_mse = torch.zeros(epochs)
    test_mse = torch.zeros(epochs)
    # training
    for epoch in range(epochs):
        train_loss = 0
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            
            y_predict = model_aug(x_batch)

            loss = loss_function(y_predict, y_batch)
            
            # NN model regularization
            beta = 1
            l2_reg = torch.tensor(0.)
            for param in model_aug.model_NN.parameters():
                l2_reg += torch.linalg.norm(param)**2            
            loss += beta * l2_reg    
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss/len(train_dataloader)
        train_mse[epoch] = train_loss
        # test loss
        
        y_predict_test = model_aug(test_x)
        
        test_loss = loss_function(y_predict_test, test_y)
        test_mse[epoch] = test_loss.detach()

    return train_mse, test_mse, model_aug


