#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
library of functions

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

import numpy as np
import torch
import sys
sys.path.insert(1, 'BFNN')
import plotly.io as pio
pio.renderers.default='browser'
import copy
sys.path.insert(1, 'pykalman')
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def normalize(tensor_list):
    
  # manage inf values
  # -inf means no sig. Should they be removed?
  
  # set them lower than the non-inf minimum, but not too much to avoid disrupting the value range
  coef = 1.5
  tensor_list[tensor_list.isneginf()] = np.nan # removing the -inf
  tmp = tensor_list.topk(1,dim=0,largest=False) # finding the min omitting the NaN
  tmpmin = tmp.values.min()
  tensor_list[tensor_list.isnan()] = coef*tmpmin # set the NaN to a new minimum
  
  tensor_list[tensor_list.isinf()] = np.nan # removing the inf
  tmp = tensor_list.topk(1,dim=0,largest=True) # finding the max omitting the NaN
  tmpmax = tmp.values.max()
  tensor_list[tensor_list.isnan()] = coef*tmpmax # set the NaN to a new minimum

  mins = tensor_list.min(dim=0, keepdim=True)[0]
  maxs = tensor_list.max(dim=0, keepdim=True)[0]
  
  tensor_list = (tensor_list - mins) / (maxs - mins)
  # tensor_list = tensor_list.float()
  return tensor_list,maxs,mins

def normalize_maxmin(tensor_list,maxs,mins):
    """
    normalize wrt to external max and min
    """
    tensor_list = (tensor_list - mins) / (maxs - mins)
    # tensor_list = tensor_list.float()
    return tensor_list


def getSampleGrid(test_y_predict,test_x, radius = 20, Npoints = 80):
    #--- based on test data
    # sample the space around test_Jloc
    # radius: sample space area (m)
    # Npoints: sample points within the radius
    MM = test_y_predict.max(dim=0,keepdim=True)[0]
    Idx = test_y_predict.argmax(dim=0,keepdim=True)[0]
    test_Jloc = test_x.numpy()[Idx,:]
    
    x = np.linspace(test_Jloc[0]-radius,test_Jloc[0]+radius,Npoints*2)
    y = np.linspace(test_Jloc[1]-radius,test_Jloc[1]+radius,Npoints*2)
    xv, yv = np.meshgrid(x, y)
    xv = xv.ravel()
    yv = yv.ravel()
    sampGrid_mini = np.array([xv,yv]).T
    sampGrid_x = torch.from_numpy(np.array(sampGrid_mini)).float()
    return sampGrid_x, sampGrid_mini, MM

def grid_peak_estimation(sampGrid_y,sampGrid_mini,trueJloc):
    MM = sampGrid_y.max(dim=0,keepdim=True)[0]
    Idx = sampGrid_y.argmax(dim=0,keepdim=True)[0]
    Jloc = sampGrid_mini[Idx,:]
    err = Jloc - trueJloc
    return Jloc, err, MM

def jam_pos_estimation_grid(test_y_predict,test_x,model_NN,trueJloc):
    sampGrid_x, sampGrid_mini, _ = getSampleGrid(test_y_predict,test_x)
    sampGrid_y = model_NN(sampGrid_x).detach()
    Jloc, err, _ = grid_peak_estimation(sampGrid_y,sampGrid_mini,trueJloc)  
    return Jloc, err

def jam_pos_estimation_grad(test_y_predict,test_x,model_NN,trueJloc):
    model = copy.deepcopy(model_NN)
    # model = type(model_NN)() # get a new instance
    # model.load_state_dict(model_NN.state_dict()) # copy weights and stuff
    Idx = test_y_predict.argmax(dim=0,keepdim=True)[0]
    x0 = test_x[Idx,:]
    x0.requires_grad = True
    for param in model.parameters():
        param.requires_grad = False
    
    # gradient ascent loop
    lr = 1e-2
    
    # optim = torch.optim.SGD(x0, lr)
    
    #y0 = -y0 # ascent
    for i in range(100):
        # optim.zero_grad()
        y0 = model(x0)
        #compute gradient
        y0.backward()
        with torch.no_grad():
            x0 = x0 + lr * x0.grad  # + for the ascent
        # optim.step()
        x0.requires_grad = True
    
    Jloc = x0.detach().numpy().reshape(-1)
    err = Jloc-trueJloc
    return Jloc, err

def dist(X,xj):
    d = np.sqrt((X[:,0]-xj[0])**2+(X[:,1]-xj[1])**2)
    return d

def crb_Jloc_power_obs(X,xj,noise_var,gamma):
    """
    Computes CRB for Pn = Ptx -10*gamma*log10(d) + wn
    where Pn is n-th power observation, d distance between xj and power obs 
    location, wn n-th independent noise sample ~N(0,noise_var)

    Parameters
    ----------
    X : power observation locations [N measurements * 2 spatial dimensions]
    xj: true jammer location (parameter to be estimated)
    noise_var: measuerents gaussian noise variance
    gamma: path loss exponent

    Returns
    -------
    crb: Cramer Rao Bound [array of variances of parameter vector]

    """
    d = dist(X,xj)
    a = np.sum( (xj[0]-X[:,0])**2/d**4, axis=0)
    b = np.sum( (xj[1]-X[:,1])**2/d**4 , axis=0)
    c = np.sum( ((xj[0]-X[:,0])*(xj[1]-X[:,1]))/d**4 , axis=0)
    
    coef = (noise_var*np.log(10)**2)/(100*gamma**2)
    
    crb = np.zeros((2,2))
    
    # var(xj)
    crb[0,0] = coef * b/(a*b-c**2) 
    # var(yj)
    crb[1,1] = coef * a/(a*b-c**2) 
    # covar(xj,yj)
    crb[0,1] = coef * -c/(a*b-c**2) 
    crb[1,0] = crb[0,1]

    return crb
    

def mle(x,y,Ptx,sigma,gamma,trueJloc):
    """
    MLE for pathloss
    computed through a coarse grid search and refined through a scipy optimizater
    
    # log-likelihood formula:
    # l = -N/2*np.log(2*np.pi*sigma**2) + ((-1/(2*sigma**2)* tot  ))
    # use only maximizable part
    """
    Nss = 40 # search space size
    
    # define search space (could be enlarged to be safer)
    min_ss = x.min(dim=0) # search space
    min_ss_x = min_ss[0][0] 
    min_ss_y = min_ss[0][1]
    max_ss = x.max(dim=0) # search space
    max_ss_x = max_ss[0][0] 
    max_ss_y = max_ss[0][1]
    
    # search for the max
    xj_v = np.linspace(min_ss_x,max_ss_x,Nss)
    yj_v = np.linspace(min_ss_y,max_ss_y,Nss)
    
    l_mat = np.zeros([xj_v.shape[0],yj_v.shape[0]])

    lmax = -float("inf")
    for ii, xj in enumerate(xj_v):
        for jj, yj in enumerate(yj_v):
            
            l = -pl_logLH([xj,yj],x,y,gamma,Ptx,sigma)           
            l_mat[ii,jj] = l
            
            if l > lmax:
                lmax = l
                xj_mle = xj; yj_mle = yj
    xj_vec_coarse = [xj_mle, yj_mle]
    
    # res  = minimize(pl_logLH,xj_vec_coarse,args=(x,y,gamma,Ptx,sigma),
    #                    method='BFGS')
    # res  = minimize(pl_logLH,xj_vec_coarse,args=(x,y,gamma,Ptx,sigma),
    #                     method='Nelder-Mead',tol=1e-6)
    # xj_vec = res.x
    xj_vec = xj_vec_coarse
    
    err = xj_vec-trueJloc
    
    # import plotly.graph_objects as go
    # import plotly.io as pio
    # pio.renderers.default='browser'
    # #pio.renderers.default='svg'
    # # plot the train data 3D figure
    # fig = go.Figure(data=[go.Surface(x = xj_v,y=yj_v,z=l_mat)])
    # fig.update_layout(title_text='Train data', title_x=0.5)
    # fig.show()
       
    return xj_vec, err

def pl_logLH(xj,x,y,gamma,Ptx,sigma):
    """
    MINUS Path loss log likelihood function 
    logLH function is inverted before return to allow minimization
    (only maximizable part, a constant term has been neglected)
    """
    
    N = x.shape[0]
    tot = 0
    for i in range(N):
        # maximizable part of log-likelihood
        d = np.sqrt((xj[0]-x[i,0])**2 + (xj[1]-x[i,1])**2)
        # compute path loss
        if d<=np.pi:
        # manage log10(0)
            L = 9.942997453882677# Ptx-y[i] # when too close to a datapoint the power is the power of the datapoint
        else:
            L = 10*gamma*np.log10(d)  
            
            # f = 1575.42e6
            # gamma = 2
            # c = 299792458
            # L = 10*gamma*np.log10(4*np.pi*f*d/c) 

        tot += (y[i]-(Ptx-L))**2
    l = -1/(2*sigma**2) * tot 
    return -l



def compute_cdf(data):
    bins = 15
    # getting data of the histogram
    count, bins_count = np.histogram(data.ravel(), bins=bins)
      
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
      
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
    return cdf, pdf, bins_count  

def highest_values_based_selection(train_y,test_y,train_x,test_x,
                    constant_data_select_size,ttSplit):
    
    test_size = round(constant_data_select_size*(1/ttSplit-1))
    idx = train_y.argsort(dim=0,descending=True)
    train_y[:] = train_y[idx]            
    train_x[:,:] = train_x[idx,:]            
    idx = test_y.argsort(dim=0,descending=True)
    test_y[:] = test_y[idx]
    test_x[:,:] = test_x[idx,:]                     
    # trim
    train_y = train_y[:constant_data_select_size].detach().clone() 
    train_x = train_x[:constant_data_select_size,:].detach().clone()
    # keep the same training testing ratio        
    test_y = test_y[:test_size].detach().clone()
    test_x = test_x[:test_size,:].detach().clone()
        
    return train_y,test_y,train_x,test_x 



def estimate_param_rate(reg_weights_rate_type,param,param_old,fixed_weight_rate = 0.1):
    # can be modified with linear prediction method given the set of past parameters
    
    if reg_weights_rate_type == 'time_instants':
        param_rate = param-param_old
        DeltaT = 1 # time instnts elapsed (over whch rate is computed)(batches don't matter because the dataset is not related to a different jammer location)
    elif reg_weights_rate_type == 'fixed':
        param_rate = fixed_weight_rate
        DeltaT = 1
    else:
        print('unknown parameter rate estimation method')
    return param_rate, DeltaT
