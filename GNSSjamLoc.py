"""
 *************************************************************************
 **************************** GNSSjamLoc v0.1 **************************** 
 ************ localization of intentional interfence sources *************
 *************************************************************************

This file is part of GNSSjamLoc. 

GNSSjamLoc is a localization tool for intentional 
interfence sources (jammers), that are potentially disturbing
ligitimate signal reception (e.g. GPS). Given a dataset of power
field measurement associated with spatial coordinates and
distributed over an observation area, GNSSjamLoc can locate the
source of interference, even in complex propagation scenarios (e.g. urban).
It is based on a path loss physics-based model augmented with 
a data-driven component, i.e. a Neural Netrwork.
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

If you use this software, please cite the relative paper as:
A. Nardin, T. Imbiriba, P. Closas, "Jamming Source Localization Using Augmented Physics-Based Model",
ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
Rhodes, Greece, 2023, pp. -, doi: -.

Augmented Physics-based model for Jammer localization
    
    Monte Carlo simulations might run over two sources of randomness: 
        1 the NN intrinsic randmoness of pytorch and data scrambling
        2 the measurement noise (only if generated here and not in Matlab)
    Note: 
        - normalization is performed at each Monte Carlo run, so it does scale each realization differently
        - trimming the data allows to select the most powerful measurements. 
            It is assumed to be able to do so after data collection, thus potentially 
            having access to better measurments with respect to collecting data over a smaller area
Author: Andrea Nardin <andrea.nardin@polito.it>

"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io as io
import numpy as np
import torch
import model_nn
import time
import matplotlib.pyplot as plt
#import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'
import functions as f

# import random
# random.seed(42)
# torch.manual_seed(42)


tic = time.time()

#%% SETTINGS

# flags
aggregate_over_time = 0 # aggregate output data over time (other than over Monte Carlo)
grad_estimation = 0 # ultimately estimate theta through maximization of the NN model

noiseless_data_in = 1  # add noise here within Monte Carlo runs, if data comes from MATLAB w/out noise
computeMle = 0
# Data_selection criterion
data_selection = 1 #[0: No selection; 1: based on power level;]
scramble_data = 0

# Parameters
ttSplit = 2/3 # train and test split ratio
Nmc =20 # no. of Monte Carlo simulations runs
fixed_data_select_size = 15 # you get fixed_data_select_size training datapoints at each time
min_data_size = 10
max_data_size = 150

# NN hyper-parameters

layer_wid =[200,100,1]
nonlinearity = 'tanh'
epochs = 50
epochs_init = 200
lr = 0.4 #0.003

# Data observations characteristics
# if meas_noise_var_iter is a vector, evaluate a number of SNR, collect variances only
meas_noise_var_iter = np.array([10, 10/np.sqrt(10),  1, 0.1, 0.01]) #, 0.001])  # noise variance affecting power measurements
gamma = 2  # path loss exponent
Ptx = 10  # assumed known by MLE
time_step = 1


#%% --- LOAD DATA
# load true position
fname_Jloc = 'trueJamLoc.mat'
data = io.loadmat(fname_Jloc)
trueJloc = data['Jloc']
# load matlab data
fname_x = 'X.mat'
data = io.loadmat(fname_x)
X = data['XX']
fname_y = 'Y.mat'
data = io.loadmat(fname_y)
Y = data['YY']
q = round(X.shape[0]*ttSplit)

# prepare training and test datasets
# is data a time sequence?
if len(X.shape)>2:
    T = X.shape[-1]
else:
    T = 1
    
#%% scramble data (every time instant has the same scrambled order)
if scramble_data:
    Idxs = list(range(X.shape[0]))
    np.random.shuffle(Idxs)
    # Xsh = [X[i] for i in Idx]
    # Ysh = [Y[i] for i in Idx]
    Xsh = np.empty_like(X)
    for i in range(X.shape[0]):
        idx = Idxs[i]
        Xsh[i,:,:] = X[idx,:,:]
    #X = Xsh; Y = Ysh;
        
#%% divide into train and test
Xtrain = X[:q]; Ytrain = Y[:q]
Xtest  = X[q:]; Ytest = Y[q:]
# add a dimension to keep using t index even when T=1
if T==1:
    Xtrain = np.expand_dims(Xtrain,2)
    Xtest = np.expand_dims(Xtest,2)
    # Ytrain = np.expand_dims(Ytrain,1)  #no need
    # Ytest = np.expand_dims(Ytest,1)

train_x_original = torch.from_numpy(np.array(Xtrain)).float()
train_y_original = torch.from_numpy(np.array(Ytrain)).float()
test_x_original = torch.from_numpy(np.array(Xtest)).float()
test_y_original = torch.from_numpy(np.array(Ytest)).float()

# count obstructed signal
#noSig_test = train_y_original[train_y_original.isneginf()]
#noSig_train = tensor_list[tensor_list.isneginf()]

#%% --- TRAIN & TEST

# initialize arays
Jloc_vec = np.zeros((T,X.shape[1],Nmc)); err_vec = np.zeros((T,X.shape[1],Nmc)); 
rmse_vec = np.zeros(T)
var_x_crb = np.zeros(T)
var_y_crb = np.zeros(T)
Jloc_mle_vec = np.zeros((T,X.shape[1],Nmc)); err_mle_vec = np.zeros((T,X.shape[1],Nmc)); 
data_select_sizes_train = np.empty([Nmc,T])
data_select_sizes_test = np.empty([Nmc,T])
fixed_data_select_flags = np.zeros([Nmc,T]) # non zeros when fixed size data selection is used

rmse_vec2 = np.zeros(T)

#%% SNR loop
# init SNR iterations stuff
N_snr = meas_noise_var_iter.size
mle_varx_snr    = np.zeros(N_snr)
mle_vary_snr    = np.zeros(N_snr)
crb_varx_snr    = np.zeros(N_snr) 
crb_vary_snr    = np.zeros(N_snr) 
nn_varx_snr     = np.zeros(N_snr) 
nn_vary_snr     = np.zeros(N_snr) 
snr_values_snr  = np.zeros(N_snr)

err_mle_rmsx_snr    = np.zeros(N_snr)
err_mle_rmsy_snr    = np.zeros(N_snr)
err_nn_rmsx_snr     = np.zeros(N_snr)
err_nn_rmsy_snr     = np.zeros(N_snr)
        
mle_meanx_snr   = np.zeros(N_snr)
mle_meany_snr   = np.zeros(N_snr)
nn_meanx_snr    = np.zeros(N_snr)
nn_meany_snr    = np.zeros(N_snr)
true_Jlocx_snr  = np.zeros(N_snr)
true_Jlocy_snr  = np.zeros(N_snr)
           
# The "signal" is the observation function. i.e. the noiseless observations
# compute the average of their square, to compare it to the noise variance
# 1 "signal" power value for each t
tmp = Ytrain
tmp[np.isneginf(tmp)] = np.nan # removing the -inf
logicals = np.logical_not(np.isnan(tmp))
Nmeas = logicals.sum()
S = np.nansum(tmp**2,0) / Nmeas

for ii_snr in range(N_snr):
    meas_noise_var = meas_noise_var_iter[ii_snr]
        
    #%% CRB computation
    for t in range(T):
        crb = f.crb_Jloc_power_obs(Xtrain[:,:,[t]],trueJloc[t,:],meas_noise_var,gamma)
        var_x_crb[t] = crb[0,0]
        var_y_crb[t] = crb[1,1]
    
    
    #%% -- Monte Carlo runs
    # kf initial param learned just una tantum
    learned_kf = False; kf_smoothed = None
    iter_list = list(np.arange(0,Nmc))

    for n in iter_list:
        if n == 0:
            fixed_data_select_flags = np.zeros([Nmc,T]) # non zeros when fixed size data selection is used
    
        print('Monte Carlo run: ',n)
            
        #%% Add noise to power measurments
        if noiseless_data_in:
            train_y_original_noisy = train_y_original.detach().clone() + np.sqrt(meas_noise_var)* torch.randn(train_y_original.shape)
            test_y_original_noisy = test_y_original.detach().clone() + np.sqrt(meas_noise_var)* torch.randn(test_y_original.shape)
            # repeat the same for x data if you want noisy position estimations
            train_x_original_noisy = train_x_original.detach().clone()
            test_x_original_noisy = test_x_original.detach().clone()
        else:
            train_y_original_noisy = train_y_original.detach().clone()
            test_y_original_noisy = test_y_original.detach().clone()
            train_x_original_noisy = train_x_original.detach().clone()
            test_x_original_noisy = test_x_original.detach().clone()
        
        #%% normalize the noisy data
        train_y_original_noisy_nonorm = train_y_original_noisy.detach().clone()
       
        data_max = None; data_min = None
        # normalize test data with the same function of train data
        # train_y_original_noisy,data_max,data_min = f.normalize(train_y_original_noisy)
        # test_y_original_noisy = f.normalize_maxmin(test_y_original_noisy,data_max,data_min)
                
        # save cost and regularization values   
        loss_part1 = list()
        loss_part2 = list()    
        loss_part3 = list()    
    
    
        # just to not be bothered by the interpreter's warnings
        filt_state_cov = None; filt_state_mean = None; kf = None
        param_sequence = torch.tensor(())
        
        #%% - Training time instants
        for t in range(T):
            
            # generate tensors for current time instant
            # slice datasets
            train_y_nonorm = train_y_original_noisy_nonorm[:,t].detach().clone()
            train_y = train_y_original_noisy[:,t].detach().clone()
            test_y = test_y_original_noisy[:,t].detach().clone()
            train_x = train_x_original_noisy[:,:,t].detach().clone()
            test_x = test_x_original_noisy[:,:,t].detach().clone()
            
            train_x_whole = train_x.detach().clone() # no data selection for MLE
            
            #%% Data preselection - Agent position disruption
            
            
            #%% Data down-selection 
            if data_selection==1 or t==0:
                # apply data select to not normalized (just for MLE)
                train_y_nonorm_select = f.highest_values_based_selection(
                    train_y_nonorm,test_y.detach().clone(),
                    train_x.detach().clone(),test_x.detach().clone(),
                    fixed_data_select_size,ttSplit
                    )[0]                
                
                train_y,test_y,train_x,test_x = f.highest_values_based_selection(
                    train_y,test_y,train_x,test_x,
                    fixed_data_select_size,ttSplit
                    )
                fixed_data_select_flags[n,t] = 1 # 1 to highlight the presence of fixed size data selection
                

    
            data_select_sizes_train[n,t] = train_y.shape[0]
            data_select_sizes_test[n,t] = test_y.shape[0]
              
            #%% Learning       
            # learning
            train_loss, test_loss, model_NN = model_nn.main(train_x,
                                                                train_y.reshape(-1,1),
                                                                test_x,
                                                                test_y.reshape(-1,1),
                                                                layer_wid,nonlinearity,epochs_init,lr,
                                                                data_max,data_min)
            param_rate = []; DeltaT = [] # estimated values used from t=3
           
                
                       
            if grad_estimation:
                # test predict (USE TRAIN TO BE FAIR)
                test_y_predict = model_NN(train_x).detach()
                # J pos estimation         
                Jloc,err = f.jam_pos_estimation_grad(test_y_predict,train_x,model_NN,trueJloc[t,:])
                # Jloc, err = f.jam_pos_estimation_grid(test_y_predict,train_x,model_NN,trueJloc[t,:])
                #rmse_vec2[t] = np.sqrt(np.mean(err**2))
            else:            
                Jloc = model_NN.get_theta().detach().clone().numpy()
                err = Jloc-trueJloc[t,:]
            
            
            
    
            Jloc_vec[t,:,n] = Jloc; err_vec[t,:,n] = err 
            rmse_vec[t] = np.sqrt(np.mean(err**2))
            print('Processing data collection epoch: ',t)
                            

            
            #%% MLE computation
            if computeMle:
                # data selected (specialized data)
                x_mle = train_x.detach().clone()
                y_mle = train_y_nonorm_select
                
                # # whole dataset (not recommended, very slow)
                # x_mle = train_x_whole.detach().clone()
                # y_mle = train_y_nonorm.detach().clone()
                
                Jloc_mle, err = f.mle(x_mle,y_mle,Ptx,meas_noise_var,gamma,
                                      trueJloc[t,:])
                Jloc_mle_vec[t,:,n] = Jloc_mle
                err_mle_vec[t,:,n] = err
                
                    
        # out: train_loss, test_loss, loss_part1, loss_part2, Jloc_vec, rmse_vec, rmse_vec2
            
        toc = time.time()
        print("%.2f seconds elapsed." % (toc-tic))
        
        
        #%% collect Monte Carlo   
        if n == 0:
            train_loss_tot = train_loss.detach().clone(); test_loss_tot = test_loss.detach().clone()
            rmse_vec_tot = rmse_vec.copy(); rmse_vec2_tot = rmse_vec2.copy()
            loss_part1_tot = np.array(loss_part1); loss_part2_tot = np.array(loss_part2); loss_part3_tot = np.array(loss_part3)
            param_sequence_tot = param_sequence.detach().clone()
    
        else:
            train_loss_tot += train_loss; test_loss_tot += test_loss
            rmse_vec_tot += rmse_vec; rmse_vec2_tot += rmse_vec2
            if data_selection == 1:
                loss_part1_tot += np.array(loss_part1); loss_part2_tot += np.array(loss_part2); loss_part3_tot += np.array(loss_part3) 
            param_sequence_tot += param_sequence.detach().clone()
            
            
    Jloc_vec_avg = np.mean(Jloc_vec,axis=2)
    Jloc_vec_var = np.var(Jloc_vec,axis=2)
    err_vec_avg = np.mean(err_vec,axis=2)
    err_vec_rms = np.sqrt(np.mean(err_vec**2,axis=2))
    Jloc_mle_vec_avg = np.mean(Jloc_mle_vec,axis=2)
    Jloc_mle_vec_var = np.var(Jloc_mle_vec,axis=2)
    err_mle_vec_avg = np.mean(err_mle_vec,axis=2)
    err_mle_vec_rms = np.sqrt(np.mean(err_mle_vec**2,axis=2))
    param_seq_avg = param_sequence_tot/Nmc
    
    if aggregate_over_time:
        Jloc_vec_avg = np.mean(Jloc_vec_avg,axis=0,keepdims=True)
        Jloc_vec_var = np.var(Jloc_vec_var,axis=0,keepdims=True)
        err_vec_avg = np.mean(err_vec_avg,axis=0,keepdims=True)
        err_vec_rms = np.sqrt(np.mean(err_vec_rms**2,axis=0,keepdims=True))
        Jloc_mle_vec_avg = np.mean(Jloc_mle_vec_avg,axis=0,keepdims=True)
        Jloc_mle_vec_var = np.var(Jloc_mle_vec_var,axis=0,keepdims=True)
        err_mle_vec_avg = np.mean(err_mle_vec_avg,axis=0,keepdims=True)
        err_mle_vec_rms = np.sqrt(np.mean(err_mle_vec_rms**2,axis=0,keepdims=True))

    
    err_MAE = np.mean(np.absolute(err_vec))
    
    # compute CDF and PDF
    cdf_nn, pdf_nn, bins_count = f.compute_cdf(np.abs(err_vec))
    if computeMle:
        cdf_mle, pdf_mle, _ = f.compute_cdf(np.abs(err_mle_vec))
    
    # average results
    train_loss = train_loss_tot/Nmc; test_loss = test_loss_tot/Nmc
    rmse_vec = rmse_vec_tot/Nmc; rmse_vec2 = rmse_vec2_tot/Nmc
    loss_part1 = loss_part1_tot/Nmc; loss_part2 = loss_part2_tot/Nmc; loss_part3 = loss_part3_tot/Nmc
    
    
    #%% Collect SNR
    # Assume T=1 !
    if T==1 or aggregate_over_time:
        # snr_values_snr[ii_snr]  = 10*np.log10(S/meas_noise_var)        
        snr_values_snr[ii_snr]  = 10*np.log10(Ptx/meas_noise_var)

        # variances (good for unbiased estimators)
        crb_varx_snr[ii_snr]    = np.mean(var_x_crb) # not the best doing the mean
        crb_vary_snr[ii_snr]    = np.mean(var_y_crb)
        mle_varx_snr[ii_snr]    = Jloc_mle_vec_var[:,0]
        mle_vary_snr[ii_snr]    = Jloc_mle_vec_var[:,1]
        nn_varx_snr[ii_snr]     = Jloc_vec_var[:,0] 
        nn_vary_snr[ii_snr]     = Jloc_vec_var[:,1] 
            
        # RMS wrt to true error
        err_mle_rmsx_snr[ii_snr]    = err_mle_vec_rms[:,0]
        err_mle_rmsy_snr[ii_snr]    = err_mle_vec_rms[:,1]
        err_nn_rmsx_snr[ii_snr]     = err_vec_rms[:,0] 
        err_nn_rmsy_snr[ii_snr]     = err_vec_rms[:,1]
        # err_nn_RMS_snr[ii_snr]     =  np.sqrt(1/2*(err_vec_rms[:,1]**2 + err_vec_rms[:,1]**2)) # equivalent to RMS over all dimensions

        
        # Means to evaluate unbiased estimators
        mle_meanx_snr[ii_snr]    = Jloc_mle_vec_avg[:,0]
        mle_meany_snr[ii_snr]    = Jloc_mle_vec_avg[:,1]
        nn_meanx_snr[ii_snr]     = Jloc_vec_avg[:,0] 
        nn_meany_snr[ii_snr]     = Jloc_vec_avg[:,1] 
        true_Jlocx_snr[ii_snr]   = np.mean(trueJloc[:,0])
        true_Jlocy_snr[ii_snr]   = np.mean(trueJloc[:,1])

        ii_snr = ii_snr+1   


# Export data to MATLAB
io.savemat('data/matlab/rmsx.mat', mdict={'rmsx': err_nn_rmsx_snr})
io.savemat('data/matlab/rmsy.mat', mdict={'rmsy': err_nn_rmsy_snr})
io.savemat('data/matlab/mle_rmsx.mat', mdict={'mle_rmsx': err_mle_rmsx_snr})
io.savemat('data/matlab/mle_rmsy.mat', mdict={'mle_rmsy': err_mle_rmsy_snr})
io.savemat('data/matlab/crbx.mat', mdict={'crbx': crb_varx_snr})
io.savemat('data/matlab/crby.mat', mdict={'crby': crb_vary_snr})

io.savemat('data/matlab/meanx.mat', mdict={'meanx': nn_meanx_snr})
io.savemat('data/matlab/meany.mat', mdict={'meany': nn_meany_snr})
io.savemat('data/matlab/mle_meanx.mat', mdict={'mle_meanx': mle_meanx_snr})
io.savemat('data/matlab/mle_meany.mat', mdict={'mle_meany': mle_meany_snr})
io.savemat('data/matlab/truex.mat', mdict={'truex': true_Jlocx_snr})
io.savemat('data/matlab/truey.mat', mdict={'truey': true_Jlocy_snr})

io.savemat('data/matlab/snr.mat', mdict={'snr': snr_values_snr})


#%% --- PLOTS
#%% learning curve
plt.close('all')
plt.plot(train_loss,label='train')
plt.plot(test_loss,label='test')
plt.xlabel('epoch')
plt.legend()
plt.title('Learning curve')
plt.savefig('figs/learning_curve.png')

#%% loss function and regularization
plt.figure()
plt.plot(range(epochs_init,epochs_init+len(loss_part1)),loss_part1,label='loss (no reg)')
if loss_part2.shape[0]!=0:
    plt.plot(range(epochs_init,epochs_init+len(loss_part1)),loss_part2,label='reg past weights')
if loss_part3.shape[0]!=0:
    tmp = len(loss_part1)-len(loss_part3)
    plt.plot(range(epochs_init+tmp,epochs_init+len(loss_part1)),loss_part3,label='reg est. weight rates')

plt.legend()
plt.savefig('figs/loss_and_reg.png')

#%% J location estimation
plt.figure()
plt.plot(Jloc_vec_avg[:,0], Jloc_vec_avg[:,1],'-+',label='NN')
plt.plot(trueJloc[:,0], trueJloc[:,1],'-o',label='true')
if computeMle:
    plt.plot(Jloc_mle_vec_avg[:,0], Jloc_mle_vec_avg[:,1],'-o',label='MLE')

plt.plot(Jloc_vec_avg[0,0], Jloc_vec_avg[0,1],'c*',label='start')
plt.legend()
plt.title('Average Jammer position - MAE = %.2f' %err_MAE)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig('figs/jammer_positions.png')

# RMSE
# plt.figure()
# #plt.plot(rmse_vec2,'-o',label = 'sample grid')
# plt.plot(rmse_vec,'-o',label = 'gradient ascent')
# plt.legend()
# plt.title('Jammer position estimation RMSE')
# plt.xlabel('Data collection epoch (s)')
# plt.ylabel('RMSE (m)')
# plt.savefig('figs/jloc_rmse.png')


#%% Variance and CRB
fig, axs = plt.subplots(2)
fig.suptitle('Jammer position estimation std')
axs[0].plot(np.sqrt(Jloc_vec_var[:,0]),'-o',label = 'NN std')
axs[0].plot(np.sqrt(var_x_crb),'--',label = 'CRB std')
if computeMle:
    axs[0].plot(np.sqrt(Jloc_mle_vec_var[:,0]),'-o',label = 'MLE std')

axs[0].legend()
#axs[0].set_xlabel('Data collection epoch (s)')
axs[0].set_ylabel('x std (m)')

axs[1].plot(np.sqrt(Jloc_vec_var[:,1]),'-o',label = 'NN std')
axs[1].plot(np.sqrt(var_y_crb),'--',label = 'CRB std')
if computeMle:
    axs[1].plot(np.sqrt(Jloc_mle_vec_var[:,1]),'-o',label = 'MLE std')

axs[1].legend()
axs[1].set_xlabel('Data collection epoch (s)')
axs[1].set_ylabel('y std (m)')
plt.savefig('figs/jloc_var.png')

#%% RMSE and CRB
fig, axs = plt.subplots(2)
fig.suptitle('Jammer position estimation RMSE (true error)')
axs[0].plot(err_vec_rms[:,0],'-o',label = 'NN rmse')
axs[0].plot(np.sqrt(var_x_crb),'--',label = 'CRB std')
if computeMle:
    axs[0].plot(err_mle_vec_rms[:,0],'-o',label = 'MLE rmse')

axs[0].legend()
#axs[0].set_xlabel('Data collection epoch (s)')
axs[0].set_ylabel('x error (m)')

axs[1].plot(err_vec_rms[:,1],'-o',label = 'NN rmse')
axs[1].plot(np.sqrt(var_y_crb),'--',label = 'CRB std')
if computeMle:
    axs[1].plot(err_mle_vec_rms[:,1],'-o',label = 'MLE rmse')

axs[1].legend()
axs[1].set_xlabel('Data collection epoch (s)')
axs[1].set_ylabel('y error (m)')
plt.savefig('figs/err_var.png')




#%% plot CDF of error
plt.figure()
plt.plot(bins_count[1:], cdf_nn, label="NN")
if computeMle:
    plt.plot(bins_count[1:], cdf_mle, label="MLE")
plt.legend()
plt.title('CDF of pos. est. error (all errors)')
plt.xlabel('abs. error (m)')
#plt.ylabel('y (m)')
plt.savefig('figs/cdf.png')

# plot PDF of error
plt.figure()
plt.plot(bins_count[1:], pdf_nn, label="NN")
if computeMle:
    plt.plot(bins_count[1:], pdf_mle, label="MLE")
plt.legend()
plt.title('PDF of pos. est. error (all errors)')
plt.xlabel('abs. error (m)')
#plt.ylabel('y (m)')
plt.savefig('figs/pdf.png')

#%% Data selection size
if data_selection:
    fig, axs = plt.subplots(2)
    fig.suptitle('Data selection size (MC runs superimposed) vs time')
    axs[0].plot(data_select_sizes_train.T,'-o')
    # highlight fixed data size
    tmp = data_select_sizes_train.copy()
    tmp[fixed_data_select_flags<1] = np.nan
    axs[0].plot(tmp.T,'*',color = 'yellow')
    axs[0].legend()
    axs[0].set_xlabel('Data collection epoch (s)')
    axs[0].set_ylabel('Train Dataset size')
        
    axs[1].plot(data_select_sizes_test.T,'-o')
    # highlight fixed data size
    tmp = data_select_sizes_test.copy()
    tmp[fixed_data_select_flags<1] = np.nan
    axs[1].plot(tmp.T,'*',color = 'yellow')
    axs[1].legend()
    axs[1].set_xlabel('Data collection epoch (s)')
    axs[1].set_ylabel('Test Dataset size')
    plt.savefig('figs/data_selection_size_evolution.png')
   
#%% plot weights parameters
# random selection of 10 parameters 
# pointless monitor parameters change if T=1
if T>1 and not(aggregate_over_time):
    A = np.random.uniform(0,param_seq_avg.shape[0],20)   
    fig, axs = plt.subplots(2)
    fig.suptitle('Parameter dynamics (random selection)')
    axs[0].plot(param_seq_avg[A,:].transpose(0,1))
    axs[0].set_xlabel('time instant')
    axs[0].set_ylabel('parameter')
    axs[1].plot(param_seq_avg[A,:].diff(dim=1).transpose(0,1))
    axs[1].set_xlabel('time instant')
    axs[1].set_ylabel('parameter rate')
    plt.savefig('figs/param_time_all.png')
    
    fig, axs = plt.subplots(2)
    fig.suptitle('Parameter dynamics (average over Monte Carlo, statistcs among parameters)')
    
    # axs[0].plot(param_seq_avg.mean(0))
    flierprops = dict(marker='.', markerfacecolor='silver', markeredgecolor='gray', markersize=4, linestyle='none')
    medianprops = dict(linestyle='-', linewidth=5, color='orange')
    axs[0].boxplot(param_seq_avg.numpy(), zorder=10, notch=True, flierprops=flierprops, medianprops=medianprops)
    axs[0].set_xlabel('time instant')
    axs[0].set_ylabel('average parameter')
    
    # axs[1].plot(param_seq_avg.mean(0).diff())
    axs[1].boxplot(param_seq_avg.diff().numpy(), zorder=10, notch=True, flierprops=flierprops, medianprops=medianprops)
    axs[1].set_xlabel('time instant')
    axs[1].set_ylabel('average parameter rate')
    plt.savefig('figs/param_time_aggregate.png')

if N_snr>1:
    #%% Variance and CRB
    fig, axs = plt.subplots(2)
    fig.suptitle('Estimation std vs SNR')
    axs[0].plot(snr_values_snr,np.sqrt(nn_varx_snr),'-o',label = 'NN std')
    axs[0].plot(snr_values_snr,np.sqrt(crb_varx_snr),'--',label = 'CRB std')
    if computeMle:
        axs[0].plot(snr_values_snr,np.sqrt(mle_varx_snr),'-o',label = 'MLE std')
    axs[0].legend()
    #axs[0].set_xlabel('Data collection epoch (s)')
    axs[0].set_ylabel('x std (m)')

    axs[1].plot(snr_values_snr,np.sqrt(nn_vary_snr),'-o',label = 'NN std')
    axs[1].plot(snr_values_snr,np.sqrt(crb_vary_snr),'--',label = 'CRB std')
    if computeMle:
        axs[1].plot(snr_values_snr,np.sqrt(mle_vary_snr),'-o',label = 'MLE std')
    axs[1].legend()
    axs[1].set_xlabel('INR (dB)')
    axs[1].set_ylabel('y std (m)')
    plt.savefig('figs/jloc_var_vs_SNR.png')
    
    # err RMS
    fig, axs = plt.subplots(2)
    fig.suptitle('Estimation RMS (true error) vs SNR')
    axs[0].plot(snr_values_snr,err_nn_rmsx_snr,'-o',label = 'NN std')
    axs[0].plot(snr_values_snr,np.sqrt(crb_varx_snr),'--',label = 'CRB std')
    if computeMle:
        axs[0].plot(snr_values_snr,err_mle_rmsx_snr,'-o',label = 'MLE std')  
    axs[0].legend()
    #axs[0].set_xlabel('Data collection epoch (s)')
    axs[0].set_ylabel('x rms (m)')

    axs[1].plot(snr_values_snr,err_nn_rmsy_snr,'-o',label = 'NN std')
    axs[1].plot(snr_values_snr,np.sqrt(crb_vary_snr),'--',label = 'CRB std')
    if computeMle:
        axs[1].plot(snr_values_snr,err_mle_rmsy_snr,'-o',label = 'MLE std')
    axs[1].legend()
    axs[1].set_xlabel('INR (dB)')
    axs[1].set_ylabel('y rms (m)')
    plt.savefig('figs/jloc_RMS_vs_SNR.png')
    
    
    # mean 
    fig, axs = plt.subplots(2)
    fig.suptitle('Estimation mean vs SNR')
    axs[0].plot(snr_values_snr,nn_meanx_snr,'-o',label = 'NN mean')
    axs[0].plot(snr_values_snr,true_Jlocx_snr,'--',label = 'true value')
    if computeMle:
        axs[0].plot(snr_values_snr,mle_meanx_snr,'-o',label = 'MLE mean')  
    axs[0].legend()
    #axs[0].set_xlabel('Data collection epoch (s)')
    axs[0].set_ylabel('x mean (m)')

    axs[1].plot(snr_values_snr,nn_meany_snr,'-o',label = 'NN mean')
    axs[1].plot(snr_values_snr,true_Jlocy_snr,'--',label = 'true value')
    if computeMle:
        axs[1].plot(snr_values_snr,mle_meany_snr,'-o',label = 'MLE mean')
    axs[1].legend()
    axs[1].set_xlabel('INR (dB)')
    axs[1].set_ylabel('y mean (m)')
    plt.savefig('figs/jloc_Mean_vs_SNR.png')
    
    