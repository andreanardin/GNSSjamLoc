% *************************************************************************
% ************************ GNSSjamLoc v0.1 ******************************** 
% ************ localization of intentional interfence sources *************
% *************************************************************************

% This file is part of GNSSjamLoc. 
% 
% GNSSjamLoc is a localization tool for intentional 
% interfence sources (jammer), that are potentially disturbing
% ligitimate signal reception (e.g. GPS). Given a dataset of power
% field measurement associated with spatial coordinates and
% distributed over an observation area, GNSSjamLoc can locate the
% source of interference, even in complex propagation scenarios (e.g. urban).
% It is based on a path loss physics-based model augmented with 
% a data-driven component, i.e. a Neural Netrwork.
% Additional information can be found at https://doi.org/10.48550/arXiv.2212.08097
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% 
% Copyright (C) 2023 Andrea Nardin <andrea.nardin@polito.it>
% Navigation, Signal Analysis and Simulation (NavSAS) group,
% Politecnico di Torino 
% Copyright (C) 2023 Peng Wu, Tales Imbiriba, Pau Closas
% Signal Processing Imaging Reasoning and Learning (SPIRAL) Lab
% Northeastern University
% 
% If you use this software, please cite the relative paper as:
% A. Nardin, T. Imbiriba, P. Closas, "Jamming Source Localization Using Augmented Physics-Based Model",
% ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
% Rhodes, Greece, 2023, pp. -, doi: -.

% Jammed GNSS agents simulator
% Warning: obs_time>1 not compatible with GNSSjamLoc estimator
%
% General algorithm
% 1. Agents' states generation
%   - initial position: uniform distribution over a specified area
%   - agents motion: random position at each instant
% 2. Jammer power measurements
%   - jammer signal type: true power computation
%   - radio propagation model: free space, ray tracing, log-distance
%   - estimation method: true meas. + measurement noise
%


close all
clearvars
addpath matlab_functions
% rng(2342) % not close-to.edge Jloc
rng(3) % close to edge Jloc

%% Settings

dB_flag         = 1;        % Output measurements in dB
N               = 1e4;      % no. of agents
D               = 2;        % spatial coordinates (2 or 3)
obs_time        = 1;       % (s)
obs_meas_rate   = 1;        % (Hz)
obs_area        = 1e6;      % (m^2)
T = obs_time*obs_meas_rate; % discrete time length
moving_obs_area = 0; % data collection area moves with the jammer

% Jammer settings ---------------------------------------------------------
P_jam_tx        = 10;          % (dbW)
f_jam           = 1575.42e6;   % jammer central frequency (Hz)
meas_noise_var  = 0;%1e0;% 1e-11; % check where it's added (db or linear). measurement noise variance (dBW^2)
no_jam_time     = 0; % (s) initial seconds without jammers

% Jammer trajectory
jam_motion      = 'static';
% jam_motion      = 'linear';
% jam_motion      = 'parabola';
% jam_motion      = 'random walk';
static_jam_time = 0; % (s) initial seconds with static jammer

% Path loss model 
% pathloss_type = 'free_space';
% pathloss_type = 'path_loss_exp'; gamma = 2;
% pathloss_type = 'rain';
pathloss_type = 'ray_tracing';

% Rx settings
Gt = 0; % tx gain (dB)
Gr = 0; % rx gain (dB)

% Output Settings
% merge all the generated data in one matrix or divide them along a time
% dimension
data_time_sequence = 0;

% Agents' position disruption
loadMeasurements = 0;     % avoid recomputing raytracing to disrupt data
N0 = -204.4; % dBW-Hz
cn0_nom = 45; % dBHz (45 for pathloss, 40 for urban)
rx_sens = 21; % dBHz
posEstVar = 100; % agents' position estimation error variance (m^2)

%--- future additional settings
% jammmer_type = 'CW'; % 'CW', 'chirp', ...
% agents motion (pedestrian random walk, etc)
% P_jam_est_method


%% Jammer position and signal
init_loc = sqrt(obs_area)*rand([1,D]);
time_vector = 0:1/obs_meas_rate:obs_time-1;
velx = 5; vely=5;

% array of initially static positions, to be attached at the beginning
jammer_loc_pre = [];
if static_jam_time > 0
    time_vector = time_vector(1:end-static_jam_time);
    jammer_loc_pre = repmat(init_loc,static_jam_time,1); 
end

switch lower(jam_motion)
    case 'linear'
        jammer_loc = init_loc + [velx*time_vector' vely*time_vector'];
    case 'parabola'
        jammer_loc = init_loc + [velx*time_vector' vely*time_vector.^2'];
    case 'random walk'
        jammer_loc = random_walk2D(init_loc,length(time_vector),[velx vely]/obs_meas_rate);
    otherwise
        % static jammer
        jammer_loc = repmat(init_loc,length(time_vector),1);        
end

jammer_loc = [jammer_loc_pre; jammer_loc];


%% Agents position and motion generation

% random position at each time instant
if moving_obs_area
    % agent position uniformly distributed around the jammer
    tmp = repmat(jammer_loc,1,1,N);
    tmp = permute(tmp,[3 2 1]);
    X = sqrt(obs_area)*(rand([N,D,T])-0.5) + tmp;
else
    X = sqrt(obs_area)*rand([N,D,T]);
end


%% Jammer power estimation

if ~loadMeasurements
switch lower(pathloss_type)
    case 'free_space'
        L = my_fspl(jammer_loc,X,f_jam);
    case 'ray_tracing'
        L = ray_tracing_pl(jammer_loc,X,f_jam,P_jam_tx);  
    case 'path_loss_exp'
        L = my_fspl(jammer_loc,X,f_jam,gamma);
    otherwise
        error 'unknown pathloss type'
        
end

% True power (dB)
P_jam_rx = P_jam_tx+Gt+Gr - L;

% Initial time without jamming
if no_jam_time > 0
    P_jam_rx(:,1:no_jam_time) = min(min(-L));
    %P_jam_rx(:,1:no_jam_time) = min(min(-L(~isinf(L)))); No inf with raytracing
end

%--- Jammer power estimation
% true power + noise
% Y = db2pow(P_jam_rx) + sqrt(meas_noise_var)*randn(N,T);
% Y = abs(Y); % gaussian dist. is gone now (folded dist.)
Y = P_jam_rx + sqrt(meas_noise_var)*randn(N,T); % dB
if ~dB_flag
    Y = db2pow(Y); 
end
save('data/tmpMeas.mat');
else
    % avoid recomputing raytracing to disrupt data
    load('data/tmpMeas.mat');
end


%% Build the dataset

% Position estimation error
X = X + sqrt(posEstVar)*randn(size(X));

% data preparation
if data_time_sequence
    XX = X; YY = Y;
else
    % treat each datapoint equally, whether it is a diffrent agent or the same
    % agent at a different time
    XX = zeros(N*T,D);
    YY = zeros(N*T,1);
    for ii = 1:T
        XX((ii-1)*N+1:ii*N,:) = X(:,:,ii);
        YY((ii-1)*N+1:ii*N) = Y(:,ii);
    end
end
save('X.mat','XX'), save('Y.mat','YY')


if data_time_sequence
    Jloc = jammer_loc;
else
    Jloc = jammer_loc(1,:);
end
save('trueJamLoc.mat','Jloc')


%% Plots

defaultAxesFontsize = 16;
defaultLegendFontsize = 16;
defaultLegendInterpreter = 'latex';
defaultLinelinewidth = 2;
defaultStemlinewidth = 2;
defaultAxesTickLabelInterpreter = 'latex';
defaultTextInterpreter = 'latex';
set(0,'defaultAxesFontsize',defaultAxesFontsize,'defaultLegendFontsize',defaultLegendFontsize,...
    'defaultLegendInterpreter',defaultLegendInterpreter,'defaultLinelinewidth',defaultLinelinewidth,...
    'defaultAxesTickLabelInterpreter',defaultAxesTickLabelInterpreter);
set(0,'defaultTextInterpreter',defaultTextInterpreter);
set(0,'defaultFigurePaperPositionMode','auto')
set(0,'defaultStemlinewidth',defaultStemlinewidth)


h = zeros(1,3); % handles
fig = figure(1);

step = ceil(obs_time/20);
for pt = 1:step:obs_time % plot time - time instant to plot
    hold off
    x=X(:,1,pt);y=X(:,2,pt);z=Y(:,pt);

%     if disrupt_pos
%         x=X(not(denied_rxs(:,pt)),1,pt);y=X(not(denied_rxs(:,pt)),2,pt);z=Y(not(denied_rxs(:,pt)),pt);
%     end
    
    % surf plot
    xv = linspace(min(x), max(x), 100);
    yv = linspace(min(y), max(y), 100);
    [XX,YY] = meshgrid(xv, yv);
    ZZ = griddata(x,y,z,XX,YY);
    h(1) = surf(XX, YY, ZZ);
    grid on, hold on
    % set(gca, 'ZLim',[0 100])
    shading interp
    h(2) = plot3(jammer_loc(pt,1)*[1 1],jammer_loc(pt,2)*[1 1],max(z)+[-abs(max(z))*1.8 +abs(max(z))*0.8],'-ro','MarkerFaceColor','r','markerSize',9);
    h(3) = plot3(x,y,z,'k.','markersize',8);
    legend(h(1:3),'estimated power','jammer location','datapoints','location','best')
    xlabel 'x coord.'
    ylabel 'y coord.'
    if dB_flag
        zlabel 'Est. power (dBW)'
    else
        zlabel 'Est. power (W)'
    end
    title(['time = ' num2str(pt)])
    mm = min(min(min(X))); MM = max(max(max(X)));
    xlim([mm MM]); ylim([mm MM])
    drawnow
    
    % --- export gif
    frame = getframe(fig);
    im = frame2im(frame);
    filename = 'figs/animatedField.gif'; % Specify the output file name
    [A,map] = rgb2ind(im,256);
    if pt == 1
        imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',1);
    else
        imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',1);
    end
    %---------------------------
    
    pause(1)
end

% plot true power field
figure
if dB_flag
    trueY = P_jam_rx;
else
    trueY = db2pow(P_jam_rx);
end
x=X(:,1,pt);y=X(:,2,pt);z=trueY(:,pt);


xv = linspace(min(x), max(x), 100);
yv = linspace(min(y), max(y), 100);
[XX,YY] = meshgrid(xv, yv);
ZZ = griddata(x,y,z,XX,YY);
h(1) = surf(XX, YY, ZZ);
grid on, hold on
% set(gca, 'ZLim',[0 100])
shading interp
h(2) = plot3(jammer_loc(pt,1)*[1 1],jammer_loc(pt,2)*[1 1],max(z)+[-abs(max(z))*1.8 +abs(max(z))*0.8],'-ro','MarkerFaceColor','r','markerSize',9);
h(3) = plot3(x,y,z,'k.','markersize',5,'LineWidth',1);
legend(h(1:3),'true power','jammer location','datapoints','location','best')
xlabel '$\theta_1$'
ylabel '$\theta_2$'
if dB_flag
    zlabel 'True power (dBW)'
else
    zlabel 'True power (W)'
end
title(['True received power - time = ' num2str(pt)])
xlim([mm MM]); ylim([mm MM])
savefig(gcf,'figs/jammer_field_true.fig')
saveas(gcf,'figs/jammer_field_true.png')

% plot jammer motion
figure
plot(jammer_loc(:,1),jammer_loc(:,2),'-o')
xlabel 'North (m)'
ylabel 'East (m)'
title 'Jammer trajectory'


