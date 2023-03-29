function losses = ray_tracing_pl(jammer_loc,X,f_jam,P_tx)
%Computes the path loss using a ray tracing model in an urban scenario
%   The initial position of the tx is specified by init_lat_lon



%viewer = siteviewer("Buildings","chicago.osm");
% Chicago (to center the jammer in an urban environment)
init_lat_lon = [41.8800 -87.6295]; % chicago downtown
% init_lat_lon = [41.8800+km2deg(1e3*1e-3), -87.6295]; % chicago ~half km outsie of downtown

viewer = siteviewer("Buildings","chicago.osm");

% Matlab 2022?
pm = propagationModel("raytracing","Method","sbr",  "MaxNumReflections",4);



%pm = propagationModel("raytracing","MaxNumReflections",1);



% test if it's different
% pm = propagationModel("raytracing","MaxNumReflections",[0 1]);





[N,D,T] = size(X);

% convert tx & rx position in lat lon
% Note: deg2km operation assumes perfect Earth sphere with avg radius of 6371
if T==1
    deltas = [0,0];
else
    deltas = diff(jammer_loc);
    deltas = [0, 0 ; km2deg(deltas*1e-3)]; % convert the position increments in deg coordinates
end
rx_positions = zeros(N,D,T);
tx_positions = zeros(T,D);
losses = zeros(N,T);
for t = 1:T
    if t==1
        tx_positions(t,:) = init_lat_lon + deltas(t,:);
    else
        tx_positions(t,:) = tx_positions(t-1,:) + deltas(t,:);
    end
    for ii = 1:N
        d = X(ii,:,t)-jammer_loc(t,:);
        rx_positions(ii,:,t) = tx_positions(t,:) + km2deg(d*1e-3);
    end
end

for t = 1:T
    tx = txsite("Latitude",tx_positions(t,1), ...
    "Longitude",tx_positions(t,2), ...
    "TransmitterFrequency",f_jam,"TransmitterPower",P_tx);
    for ii = 1:N
        rx = rxsite("Latitude",rx_positions(ii,1,t), ...
        "Longitude",rx_positions(ii,2,t));
        %show(tx), show(rx) 
        %raytrace(tx,rx,pm)
%         raytrace(tx,rx,'NumReflections',[0 1 2])

        
        
        ss = sigstrength(rx,tx,pm)-30; % signal strength in dBm (accounts for several multipath rays)
        
        losses(ii,t) = P_tx - ss;
        
        % this would copute a loss per ray
        % losses(ii,t) = pathloss(pm,rx,tx);
    end
    

end





end

