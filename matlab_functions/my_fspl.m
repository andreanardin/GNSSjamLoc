function L = my_fspl(jammer_loc,X,f_jam, gamma)
%Compute the free space path loss
%   short distances are managed

[N,~,T] = size(X);
lambda = physconst('lightspeed')/f_jam;

distances = zeros(N,T);
for ii = 1:T
    for jj = 1:N
        distances(jj,ii) = norm(X(jj,:,ii)-jammer_loc(ii,:));
    end
end

%--- Path loss function
if nargin <= 3
    % path loss assumption check
    if min(min(distances)) <= lambda/4 %lambda/4 ensures the far field assumption for path loss holds
        warning 'free space path loss formula does not hold in the near-field'
    end
    % free space path loss (dB)
    L = 20*log10(4*pi*distances*f_jam/physconst('lightspeed'));
else
    % if gamma is given assume a simplified exponential path loss model
    L = gamma*10*log10(distances);
    if min(min(distances)) <= pi %lambda/4 is pi if plain pathloss formula is used (gamma*10*log10(d))
        warning 'path loss formula does not hold in the near-field'
    end
end


% L = 20*log10(4*pi*sqrt(distances)*f_jam/physconst('lightspeed')); % TESTING PURPOSE

if sum(sum(L<0))
    L(L<0)=0;
    warning 'excessivey short distances approximated as zero loss'
end

end

