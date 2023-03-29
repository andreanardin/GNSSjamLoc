function positions = random_walk2D(initial_pos,n_time_steps,delta)
    r = delta.*sign(randn(n_time_steps,2));
    positions = zeros(n_time_steps,2);
    positions(1,:) = initial_pos;
    
    for t = 2:n_time_steps
        positions(t,:) = positions(t-1,:) + r(t,:);
    end 
end