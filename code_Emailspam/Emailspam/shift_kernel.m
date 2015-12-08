function K_shift = shift_kernel(K, domain_index, alpha)

% first one is target domain
K_shift = zeros(size(K));
n = length(domain_index{1});
idx_t = domain_index{1};


for si = 2 : length(domain_index)
    idx = domain_index{si};
    m = length(idx);
    a = -alpha/m;
    b = alpha/n;
    % same source vs same source  
    Kss = K(idx,idx) + a.^2 * sum(sum(K(idx,idx))) + b^2 * sum(sum(K(idx_t,idx_t))) ...
        + a * (K(idx,idx)*ones(m,1))*ones(1,m) + a*b*sum(sum(K(idx,idx_t))) + b * (K(idx,idx_t)*ones(n,1))*ones(1,m)...
        + a * ones(m,1)*(ones(1,m)*K(idx,idx)) + a*b*sum(sum(K(idx_t,idx))) + b * ones(m,1)*(ones(1,n)*K(idx_t,idx));
    K_shift(idx,idx) = Kss;
    clear Kss
    
    % source vs target
    Kst = K(idx,idx_t);
    Kst = Kst + a * ones(m,1)*(ones(1,m)*Kst) + b * ones(m,1)*(ones(1,n)*K(idx_t,idx_t));
    K_shift(idx,idx_t) = Kst;
    K_shift(idx_t,idx) = Kst';
    clear Kst
end

% source vs source
K_shift(idx_t,idx_t) = K(idx_t,idx_t);
for s1 = 2 : length(domain_index)
    for s2 = s1+1 : length(domain_index)
        idx_1 = domain_index{s1};
        m1 = length(idx_1);
        a1 = -alpha/m1;
        
        idx_2 = domain_index{s2};
        m2 = length(idx_2);
        a2 = -alpha/m2;
        
        K12 = K(idx_1,idx_2);
        K12 = K12 + a1*a2 * sum(sum(K12)) + b^2 * sum(sum(K(idx_t,idx_t))) ...
            + a1 * ones(m1,1)* (ones(1,m1)*K12) + a1*b*sum(sum(K(idx_1,idx_t))) + b * (K(idx_1,idx_t)*ones(n,1))*ones(1,m2)...
            + a2 * (K12 * ones(m2,1)) * ones(1,m2) + a2*b*sum(sum(K(idx_t,idx_2))) + b * ones(m1,1)*(ones(1,n)*K(idx_t,idx_2));
        
        K_shift(idx_1,idx_2) = K12;
        K_shift(idx_2,idx_1) = K12';
    end
end