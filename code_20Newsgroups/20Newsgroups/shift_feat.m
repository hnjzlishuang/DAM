function X = shift_feat(X, domain_index, alpha)
X = full(X);
% first one is target domain
idx_t = domain_index{1};
n =length(idx_t);
xt_m0 = mean(X(idx_t,:));
for si = 2 : length(domain_index)
    idx = domain_index{si};
    m = length(idx);
    xs_m0 = mean(X(idx,:));
    X(idx,:) = X(idx,:) - alpha*( ones(m,1)*xs_m0 - ones(m,1)*xt_m0);
end
