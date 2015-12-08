function beta = compute_beta(KA, KT,  B)
%
% -KA   : nA x nA
% -KB   : nA x nB
% -B    : parameter


nA = size(KA,1);
nT = size(KT,1);

if ~exist('B', 'var')
    B = 0.99;
end
epsilon = 5 * B / sqrt(nA);

KA = KA + diag(ones(nA,1)* 1e-8);

tic
kappa_vec = nA / nT * sum(KT, 2);

f   = - kappa_vec;
A   = [ones(1, nA); -ones(1, nA)];
b   = [(1+epsilon) * nA; -(1-epsilon) * nA];
lb  = zeros(nA,1);
ub  = ones(nA,1) * B;

[beta, obj, how, output, lambda] = quadprog(KA, f, A, b, [], [], lb, ub, [], []);
assert(how==1);
clear A b f lb ub
toc