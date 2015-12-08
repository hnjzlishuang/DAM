function [dv,model] = train_fast_dam(K, y, f_s, gamma_s, theta1,theta2, thr, f_p, lambda, C, epsilon)
%
% Inputs
%   K : n x n kernel matrix
%   y : n x 1 label vector with the label of unlabeled data set to 0
%   f_s : vitural label
%   gamma_s : weight of virtual label from different sources
%   theta1,theta2,C: regularizaion parameters
%   f_p : prior information, can be decision values from the model of other
%   classes
%
% Outputs
%   dv : decision values
%  model: trained model
%
assert(all(ismember(unique(y), [-1 0 1])));
n = size(K,1);
idx_l = find(y~=0);
idx_u = find(y==0);
II = ones(n,1); II(idx_u)=1/sum(gamma_s);
II(idx_l) = II(idx_l)./theta1;
II(idx_u) = II(idx_u)./theta2;
II = diag(II);

if ~isempty(f_p)
    Kp = (f_p*f_p')./lambda;
    hatK = K + II + Kp;
else
    hatK = K + II;
end
haty = f_s*gamma_s./sum(gamma_s);
haty(idx_l) = y(idx_l);

ind = find(abs(haty(idx_u))<thr);
v_ind = [idx_l; setdiff(idx_u, idx_u(ind))];

model = svmtrain(haty(v_ind), [(1:length(v_ind))', hatK(v_ind,v_ind)], sprintf('-s 3 -t 4 -c %g -p %g -q', C, epsilon));
model.SVs = sparse(v_ind(model.SVs));
if ~isempty(f_p)
    Ktest = K + Kp;
else
    Ktest = K;
end
dv = Ktest(:,model.SVs)*model.sv_coef - model.rho;
