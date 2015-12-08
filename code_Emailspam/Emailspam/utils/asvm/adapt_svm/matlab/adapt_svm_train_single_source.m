function adapt_model = adapt_svm_train_single_source(yt, Ktt, C, Kts, source_model)
% 
% adapt_model = adapt_svm_train_single_source(y, Ktt, C, Kts, source_model)
% Inputs
%   yt   : n x y label vector
%   Ktt  : n x n kernel matrix
%   options :
%   Kts : n x m kernel matrix, m is the number of samples used to train the source_model
%   source_model :  the model is trained based on Kts
%

assert(isequal(unique(yt), [-1 1]'));
n = length(yt);
assert(size(Ktt,1)==size(Kts,1));
assert(n==size(Ktt,1));


p = Kts(:, source_model.SVs) * source_model.sv_coef - source_model.rho; % NOTE: the sign problem should be corrected in the source_model
p = p .* yt;
p = p - 1;
adapt_model = svmtrain_casvm_p(Ktt, yt, p, C);