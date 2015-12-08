function model = merge_model(source_model, idx_s, adapt_model, idx_t)
%
% model = merge_model(source_model, idx_s, adapt_model, idx_t)
% Inputs:
%   -source_model
%   -idx_s : the index of the source data used to train the source model
%   -adapt_model :
%   -idx_t : the index of the target data used to train the target model
%  
assert(isequal(source_model.Parameters([2:3,5:end]), adapt_model.Parameters([2:3,5:end])));
assert(source_model.Parameters(1)==0);
assert(adapt_model.Parameters(1)==5);
assert(isequal(source_model.nr_class, adapt_model.nr_class));
assert(isequal(source_model.Label, adapt_model.Label));
model = source_model;
model.Parameters(4) = 1./(length(idx_s)+length(idx_t));
model.totalSV = source_model.totalSV + adapt_model.totalSV;


model.rho = source_model.rho + adapt_model.rho;


model.nSV = source_model.nSV + adapt_model.nSV;
model.sv_coef = [source_model.sv_coef; adapt_model.sv_coef];
idx_s = idx_s(:); idx_t = idx_t(:);
model.SVs = [idx_s(source_model.SVs); idx_t(adapt_model.SVs)];
[model.SVs, idx] = sort(model.SVs);
model.SVs = sparse(model.SVs);
model.sv_coef = model.sv_coef(idx);