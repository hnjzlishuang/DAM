function [tar_training_ind, test_ind] = return_ind2(tar_training_ind, tar_pos_len, nSample)

if nargin < 3
    error('vargin < 3');
end

tmp1 = tar_training_ind([1:nSample, tar_pos_len+1:tar_pos_len+nSample]);
% tmp2 = setdiff(tar_training_ind, tmp1);

test_ind = setdiff(tar_training_ind, tmp1);

tar_training_ind = tmp1(:);