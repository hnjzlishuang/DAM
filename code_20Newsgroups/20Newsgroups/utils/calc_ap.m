function [ap] = calc_ap(gt, desc)
% ap = calc_ap(gt, desc)
%
% gt : ground truth labels [-1,1]
% desc : decision_values
%  
%----------------------------
% for test only
% gt = [-1 1 1 -1 -1];
% desc = [0 1 -1 0 0];
assert(length(gt)==length(desc));
gt = gt(:);
desc = desc(:);
[dv, ind] = sort(-desc); dv = -dv;
gt = gt(ind);
pos_ind = find( gt > 0 );
npos = length(pos_ind);
if npos == 0
    %warning('pos = 0');
    ap = 0;
else
    ap = mean( (1:npos)' ./ pos_ind(:) );
end
%fprintf('Average Precision (AP) = %f\n', ap);
end
