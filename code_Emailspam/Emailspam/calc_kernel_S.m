function K = calc_kernel_S(kernel_type, kernel_param, S)
switch kernel_type
    case 'linear'
        K = S;
    case 'poly'
        K = (S+1).^kernel_param;
    otherwise
        error(mfilename);
end