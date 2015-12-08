function result = main_fast_dam(data, C, N, lambda_L, lambda_D, thr, beta, virtual_label_type, kernel_types, kernel_params)
result_dir = sprintf('result_%s', data.setting);
result_file = fullfile(result_dir, ['result_', mfilename, '.txt']);
log_print(result_file, '<==========  BEGIN @ %s, C = %g, lambda_L = %g, lambda_D = %g , thr = %g, beta = %g============>\n', datestr(now), C, lambda_L, lambda_D, thr, beta);



K = full(data.Xt*data.Xt');

for r = 1 : data.nRound
    tar_train_index = return_ind2(data.perm_tar_index{r}, sum(data.yt==1), N);
    tar_test_index = setdiff( (1:length(data.yt))', tar_train_index);
    all_test_dv = [];
    mmd_values = [];
    for s = 1 : length(data.Xs)
        if isequal(virtual_label_type(end-2:end), '_fr')
            dv_dir = fullfile(result_dir, 'decision_values', 'svm_fr', data.domain_names{s});
            mmd_dir = fullfile(result_dir , 'mmd_values_fr', data.domain_names{s});
        elseif isequal(virtual_label_type(end-1:end), '_s')
            dv_dir = fullfile(result_dir,  'decision_values','svm_s', data.domain_names{s});
            mmd_dir = fullfile(result_dir,  'mmd_values_at', data.domain_names{s});
        elseif isequal(virtual_label_type(end-2:end), '_at')
            dv_dir = fullfile(result_dir, 'decision_values','svm_at',  data.domain_names{s});
            mmd_dir = fullfile(result_dir, 'mmd_values_at', data.domain_names{s});
        end
        for kt = 1 : length(kernel_types)
            kernel_type = kernel_types{kt};
            for kp = 1 : length(kernel_params{kt})
                kernel_param = kernel_params{kt}(kp);
                if isequal(virtual_label_type(end-1:end), '_s')
                    dv_file = fullfile(dv_dir, sprintf('dv_C=%g_%s_%g.mat',C, kernel_type, kernel_param));
                else
                    dv_file = fullfile(dv_dir, sprintf('dv_round=%d_N=%d_C=%g_%s_%g.mat', r, N, C, kernel_type, kernel_param));
                end
                load(dv_file, 'decision_values');
                mmd_file = fullfile(mmd_dir, sprintf('mmd_%s_%g.mat', kernel_type, kernel_param));
                load(mmd_file, 'mmd_value');
                mmd_values = [mmd_values; mmd_value];
                all_test_dv = [all_test_dv, decision_values];
            end
        end
    end
    y = data.yt;
    y(tar_test_index) = 0;
    f_s = all_test_dv;
    pp = -beta * mmd_values.^2; %pp = pp - max(pp);
    gamma_s = exp(pp);
    gamma_s = gamma_s./sum(gamma_s);
    theta1 = lambda_L;
    theta2 = lambda_D;
    epsilon = 0.1;
    [dv,model] = train_fast_dam(K, y, f_s, gamma_s, theta1,theta2, thr, [], [], C, epsilon);
    
    ap = calc_ap(data.yt(tar_test_index), dv(tar_test_index));
    acc = mean(data.yt(tar_test_index)==sign(dv(tar_test_index)));
    result{r}.ap = ap;
    result{r}.acc = acc;
    log_print(result_file, '******%g\t%g @ round=%d, C=%g, lambda_L=%g, lambda_D=%g, thr=%g\n', ap, acc, r, C, lambda_L, lambda_D, thr);
end
log_print(result_file, '<==========  END @ %s, C = %g, lambda_L = %g, lambda_D = %g, thr = %g, beta = %g ============>\n', datestr(now), C, lambda_L, lambda_D, thr, beta);
