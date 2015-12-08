function result = main_fast_dam(data, C, lambda_L, lambda_D, thr, beta, virtual_label_type, kernel_types, kernel_params)
result_dir = 'results';
result_file = fullfile(result_dir, 'fast_dam', ['result_', mfilename, '.txt']);
if ~exist(fullfile(result_dir, 'fast_dam'), 'dir')
    mkdir(fullfile(result_dir, 'fast_dam'));
end
log_print(result_file, '<==========  BEGIN @ %s, C = %g, lambda_L = %g, lambda_D = %g , thr = %g, beta = %g============>\n', datestr(now), C, lambda_L, lambda_D, thr, beta);



K = full(data.Xt*data.Xt');

for r = 1 : data.nRound
    tar_train_index = data.tar_train_index{r};
    tar_test_index = data.tar_test_index{r};
    all_test_dv = [];
    mmd_values = [];
    for s = 1 : length(data.Xs)
        if isequal(virtual_label_type(end-2:end), '_fr')
            dv_dir = fullfile(  result_dir, 'svm_fr','decision_values', data.domain_names{s});
            mmd_dir = fullfile( result_dir, 'mmd_values_fr', data.domain_names{s});
        elseif isequal(virtual_label_type(end-1:end), '_s')
            dv_dir = fullfile( result_dir, 'svm_s', 'decision_values', data.domain_names{s});
            mmd_dir = fullfile( result_dir, 'mmd_values_at', data.domain_names{s});
        elseif isequal(virtual_label_type(end-2:end), '_st')
            dv_dir = fullfile( result_dir, 'svm_at', 'decision_values',  data.domain_names{s});
            mmd_dir = fullfile(result_dir, 'mmd_values_at', data.domain_names{s});
        else
            error('the type of the pre-learned classifier is not supported.');
        end
        for kt = 1 : length(kernel_types)
            kernel_type = kernel_types{kt};
            for kp = 1 : length(kernel_params{kt})
                kernel_param = kernel_params{kt}(kp);
                dv_file = fullfile(dv_dir, sprintf('dv_round=%d_C=%g_%s_%g.mat', r, C, kernel_type, kernel_param));
                try
                    load(dv_file, 'decision_values');
                catch ex
                    disp(ex.message);
                    error('You need to run the required baseline algorithms to obtain the decision values required by algorithm');
                end
                mmd_file = fullfile(mmd_dir, sprintf('mmd_%s_%g.mat', kernel_type, kernel_param));
                try
                   load(mmd_file, 'mmd_value');
                catch ex
                    disp(ex.message);
                    error('please run the proper save_mmd first to prepare the mmd values required by this algorithm.');
                end
                mmd_values = [mmd_values; mmd_value];
                all_test_dv = [all_test_dv, decision_values];
            end
        end
    end
    y = data.yt;
    y(tar_test_index) = 0;
    f_s = all_test_dv;
    gamma_s = exp(-beta * mmd_values.^2);
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