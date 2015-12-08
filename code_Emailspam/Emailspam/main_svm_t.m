function result = main_svm_t(data, C, kernel_types, kernel_params)
result_dir = 'results';
result_file = fullfile(result_dir, 'svm_t', ['result_', mfilename, '.txt']);
if ~exist(fullfile(result_dir, 'svm_t'), 'dir')
    mkdir(fullfile(result_dir, 'svm_t'));
end
log_print(result_file, '<==========  BEGIN @ %s, C = %g ============>\n', datestr(now), C);


dv_dir = fullfile(result_dir, 'svm_t', 'decision_values', data.domain_names{end});
if ~exist(dv_dir, 'dir')
    mkdir(dv_dir);
end


S = full(data.Xt*data.Xt');

for kt = 1 : length(kernel_types)
    kernel_type = kernel_types{kt};
    for kp = 1 : length(kernel_params{kt})
        kernel_param = kernel_params{kt}(kp);
        K = calc_kernel_S(kernel_type, kernel_param, S);
        for r = 1 : data.nRound
            tar_train_index = data.tar_train_index{r};
            tar_test_index = data.tar_test_index{r};
            
            dv_file = fullfile(dv_dir, sprintf('dv_round=%d_C=%g_%s_%g.mat', r, C, kernel_type, kernel_param));
            if exist(dv_file, 'file')
                load(dv_file, 'decision_values');
            else
                model = svmtrain(data.yt(tar_train_index), [(1:length(tar_train_index))', K(tar_train_index,tar_train_index)], sprintf('-c %g -t 4 -q', C));
                [tt,tt,decision_values] = svmpredict(data.yt, [(1:length(data.yt))', K(:,tar_train_index)], model);
                decision_values = decision_values * model.Label(1);
                save(dv_file, 'decision_values');
            end
        end
    end
end


for r = 1 : data.nRound
    tar_train_index = data.tar_train_index{r};
    tar_test_index = data.tar_test_index{r};
    all_test_dv = [];
    dv_dir = fullfile(result_dir, 'svm_t', 'decision_values', data.domain_names{end});
    for kt = 1 : length(kernel_types)
        kernel_type = kernel_types{kt};
        for kp = 1 : length(kernel_params{kt})
            kernel_param = kernel_params{kt}(kp);
            
            dv_file = fullfile(dv_dir, sprintf('dv_round=%d_C=%g_%s_%g.mat', r, C, kernel_type, kernel_param));
            load(dv_file, 'decision_values');
            ap = calc_ap(data.yt(tar_test_index), decision_values(tar_test_index));
            acc = mean(data.yt(tar_test_index)==sign(decision_values(tar_test_index)));
            log_print(result_file, '%g\t%g @ round=%d, C=%g, kernel=%s, kernel_param=%g, %s\n', ap, acc, r, C, kernel_type, kernel_param, data.domain_names{end});
            all_test_dv = [all_test_dv, decision_values];
        end
    end
    dv = mean( 1./(1+exp(-all_test_dv)) , 2);
    ap = calc_ap(data.yt(tar_test_index), dv(tar_test_index));
    acc = mean(data.yt(tar_test_index)==sign(dv(tar_test_index)));
    result{r}.ap_sigmoid = ap;
    result{r}.acc_sigmoid = acc;
    log_print(result_file, 'SIGMIOD %g\t%g @ round=%d, C=%g\n', ap, acc, r, C);
    
    dv = mean( all_test_dv , 2);
    ap = calc_ap(data.yt(tar_test_index), dv(tar_test_index));
    acc = mean(data.yt(tar_test_index)==sign(dv(tar_test_index)));
    result{r}.ap_no_sigmoid = ap;
    result{r}.acc_no_sigmoid = acc;
    log_print(result_file, 'NO_SIGMIOD %g\t%g @ round=%d, C=%g\n', ap, acc, r, C);    
end
log_print(result_file, '<==========  END: %s @ %s, C = %g ============>\n', datestr(now), C);