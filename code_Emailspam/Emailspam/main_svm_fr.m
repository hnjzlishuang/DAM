function result = main_svm_fr(data, C, kernel_types, kernel_params)
result_dir = 'results';
result_file = fullfile(result_dir, 'svm_fr', ['result_', mfilename, '.txt']);
if ~exist(fullfile(result_dir, 'svm_fr'), 'dir')
    mkdir(fullfile(result_dir, 'svm_fr'));
end
log_print(result_file, '<==========  BEGIN @ %s, C = %g ============>\n', datestr(now), C);


for s = 1 : length(data.Xs)    
    dv_dir = fullfile( result_dir, 'svm_fr', 'decision_values',data.domain_names{s});
    if ~exist(dv_dir, 'dir')
        mkdir(dv_dir);
    end
    
    Xs = data.Xs{s};
    ys = data.ys{s};
    
    
    S = full([data.Xt;Xs]*[data.Xt;Xs]');
    y = [data.yt; ys];
    src_index = (1:size(Xs,1))' + size(data.Xt,1);
    tar_index = (1:size(data.Xt,1))';
    
    
    for kt = 1 : length(kernel_types)
        kernel_type = kernel_types{kt};
        for kp = 1 : length(kernel_params{kt})
            kernel_param = kernel_params{kt}(kp);
            K = calc_kernel_S(kernel_type, kernel_param, S);
           
            K(src_index,src_index) = 2 * K(src_index,src_index);
            K(tar_index,tar_index) = 2 * K(tar_index,tar_index);
            
            
            for r = 1 : data.nRound
                tar_train_index = data.tar_train_index{r};
                tar_test_index = data.tar_test_index{r};
                train_index = [src_index; tar_train_index];
                
                dv_file = fullfile(dv_dir, sprintf('dv_round=%d_C=%g_%s_%g.mat', r, C, kernel_type, kernel_param));
                if exist(dv_file, 'file')
                    load(dv_file, 'decision_values');
                else
                    model = svmtrain(y(train_index), [(1:length(train_index))', K(train_index,train_index)], sprintf('-c %g -t 4 -q', C));
                    [tt,tt,decision_values] = svmpredict(y(tar_index), [(1:length(tar_index))', K(tar_index,train_index)], model);
                    decision_values = decision_values * model.Label(1);
                    save(dv_file, 'decision_values');
                end               
            end
        end
    end
end


for r = 1 : data.nRound
    tar_train_index = data.tar_train_index{r};
    tar_test_index = data.tar_test_index{r};
    all_test_dv = [];
    for s = 1 : length(data.Xs)
        dv_dir = fullfile(result_dir, 'svm_fr', 'decision_values', data.domain_names{s});
        for kt = 1 : length(kernel_types)
            kernel_type = kernel_types{kt};
            for kp = 1 : length(kernel_params{kt})
                kernel_param = kernel_params{kt}(kp);
                
                dv_file = fullfile(dv_dir, sprintf('dv_round=%d_C=%g_%s_%g.mat', r, C, kernel_type, kernel_param));
                load(dv_file, 'decision_values');
                ap = calc_ap(data.yt(tar_test_index), decision_values(tar_test_index));
                acc = mean(data.yt(tar_test_index)==sign(decision_values(tar_test_index)));
                log_print(result_file, '%g\t%g @ round=%d, C=%g, kernel=%s, kernel_param=%g, %s\n', ap, acc, r, C, kernel_type, kernel_param, data.domain_names{s});
                all_test_dv = [all_test_dv, decision_values];
            end
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