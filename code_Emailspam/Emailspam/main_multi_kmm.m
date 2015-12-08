function result = main_multi_kmm(data, C, alpha, kernel_types, kernel_params)
result_dir = 'results';
result_file = fullfile(result_dir, 'multi_kmm', ['result_', mfilename, '.txt']);
if ~exist(fullfile(result_dir, 'multi_kmm'), 'dir')
    mkdir(fullfile(result_dir, 'multi_kmm'));
end
log_print(result_file, '<==========  BEGIN @ %s, C = %g ============>\n', datestr(now), C);



dv_dir = fullfile(result_dir, 'multi_kmm', 'decision_values');
if ~exist(dv_dir, 'dir')
    mkdir(dv_dir);
end



X{1,1} = data.Xt;
y{1,1} = data.yt;
domain_index{1,1} = (1:length(data.yt))';
offset = length(data.yt);
for s = 1 : length(data.Xs)
    X{s+1,1} = data.Xs{s};
    y{s+1,1} = data.ys{s};
    domain_index{s+1,1} = (1:length(data.ys{s}))' + offset;
    offset = offset + length(data.ys{s});
end
X = cell2mat(X);
y = cell2mat(y);
S = full(X*X');
tar_index = domain_index{1,1};

for kt = 1 : length(kernel_types)
    kernel_type = kernel_types{kt};
    for kp = 1 : length(kernel_params{kt})
        kernel_param = kernel_params{kt}(kp);
        K = calc_kernel_S(kernel_type, kernel_param, S);
        K = shift_kernel(K, domain_index, alpha);
        
        for r = 1 : data.nRound
            tar_train_index = data.tar_train_index{r};
            tar_test_index = data.tar_test_index{r};
            train_index = [tar_train_index; cell2mat(domain_index(2:end))];
            n_train = length(train_index);
             
            dv_file = fullfile(dv_dir, sprintf('dv_round=%d_C=%g_alpha=%g_%s_%g.mat', r, C, alpha, kernel_type, kernel_param));
            if exist(dv_file, 'file')
                load(dv_file, 'decision_values');
            else
                model = svmtrain(y(train_index), [(1:length(train_index))', K(train_index,train_index)], sprintf('-c %g -t 4 -q', C));
                [~,~,decision_values] = svmpredict(y(tar_index), [(1:length(tar_index))', K(tar_index,train_index)], model);
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
    for kt = 1 : length(kernel_types)
        kernel_type = kernel_types{kt};
        for kp = 1 : length(kernel_params{kt})
            kernel_param = kernel_params{kt}(kp);
            dv_file = fullfile(dv_dir, sprintf('dv_round=%d_C=%g_alpha=%g_%s_%g.mat', r, C, alpha, kernel_type, kernel_param));
            load(dv_file, 'decision_values');
            ap = calc_ap(data.yt(tar_test_index), decision_values(tar_test_index));
            acc = mean(data.yt(tar_test_index)==sign(decision_values(tar_test_index)));
            log_print(result_file, '%g\t%g @ round=%d, C=%g, alpha=%g, kernel=%s, kernel_param=%g, %s\n', ap, acc, r, C, alpha, kernel_type, kernel_param, data.domain_names{s});
            all_test_dv = [all_test_dv, decision_values];
        end
    end
    dv = mean( 1./(1+exp(-all_test_dv)) , 2);
    ap = calc_ap(data.yt(tar_test_index), dv(tar_test_index));
    acc = mean(data.yt(tar_test_index)==sign(dv(tar_test_index)));
    result{r}.ap_sigmoid = ap;
    result{r}.acc_sigmoid = acc;
    log_print(result_file, 'SIGMIOD %g\t%g @ round=%d, C=%g, alpha=%g\n', ap, acc, r, C, alpha);
    
    dv = mean( all_test_dv , 2);
    ap = calc_ap(data.yt(tar_test_index), dv(tar_test_index));
    acc = mean(data.yt(tar_test_index)==sign(dv(tar_test_index)));
    result{r}.ap_no_sigmoid = ap;
    result{r}.acc_no_sigmoid = acc;
    log_print(result_file, 'NO SIGMIOD %g\t%g @ round=%d, C=%g, alpha=%g\n', ap, acc, r, C, alpha);
    
end
log_print(result_file, '<==========  END: %s @ %s, C = %g ============>\n', datestr(now), C);