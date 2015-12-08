function run_univer_dam()
fprintf('============= %s =================\n', mfilename);
setpaths

data = load_data();

C = 1;
lambda_L=1;
lambda_D1=1;
lambda_D2=1;
beta = 100;
thr = 0.3;
virtual_label_type = 'svm_fr';
kernel_types = {'linear', 'poly'};
kernel_params = {0, 1.1:0.1:1.5};
result = main_univerdam(data, C, lambda_L, lambda_D1, lambda_D2, thr, beta, virtual_label_type, kernel_types, kernel_params);
result_dir = 'results';
result_file = fullfile(result_dir, sprintf('result_main_univerdam__%s.mat', virtual_label_type));
save(result_file, 'result', 'C', 'lambda_L', 'lambda_D1', 'lambda_D2', 'beta', 'kernel_types', 'kernel_params', 'thr');
show_result(result_file);
