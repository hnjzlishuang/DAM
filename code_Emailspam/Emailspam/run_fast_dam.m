function run_fast_dam()
fprintf('============= %s =================\n', mfilename);
setpaths

data = load_data();

C = 1;
lambda_L=1;
lambda_D=1;
beta = 100;
thr = 0.3;
virtual_label_type = 'svm_fr';
kernel_types = {'linear', 'poly'};
kernel_params = {0, 1.1:0.1:1.5};
result = main_fast_dam(data, C, lambda_L, lambda_D, thr, beta, virtual_label_type, kernel_types, kernel_params);
result_dir = 'results';
result_file = fullfile(result_dir, 'fast_dam', sprintf('result_main_fast_dam__%s.mat', virtual_label_type));
save(result_file, 'result', 'C', 'lambda_L', 'lambda_D', 'beta', 'kernel_types', 'kernel_params', 'thr');
show_result(result_file);
