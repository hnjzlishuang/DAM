function run_mcc_svm()
fprintf('============= %s =================\n', mfilename);
setpaths

data = load_data;
C = 1;
kernel_types = {'linear'};
kernel_params = {0};


result = main_mcc_svm(data, C, kernel_types, kernel_params);
result_dir = 'results';
result_file = fullfile(result_dir, 'mcc_svm', 'result_main_mcc_svm.mat');
save(result_file, 'result', 'C', 'kernel_types', 'kernel_params');
show_result(result_file);
