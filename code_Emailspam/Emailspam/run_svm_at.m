function run_svm_at()
fprintf('============= %s =================\n', mfilename);
setpaths

data = load_data();

C = 1;
kernel_types = {'linear', 'poly'};
kernel_params = {0, 1.1:0.1:1.5};
result = main_svm_at(data, C, kernel_types, kernel_params);
result_dir = 'results';
result_file = fullfile(result_dir, 'svm_at', 'result_main_svm_at.mat');
save(result_file, 'result', 'C', 'kernel_types', 'kernel_params');
show_result(result_file);
