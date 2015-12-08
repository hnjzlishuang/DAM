function run_svm_t()
fprintf('=============== %s ===============\n', mfilename);
setpaths

data = load_data;

C = 1;
%==== prepare decision values
kernel_types = {'linear', 'poly'};
kernel_params = {0, 1.1:0.1:1.5};
main_svm_s(data, C, kernel_types, kernel_params);


kernel_types = {'linear'};
kernel_params = {0};
result = main_svm_t(data, C, kernel_types, kernel_params);


result_dir = 'results';
result_file = fullfile(result_dir, 'svm_t', 'result_main_svm_t.mat');
save(result_file,  'result', 'C', 'kernel_types', 'kernel_params');
show_result(result_file);
