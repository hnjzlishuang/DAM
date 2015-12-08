function run_svm_fr()
fprintf('============= %s =================\n', mfilename);
setpaths

data = load_data();

C = 1;
kernel_types = {'linear', 'poly'};
kernel_params = {0, 1.1:0.1:1.5};
func = 'main_svm_fr';
eval(sprintf('result = %s(data, C, kernel_types, kernel_params);', func));


kernel_types = {'linear'};
kernel_params = {0};
func = 'main_svm_fr';
eval(sprintf('result = %s(data, C, kernel_types, kernel_params);', func));
result_dir = 'results';
result_file = fullfile(result_dir, 'svm_fr', sprintf('result_%s.mat',func));
save(result_file, 'result', 'C', 'kernel_types', 'kernel_params');
show_result(result_file);
