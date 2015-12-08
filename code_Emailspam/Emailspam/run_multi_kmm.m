function run_multi_kmm()
fprintf('============= %s =================\n', mfilename);
setpaths

data = load_data();

C = 1;
kernel_types = {'linear'};
kernel_params = {0};

alpha = 1;
func = 'main_multi_kmm';
eval(sprintf('result = %s(data, C, alpha,  kernel_types, kernel_params);', func));
result_dir = 'results';
result_file = fullfile(result_dir, 'multi_kmm', sprintf('result_%s.mat',func));
save(result_file, 'result', 'C', 'kernel_types', 'kernel_params');
show_result(result_file);
