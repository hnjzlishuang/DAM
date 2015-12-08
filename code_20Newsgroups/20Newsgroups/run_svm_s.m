function run_svm_s(setting)
fprintf('=========== %s ============\n', mfilename);
setpaths
data = load_data(setting);
result_dir = sprintf('result_%s', setting);
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

C = 1;
func = 'main_svm_s';


kernel_types = {'linear', 'poly'};
kernel_params = {0, 1.1:0.1:1.5};
N_set = [0 2 4 6 10 15 20];
for i = 1 : length(N_set)
    N = N_set(i);   
    eval(sprintf('result = %s(data, C, N, kernel_types, kernel_params);', func));   
    result_file = fullfile(result_dir, sprintf('result_%s_N=%d.mat',func, N));
    save(result_file, 'result', 'C', 'kernel_types', 'kernel_params');
end


kernel_types = {'linear'};
kernel_params = {0};
N_set = [0 2 4 6 10 15 20];
for i = 1 : length(N_set)
    N = N_set(i);   
    eval(sprintf('result = %s(data, C, N, kernel_types, kernel_params);', func));   
    result_file = fullfile(result_dir, sprintf('result_%s_N=%d.mat',func, N));
    save(result_file, 'result', 'C', 'kernel_types', 'kernel_params');
end
show_result_all_svm_s(setting, func);
