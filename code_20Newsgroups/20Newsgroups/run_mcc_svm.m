function run_mcc_svm(setting)
fprintf('=========== %s ==============\n', mfilename);
setpaths
data = load_data(setting);
result_dir = sprintf('result_%s', setting);
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

C = 1;
kernel_types = {'linear'};
kernel_params = {0};

N_set = [2 4 6 10 15 20];
for i = 1 : length(N_set)
    N = N_set(i);
    func = 'main_mcc_svm';
    eval(sprintf('result = %s(data, C, N, kernel_types, kernel_params);', func));   
    result_file = fullfile(result_dir,sprintf('result_%s_N=%d.mat',func, N));
    save(result_file, 'result', 'C', 'kernel_types', 'kernel_params');  
end
show_result_all_mcc_svm(setting,func);
