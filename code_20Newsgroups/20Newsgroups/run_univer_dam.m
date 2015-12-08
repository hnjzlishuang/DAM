function run_univer_dam(setting)
setpaths
data = load_data(setting);
result_dir = sprintf('result_%s', setting);
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

C = 1;
lambda_L=1;
lambda_D1=1;
lambda_D2=1;
beta = 10000;
thr = 0.3;
virtual_label_type = 'svm_s';
kernel_types = {'linear', 'poly'};
kernel_params = {0, 1.1:0.1:1.5};

N_set = [0 2 4 6 10 15 20];
for i = 1 : length(N_set)
    N = N_set(i);
    func = 'main_univerdam';
    eval(sprintf('result = %s(data, C, N, lambda_L, lambda_D1, lambda_D2, thr, beta, virtual_label_type, kernel_types, kernel_params);',func));
    result_file = fullfile(result_dir, sprintf('result_%s_%s_N=%d.mat',func, virtual_label_type, N));
    save(result_file, 'result', 'C', 'lambda_L', 'lambda_D1', 'lambda_D2', 'beta', 'kernel_types', 'kernel_params', 'thr');
end
show_result_all_univer_dam(setting, func, virtual_label_type);
