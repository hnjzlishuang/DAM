function show_result_all_svm_s(setting, func)

result_dir = sprintf('result_%s', setting);

N_set = [0 2 4 6 10 15 20];
fprintf('N\tMAP\tSTD\n');
for i = 1 : length(N_set)
    N = N_set(i);  
    result_file = fullfile(result_dir, sprintf('result_%s_N=%d.mat',func, N)); 
    load(result_file);
    r = [result{:}]; r = [r.ap_sigmoid];
    fprintf('%d\t%g\t%g\n', N, mean(r), std(r));
end