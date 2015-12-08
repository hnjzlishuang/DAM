function show_result_all_fast_dam(setting, func, virtual_label_type)

result_dir = sprintf('result_%s', setting);

fprintf('N\tMAP\tSTD\n');
N_set = [ 0 2 4 6 10 15 20];
for i = 1 : length(N_set)
    N = N_set(i);
    result_file = fullfile(result_dir, sprintf('result_%s_%s_N=%d.mat',func,virtual_label_type, N)); 
    load(result_file);
    r = [result{:}]; r = [r.ap];
    fprintf('%d\t%g\t%g\n',N, mean(r), std(r));
end
