function show_result_svm_t(result_file)
load(result_file, 'result');

r = [result{:}]; r = [r.ap_sigmoid];
fprintf('SIGMOID:\n');
fprintf('ap='); fprintf('\t%g', r); fprintf('\n');
fprintf('%g\t%g\n', mean(r), std(r));


r = [result{:}]; r = [r.ap_no_sigmoid];
fprintf('NO_SIGMOID:\n');
fprintf('ap='); fprintf('\t%g', r); fprintf('\n');
fprintf('%g\t%g\n', mean(r), std(r));

r = [result{:}]; r = [r.acc_no_sigmoid];
fprintf('SIGMOID:\n');
fprintf('acc='); fprintf('\t%g', r); fprintf('\n');
fprintf('%g\t%g\n', mean(r), std(r));


r = [result{:}]; r = [r.acc_no_sigmoid];
fprintf('NO_SIGMOID:\n');
fprintf('acc='); fprintf('\t%g', r); fprintf('\n');
fprintf('%g\t%g\n', mean(r), std(r));