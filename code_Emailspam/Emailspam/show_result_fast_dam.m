function show_result(result_file)
load(result_file, 'result');

r = [result{:}]; r = [r.ap];
fprintf(':\n');
fprintf('ap='); fprintf('\t%g', r); fprintf('\n');
fprintf('%g\t%g\n', mean(r), std(r));


r = [result{:}]; r = [r.acc];
fprintf(':\n');
fprintf('acc='); fprintf('\t%g', r); fprintf('\n');
fprintf('%g\t%g\n', mean(r), std(r));