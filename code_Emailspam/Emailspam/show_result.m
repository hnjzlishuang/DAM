function show_result(result_file)
load(result_file, 'result');

r = [result{:}]; r = [r.ap];
fprintf('MAP\tSTD\n');
fprintf('%g\t%g\n', mean(r), std(r));