setting = 'sci_vs_comp';

run_svm_s(setting);
run_svm_t(setting);
run_svm_fr(setting);
run_mcc_svm(setting);
run_multi_kmm(setting);

save_mmd_at(setting);
save_mmd_fr(setting);


run_svm_at(setting);
run_fast_dam(setting);
run_univer_dam(setting);
