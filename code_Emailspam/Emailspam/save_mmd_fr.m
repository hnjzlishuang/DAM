function save_mmd_fr()
result_dir = 'results';
data = load_data();
kernel_types = {'linear','poly'};
kernel_params ={0, 1.1:0.1:1.5};

for s = 1 : length(data.Xs)
    Xs = data.Xs{s};
    ys = data.ys{s};
    
    mmd_dir = fullfile(result_dir, 'mmd_values_fr', data.domain_names{s});
    if ~exist(mmd_dir, 'dir')
        mkdir(mmd_dir);
    end
    
    S = full([data.Xt; Xs]*[data.Xt; Xs]');
    src_index = (1:size(Xs,1))' + size(data.Xt,1);
    tar_index = (1:size(data.Xt,1))';
    ss = zeros(length(src_index)+length(tar_index),1);
    ss(src_index) = 1./length(src_index);
    ss(tar_index) = -1./length(tar_index);
    
    for kt = 1 : length(kernel_types)
        kernel_type = kernel_types{kt};
        for kp = 1 : length(kernel_params{kt})
            kernel_param = kernel_params{kt}(kp);
            K = calc_kernel_S(kernel_type, kernel_param, S);
            K(src_index,src_index) = 2 * K(src_index, src_index);
            K(tar_index,tar_index) = 2 * K(tar_index, tar_index);
            
            mmd_file = fullfile(mmd_dir, sprintf('mmd_%s_%g.mat', kernel_type, kernel_param));
            if exist(mmd_file, 'file')
                load(mmd_file, 'mmd_value');
            else
                mmd_value = (ss'*K*ss);
                save(mmd_file, 'mmd_value');
            end
        end
    end
end