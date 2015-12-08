function data = load_data()


DATASET_DIR = 'data\emailspam';


data.domain_names = {'U00', 'U01', 'U02', 'U03'};

for i = 1 : length(data.domain_names)-1
    t = load(fullfile(DATASET_DIR, sprintf('data_%d.mat', i))); 
    data.Xs{i} = t.features;
    data.ys{i} = t.labels;
end

t = load(fullfile(DATASET_DIR, sprintf('data_%d.mat', 4))); 
data.Xt = t.features;
data.yt = t.labels;


rand('seed', 123);
pos_index = find(data.yt==1);
pos_index = pos_index(randperm(length(pos_index)));
neg_index = find(data.yt==-1);
neg_index = neg_index(randperm(length(neg_index)));


tar_ind_dir = fullfile(DATASET_DIR, 'tar_ind2');
data.nRound = 10;

for i = 1 : 10
    t = load(fullfile(tar_ind_dir, [num2str(i) '.mat']));
    data.tar_train_index{i} = t.tar_training_ind(:);
    data.tar_test_index{i} = t.test_ind(:);
    data.tar_background_index{i} = t.tar_background_ind(:);
end
