function data = load_comp_vs_rec()

% 1     798	alt.atheism
% 2     970	comp.graphics
% 3     963	comp.os.ms-windows.misc
% 4     979	comp.sys.ibm.pc.hardware
% 5     958	comp.sys.mac.hardware
% 6     982	comp.windows.x
% 7     964	misc.forsale
% 8     987	rec.autos
% 9     993	rec.motorcycles
% 10	991	rec.sport.baseball
% 11	997	rec.sport.hockey
% 12	989	sci.crypt
% 13	984	sci.electronics
% 14	987	sci.med
% 15	985	sci.space
% 16	997	soc.religion.christian
% 17	909	talk.politics.guns
% 18	940	talk.politics.mideast
% 19	774	talk.politics.misc
% 20	627	talk.religion.misc

DATASET_DIR = 'data\20Newsgroups';


disp('comp vs rec or 2 3 4 5 vs 8 9 10 11');

DATASET_DIR = fullfile(DATASET_DIR, 'comp_vs_rec');

load(fullfile(DATASET_DIR, 'src_data.mat'));
% 2 vs 8
data.Xs{1} = src1_features;  
data.ys{1} = src1_labels;
data.domain_names{1} = '2vs8';

% 3 vs 9
data.Xs{2} = src2_features;
data.ys{2} = src2_labels;
data.domain_names{2} = '3vs9';

% 4 vs 10
data.Xs{3} = src3_features;
data.ys{3} = src3_labels;
data.domain_names{3} = '4vs10';

load(fullfile(DATASET_DIR, 'tar_data.mat'));
% 5 vs 11
data.Xt = tar_features;
data.yt = tar_labels;
data.domain_names{4} = '5vs11';

data.nRound = 10;
for r = 1 : 10
    t = load(fullfile(DATASET_DIR, 'tar_ind2', [num2str(r), '_all.mat']));
    data.perm_tar_index{r} = t.tar_training_ind(:);   
end
