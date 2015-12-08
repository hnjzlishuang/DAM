function h = calc_mmd(K,s)
M = size(K,1);
h = zeros(M,1);
KK = zeros(size(K,2),size(K,3));
for i = 1 : M
    KK(:,:) = K(i,:,:);
    h(i) = (s'*KK*s);
end