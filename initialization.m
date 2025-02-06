function initialization(X, cluse_num, tmp_dir, infRes)
%INITIALIZATION Initialize the original data and other variates
%   Detailed explanation goes here

% load data
c = cluse_num;
Y = kmeans(X', c);
gt = Y;
Data_ori = X;

[~, n]=size(Data_ori);
% [X, k, share] = pcaInit(Data_ori, infRes);
X = Data_ori;

% centralization
% H = eye(n) - 1/n*ones(n);
% X = X*H;                    

meanX = mean(X);
% 
% % 中心化矩阵
X = X - meanX;

% Step 1: Construct centralization matrix H as sparse matrix
% H = speye(n) - (1/n) * sparse(ones(n, 1));

% Step 2: Apply centralization
% X = X * H;

save([tmp_dir 'init.mat'], 'X', 'gt', 'c');

end
