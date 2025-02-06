
% Directory setting
base_dir = 'E:\f-means\siff-means\';
tmp_dir = [base_dir 'c/'];

% Parameter setting
gamma = 10^(-5); % the value of Gamma should be between 0 and 10^(-10)
lam = 10^(1);
mu = 1;
infRes = 0.90;    % the percentage of information reserved of the data during PCA dimension reduction
% data = 'UMIST';census1990_500000
% data = 'census1990_500000';
% data = 'UMIST';

save([tmp_dir 'param.mat']);
