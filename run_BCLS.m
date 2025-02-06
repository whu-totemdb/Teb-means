% Clear workspace and command window
clear all
clc

datasets = {'1-epsilon', '2-give_credit', '3-UCI_Credit_Card', '4-Spanish', '5-census1990', '6-hmda', '7-athlete', '8-svmlight'};
% 第三个数据集表现不好
for dataset_idx = 1:1%length(datasets)
    dataset_name = datasets{dataset_idx};
    fprintf('Running on dataset %s\n', dataset_name);
    file_path = strcat('dataset/output/', dataset_name,'.csv');
    X = csvread(file_path, 1, 1)';
    
    iter = 50;
    c = 4;


    gammas = [10^(-5)];
    lams = [10^(-3),10];
    rhos = [1.002];
    mus = [1.0];

    for gamma_idx = 1:length(gammas)
        gamma = gammas(gamma_idx);
        for lam_idx = 1:length(lams)
            lam = lams(lam_idx);


            for rho_idx = 1:length(rhos)
                rho = rhos(rho_idx);
                for mu = 1:50
%                     mu = mus(mu_idx);
                    rng(mu);
                    label = kmeans(X', c);

                    fprintf('gamma: %.5f, lam: %.5f, rho: %.3f, mu: %d\n', gamma, lam, rho, mu);
                    res = BCLS_ALM(X, label, c, gamma, lam, iter);
                end
            end
        end
    end
end