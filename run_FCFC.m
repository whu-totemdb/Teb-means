% Clear workspace and command window
clear all
clc
% 10e-3, 10e-4, 10e-3, 10e-3, 10e-3, 10e-3, 10e-3,
datasets = {'1-epsilon', '2-give_credit', '3-UCI_Credit_Card', '4-Spanish', '5-census1990', '6-hmda', '7-athlete', '8-svmlight'};
% 第三个数据集表现不好
for dataset_idx = 8:8%length(datasets)
    dataset_name = datasets{dataset_idx};
    fprintf('Running on dataset %s\n', dataset_name);
    file_path = strcat('dataset/output/', dataset_name,'.csv');
    X = csvread(file_path, 1, 1)';
    
    iter = 50;
    c = 4;
    label = kmeans(X', c);

    gammas = [10^(-6),10^(-5),10^(-4),10^(-3)];
    lams = [10^(-3)];
    rhos = [1.002];
    mus = [1.0];

    for gamma_idx = 1:length(gammas)
        gamma = gammas(gamma_idx);
        for lam_idx = 1:length(lams)
            lam = lams(lam_idx);

            for rho_idx = 1:length(rhos)
                rho = rhos(rho_idx);
                for mu_idx = 1:length(mus)
                    mu = mus(mu_idx);
    
                    fprintf('gamma: %.5f, lam: %.5f, rho: %.3f, mu: %.2f\n', gamma, lam, rho, mu);
                    res = FCFC(X', label, c, gamma, iter);
                end
            end
        end
    end
end