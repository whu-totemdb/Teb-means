% Clear workspace and command window
clear all
clc

% 0.test time
num_runs = 3;

% 1.comparison algorithms
methods = { 'SIFF_eta', 'Lloyd', 'CDKM', 'FCFC', 'BCLS_ALM', 'F3KM', 'SIFF' };
% methods = { 'SIFF_eta', 'CDKM', 'FCFC', 'BCLS_ALM', 'Lloyd', 'F3KM', 'SIFF' };

% 2.datasets
datasets = {'1-epsilon', '2-svmlight', '3-athlete', '4-Spanish', '5-hmda', '6-census1990', 'ori-disease', 'ori-Crime'};

% 3.thread number
threads_set = 1;

% 4.the k value size
% clusters_size = {3,4,5,6,7,8,9,10};
clusters_size = {3};


% 5.different rho/eta value
% eta_set = [0.83];
eta_set = 0.79:0.01:0.86;
 
% 6.block size
% blocks_size = [1,2,4,8,16,32,64,128,256,512];
blocks_size = [8];


result_file_name2 = 'obj-iter200.csv';
fid_time = fopen(result_file_name2, 'a');
% fprintf(fid_time, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', 'dataset', 'k', 'method', ' MEAN_SSE', 'VAR_SSE', 'MEAN_BALANCE_LOSS', 'VAR_BALANCE_LOSS', 'MEAN_CV', 'MEAN_Nentro', 'MEAN_TIME');
fprintf(fid_time, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', 'dataset', 'block size', 'method', ' MEAN_SSE', 'VAR_SSE', 'MEAN_BALANCE_LOSS', 'VAR_BALANCE_LOSS', 'MEAN_CV', 'MEAN_Nentro', 'MEAN_TIME');
% fprintf(fid_time, '%s,%s,%s,%s,%s\n', 'dataset', 'eta', 'obj value', 'balance loss', 'SSE');


% 1st loop: dataset
for dataset_idx = 1:length(datasets)
    dataset_name = datasets{dataset_idx};
    fprintf('Running on dataset %s\n', dataset_name);
    
    % 2nd loop: k value
    for iter_cluster = 1:length(clusters_size)
        c = clusters_size{iter_cluster};
        fprintf('k: %d \n', c);

        % 3rd loop: thread number
        for threads_num = threads_set
            threads = threads_num;

            % 4th loop: methods
            for method_idx = 7:7
                method_name = methods{method_idx};

                % 5th loop: rho/eta
                for eta_idx = 1:length(eta_set)
                    eta = eta_set(eta_idx);

                    % 6th loop: block size
                    for block_size_idx = 1:length(blocks_size)
                        block_size = blocks_size(block_size_idx);

                        % Construct the file path and load data
                        file_path = strcat('dataset/output/', dataset_name,'.csv');
                        X = csvread(file_path, 1, 1)';
                        [~,l] = size(X);
                        if l >= 5000
                            X = X(:, 1:5000);
                        end

                        if dataset_idx == 7
                            X = [X(60:75, :);X(495:560, :);X(650:655, :);X(85:90, :);X(465:470, :);X(125:130, :)];
                        end
                        [~,l] = size(X);
      
                        iter = 200;

                        % Initialize variables for averaging
                        SSEs = zeros(1, num_runs);
                        BALANCE_LOSSs = zeros(1, num_runs);
                        CVs = zeros(1, num_runs);
                        Nentros = zeros(1, num_runs);
                        TIMEs = zeros(1, num_runs);

                        % data in each iteration
                        iter_SSEs = zeros(num_runs, iter);
                        iter_BLs = zeros(num_runs, iter);

                        for ite_run = 1:num_runs
%                             fprintf('Running %s on dataset %s, run %d\n', method_name, dataset_name, ite_run);
                            % Initialize labels
                            seed = 3 + ite_run;
                            rng(seed);
                            fprintf('seed: %d   ', seed);
                            label = kmeans(X', c);
      
                            % Delete any existing parallel pool
                            delete(gcp('nocreate'));
           
                            % Call the method dynamically
                            switch method_name
                                case 'SIFF_eta'
                                    rho = (1 - eta) / eta;
                                    fprintf('Block size: %d\n', block_size);
%                                     fprintf('============ eta: %.6f, rho: %.6f ============\n', eta, rho);
                                    [Y_label, ~, iter_num, obj_max, balance_loss, elapsed_time] = SIFF_eta(X, label, c, block_size, eta, iter);
                                    loss = NaN;
                                case 'Lloyd'
                                    [Y_label, ~, iter_num, obj_max, balance_loss, elapsed_time, size0] = Lloyd(X, label, c, iter);
                                    sse = obj_max;
                                    loss = NaN;
                                case 'CDKM'
                                    [Y_label, ~, iter_num, obj_max, balance_loss, elapsed_time, size0] = CDKM(X, label, c, iter);
                                    sse = obj_max;
                                    loss = NaN;
                                case 'FCFC'
                                    [Y_label, ~, sse, balance_loss, elapsed_time, size0] = FCFC(X', label, c, 0.001, iter);
                                    loss = NaN;
                                case 'BCLS_ALM'
                                    base_dir = 'E:\f-means\siff-means\';
                                    tmp_dir = [base_dir 'BCLS_ALM/'];
                                    % Parameter setting
                                    gamma = 10^(-5);    % the value of Gamma should be between 0 and 10^(-10)
                                    lam = 10^(1);
                                    mu = 1;
                                    infRes = 0.90;      % the percentage of information reserved of the data during PCA dimension reduction
                                    save([tmp_dir 'param.mat']);

                                    initialization(X, c, tmp_dir, infRes);
                                    load([tmp_dir 'init.mat']);
                                    [d,n] = size(X);

                                    StartInd = randsrc(n,1,1:c);
                                    Y0 = TransformL(StartInd, c);
                                    save([tmp_dir 'Y0'], 'Y0');
                                    load([tmp_dir 'Y0']);

                                    [Y_label, ~, sse, balance_loss, elapsed_time, size0] = BCLS_ALM(X, Y0, iter, gamma, lam, mu);


                                    loss = NaN;
                                case 'F3KM'
                                    rho = 0.15;
                                    [Y_label, ~, iter_num, sse, obj, balance_loss, elapsed_time, size0] = F3KM(X, label, c, block_size, rho, threads, iter);
                                    loss = NaN;
                                case 'SIFF'
                                    [Y_label, ~, iter_num, sse, obj_max, balance_loss, elapsed_time, size0] = SIFF_eta(X, label, c, block_size, eta, iter);
                                    loss = NaN;
                            end

                            % get balance loss, cv and entro
%                             balance_loss = 0;
                            entro = 0;
                            cv = 0;
                            for jj = 1:c
%                                 balance_loss = balance_loss + (size0(jj) - l/c)^2;
                                entro = entro + size0(jj)/l * log(size0(jj)/l);
                                cv = cv + sqrt((size0(jj) - l/c)^2);
                            end
    
                            % data in the last iteration
                            SSEs(ite_run) = sse(end);
                            BALANCE_LOSSs(ite_run) = mean(balance_loss(end-4:end));
                            CVs(ite_run) = c/l * cv;
                            Nentros(ite_run) = - entro / log(c);
                            TIMEs(ite_run) = elapsed_time;

                            % data in each iteration
                            iter_SSEs(ite_run, :) = sse;
                            iter_BLs(ite_run, :) = balance_loss;
                        end

% 'dataset', 'k', 'method', ' MEAN_SSE', 'VAR_SSE', 'MEAN_BALANCE_LOSS', 'VAR_BALANCE_LOSS', 'MEAN_CV', 'MEAN_Nentro', 'MEAN_TIME'

                        % data in the last iteration
%                         fprintf(fid_time, '%s,%d,%s,', dataset_name, block_size, method_name);
%                         fprintf(fid_time, '%.5f,%.5f,', mean(SSEs), var(SSEs));
%                         fprintf(fid_time, '%.5f,%.5f,', mean(BALANCE_LOSSs), var(BALANCE_LOSSs));
%                         fprintf(fid_time, '%.5f,%.5f,%.5f\n', mean(CVs), mean(Nentros), mean(TIMEs));
                        
                        % data in each iteration
                        iter_SSEs = mean(iter_SSEs, 1);
                        iter_BLs = mean(iter_BLs, 1);
                        for ii = 1:iter
                            if ii ~= iter
                                fprintf(fid_time, '%.5f,', iter_SSEs(ii) + iter_BLs(ii));
                            else
                                fprintf(fid_time, '%.5f\n', iter_SSEs(ii) + iter_BLs(ii));
                            end
                        end
%                         fprintf(fid_time, '%s,%.4f,%.4f,%.4f,%.4f\n', dataset_name, eta, mean(SSEs)+mean(BALANCE_LOSSs), mean(BALANCE_LOSSs), mean(SSEs));
%                         fprintf('%s,%.4f,%.4f,%.4f,%.4f\n', dataset_name, eta, mean(SSEs)+mean(BALANCE_LOSSs), mean(BALANCE_LOSSs), mean(SSEs));


                        % Prepare the data for output
                        [~, data_name, ext] = fileparts(file_path);
                        file_name_with_ext = strcat(data_name, ext);
  
                        % 创建文件夹
                        if ~exist(method_name, 'dir') % 检查文件夹是否已存在
                            mkdir(method_name); % 创建文件夹
                        end

                    end
                end
            end
        end
    end
end
fclose(fid_time);






