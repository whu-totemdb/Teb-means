% Clear workspace and command window
clear all
clc

% Define the methods to test % f3km
methods = { 'CDKM', 'DBANCDKM'};

% Define the datasets to use
datasets = {'athlete', 'bank', 'census', 'creditcard', 'diabetes', 'recruitment', 'Spanish', 'yelp_academic_dataset_business'};

clusters_sets = {4, 20, 40, 60, 80, 100};
% Number of runs for averaging
num_runs = 1;
% clusters_sets = {4};

% Parameters for methods that require them
rho_set = [0.5];
point_max = [10000];
threads_set = 3;
clusters_set = 20;
epsilon_set = [0.1,0.5,1,2,3,4,5,6,7,8,9,10];

for method_idx = 8:length(datasets)
    method_name = methods{method_idx};
    % Create the result file name
    result_file_name = ['result_', method_name, '.csv'];
    % Open the result file for appending
    fid = fopen(result_file_name, 'a');
  
    % Write the header if the file is new
    if ftell(fid) == 0
        fprintf(fid, 'data_name,Clusters num,Average time,iterations,objective function value,Time Std,Obj Std,num-thr,rho,Average clusters size\n');
    end

    % Loop over each dataset


    for iter_dataset = 1:length(datasets)
        dataset_name = datasets{iter_dataset};
        for threads_num = threads_set
            threads = threads_num; % Set the number of threads
            for iter_cluster = 1:length(clusters_sets)
                c = clusters_sets{iter_cluster};
                for point_idx = 1:length(point_max)
                    max_points = point_max(point_idx);
                    for rho_idx = 1:length(rho_set)
                        rho = rho_set(rho_idx);
                        for epsilon_idx = 1:length(epsilon_set)
                            epsilon = epsilon_set(epsilon_idx); 
                            % Construct the file path and load data
                            file_path = strcat('D:\BalanceKMeams\k-means\Code\Data\individually-fair-k-clustering-main\individually-fair-k-clustering-main\yelp\output\', dataset_name, '_', num2str(max_points), '_', num2str(c), '.csv');
                            X = csvread(file_path, 1, 1)';
    
                            % Initialize variables for averaging
                            avg_obj_max = 0;
                            avg_iter_num = 0;
                            avg_time = 0;
                            avg_cluster_sizes = zeros(1, c);
                            avg_loss = zeros(1, 100);
    
                            time_record = zeros(1, num_runs);
                            obj_record = zeros(1, num_runs);
                            loss_record = zeros(1, num_runs);
    
                            for ite_run = 1:num_runs
                                fprintf('Running %s on dataset %s, run %d\n', method_name, dataset_name, ite_run);
                                % Initialize labels
                                rng(42 + ite_run);
                                label = kmeans(X', c);
    
                                % Delete any existing parallel pool
                                delete(gcp('nocreate'));
    
                                % Call the method dynamically
                                switch method_name
                                    case 'CDKM'
                                        [Y_label, ~, iter_num, obj_max, elapsed_time] = CDKM(X, label, c);
                                        loss = NaN;
                                end
    
                                % Aggregate results
                                if isnan(iter_num)
                                    avg_obj_max = avg_obj_max + obj_max;
                                else
                                    avg_obj_max = avg_obj_max + obj_max(iter_num);
                                end
                                avg_iter_num = avg_iter_num + iter_num;
                                avg_time = avg_time + elapsed_time;
    
                                time_record(ite_run) = elapsed_time;
                                if isnan(iter_num)
                                    obj_record(ite_run) = obj_max;
                                else
                                    obj_record(ite_run) = obj_max(iter_num);
                                end
                                if ~isnan(loss)
                                    loss_record(ite_run) = loss(iter_num - 1);
                                end
    
                                % Compute cluster sizes
                                for ii = 1:c
                                    avg_cluster_sizes(ii) = avg_cluster_sizes(ii) + sum(Y_label == ii);
                                end
    
                                % Aggregate loss if applicable
                                if ~isnan(loss)
                                    for ii = 1:iter_num
                                        avg_loss(ii) = avg_loss(ii) + loss(ii);
                                    end
                                end
                            end
    
                            % Compute averages
                            avg_obj_max = avg_obj_max / num_runs;
                            avg_iter_num = avg_iter_num / num_runs;
                            avg_time = avg_time / num_runs;
                            avg_cluster_sizes = avg_cluster_sizes / num_runs;
                            avg_loss = avg_loss / num_runs;
    
                            % Compute standard deviations
                            standard_deviation_time = std(time_record);
                            standard_deviation_obj = std(obj_record);
    
                            % Prepare the data for output
                            [~, data_name, ext] = fileparts(file_path);
                            file_name_with_ext = strcat(data_name, ext);
    
                            % Handle parameters specific to certain methods
                            if ismember(method_name, {'DBANCDKM', 'DBANCDKM_DP'})
                                num_threads = threads;
                                rho_value = rho;
                            else
                                num_threads = NaN;
                                rho_value = NaN;
                            end
    
                            % Write the results to the file
                            fprintf(fid, '%s,%d,%.4f,%.2f,%.2f,%.2f,%.2f,', file_name_with_ext, c, avg_time, avg_iter_num, avg_obj_max, standard_deviation_time, standard_deviation_obj);
                            fprintf(fid, '%.2f,%.2f', num_threads, rho_value);
                            if method_idx == 5  || method_idx == 7
                                fprintf(fid, ',%f', epsilon);
                            end
                            for ii = 1:c
                                fprintf(fid, ',%.2f', avg_cluster_sizes(ii));
                            end
                            if ~isnan(loss)
                                for ii = 1:length(avg_loss)
                                    if avg_loss(ii) == 0
                                        break;
                                    end
                                    fprintf(fid, ',%.2f', avg_loss(ii));
                                end
                            end
                            fprintf(fid, '\n');
                         end 
                    end
                end
            end
        end
    end

    fclose(fid);
end






