function [Y, minO, iter_num, obj, balance_loss, runtime, size0] = Lloyd(X, label, k, max_iter)
% Input
%   X: data matrix (d*n)
%   label: the initial assignment label (n*1)
%   k: the number of clusters
%   max_iter: maximum number of iterations (optional, default: 100)
% Output
%   Y: the final assignment label vector (n*1)
%   minO: the objective function value when converged
%   iter_num: the number of iterations executed
%   obj: the objective function value in each iteration
%   runtime: the total runtime of ClassicKMeans


% fprintf("Lloyd\n");
start_time = tic;

[d, n] = size(X);

% Randomly initialize cluster centroids if no label is provided
if isempty(label)
    rng('default');  % for reproducibility
    centroids = X(:, randperm(n, k));
else
    centroids = zeros(d, k);
    for j = 1:k
        centroids(:, j) = mean(X(:, label == j), 2);
    end
end

% Initialize variables
Y = zeros(n, 1);      % cluster assignments
obj = zeros(max_iter, 1);
iter_num = 0;

for iter = 1:max_iter
    iter_num = iter;
    
    % Step 1: Assign each point to the nearest centroid
    for i = 1:n
        distances = sum((X(:, i) - centroids).^2, 1); % Euclidean distances
        [~, Y(i)] = min(distances);
    end

    % Step 2: Update centroids
    new_centroids = zeros(d, k);
    for j = 1:k
        cluster_points = X(:, Y == j);
        if ~isempty(cluster_points)
            new_centroids(:, j) = mean(cluster_points, 2);
        else
            % Handle empty cluster by reinitializing centroid
            new_centroids(:, j) = X(:, randi(n));
        end
    end

    % Check convergence
%     if all(centroids == new_centroids, 'all')
%         break;
%     end
    centroids = new_centroids;

    % Compute the objective function (sum of squared distances)
    current_obj = 0;
    for j = 1:k
        cluster_points = X(:, Y == j);
        [~, cluster_size] = size(cluster_points);
        current_obj = current_obj + sum(sum((cluster_points - centroids(:, j)).^2));
        balance_loss_t(j) = (cluster_size - n/k)^2;
    end
    obj(iter) = current_obj;
    balance_loss(iter) = sum(balance_loss_t);
end

% Finalize results
obj = obj(1:iter_num);  % remove unused entries in obj
runtime = toc(start_time);
minO = obj(end);        % final objective function value

fprintf('Lloyd runtime: %.4f seconds, sse: %.4f, balance loss: %.4f\n', runtime, minO, mean(balance_loss(end-4:end)));
size0 = histcounts(Y, 1:k+1);  % count points in each cluster
% disp('Cluster sizes:');
% disp(cluster_size);
end
