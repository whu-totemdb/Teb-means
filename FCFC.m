% Fast Clustering with Flexible Balance Constraints

function [label, minO, sse, balance_loss, runtime, size_cluster] = FCFC(data, init_label, K, lambda, max_iter)
% Input
%   data: data matrix (n*d)
%   K: the number of clusters
%   lambda: the balanced constraint weight parameters
% Output
%   label: the final assignment label vector (n*1)
%   sumbest: the best objective function value

    start_time = tic;
    [n, d] = size(data);
    maxIter = max_iter;
    sse = zeros(max_iter, 1);
    balance_loss = zeros(max_iter, 1);

    % 1.initialize the centroid
    for ii = 1:K
        Xi = data(init_label == ii, :);
        ceni = mean(Xi, 1);     % the i-th centroid (1*d)
        centroid(ii, :) = ceni; % the centroid matrix (k*d)
    end
    
%     centroid = initialCentroid(data, K, n);
    size_cluster = ones(1, K);      % a 1*k vector that stores the size of each cluster
    pre_dis = inf;
    for i = 1:maxIter
        D = getDistance(data, centroid, K, n, d, size_cluster, lambda);
        [dis, idx] = min(D, [], 2);
        sum_dis = sum(dis);

        label = idx;
        size_cluster = hist(label, 1:K);
        centroid = getCentroid(data, label, K, n, d);
        pre_dis(i) = sum_dis;
%         disp(pre_dis);

        % calculate the sse and the balance loss
%         if mod(i, 2) ~= 1
%             continue
%         end

        current_obj = 0;
        for j = 1:K
            cluster_points = data(label == j, :);
            current_obj = current_obj + sum(sum((cluster_points - centroid(j, :)).^2));
            balance_loss_t(j) = (size_cluster(j) - n/K)^2;
        end

        sse(i) = current_obj;
        balance_loss(i) = sum(balance_loss_t);
%         fprintf('iter: %d, sse: %.4f, balance loss: %.4f\n', i, sse(i), balance_loss(i));
%         sse((i+1)/2) = current_obj;
%         balance_loss((i+1)/2) = sum(balance_loss_t);
%         fprintf('iter: %d, sse: %.4f, balance loss: %.4f\n', i, sse((i+1)/2), balance_loss((i+1)/2));
    end
    
    runtime = toc(start_time);
%     disp(size_cluster);
    minO = sse(end);
    fprintf('FCFC runtime: %.4f seconds, sse: %.4f, balance loss: %.4f\n', runtime, minO, balance_loss(end));
end

%% initialize k centroids randomly
function centroid = initialCentroid(data, K, n)
    centroid = data(randsample(n, K), :);
end

%% update cnetroids after the assignment phase
function centroid = getCentroid(data, label, K, n, d)
    centroid = zeros(K, d);
    for k = 1:K
       members = (label == k);
       if any(members)
          centroid(k,:) = sum(data(members, :)) / sum(members);                     
       else
          centroid(k,:) = data(randsample(n,1), :);
       end
    end
end

%% objective function: D(i,j) = distance(i-th data point, j-th centroid) + size(j-th cluster)
function D = getDistance(data, centroid, K, n, d, size_cluster, lambda)
    D = zeros(n,K);
    for k = 1:K
       D(:,k) = sum((data - centroid(repmat(k,n,1), 1:d)).^2, 2) + lambda*size_cluster(k);
    end
end