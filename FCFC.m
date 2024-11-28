% Fast Clustering with Flexible Balance Constraints

function [label, pre_dis] = FCFC(data, K, lambda)
% Input
%   data: data matrix (n*d)
%   K: the number of clusters
%   lambda: the balanced constraint weight parameters
% Output
%   label: the final assignment label vector (n*1)
%   sumbest: the best objective function value

    [n, d] = size(data);
    maxIter = 50;

    % 1.initialize the centroid
    centroid = initialCentroid(data, K, n);
    size_cluster = ones(1, K);      % a 1*k vector that stores the size of each cluster
    pre_dis = inf;
    for i = 1:maxIter
        D = getDistance(data, centroid, K, n, d, size_cluster, lambda);
        [dis, idx] = min(D, [], 2);
        sum_dis = sum(dis);
        if abs(pre_dis - sum_dis) < 1e-5
            break;
        else
            label = idx;
            size_cluster = hist(label, 1:K);
            centroid = getCentroid(data, label, K, n, d);
            pre_dis(i) = sum_dis;
            disp(pre_dis);
        end
    end
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