

function D_m = client_compute_distances(X_m, centers_idx)

    centers_m = X_m(centers_idx, :);
    n_samples = size(X_m, 1);
    n_centers = length(centers_idx);
    D_m = zeros(n_samples, n_centers);

    for j = 1:n_centers
        diff = X_m - centers_m(j, :);
        D_m(:, j) = sum(diff.^2, 2);
    end
end
function new_center_idx = server_select_next_center(D_total, already_selected)

    min_distances = min(D_total, [], 2);


    min_distances(already_selected) = 0;


    prob = min_distances / sum(min_distances);

    edges = [0; cumsum(prob)];
    r = rand;
    new_center_idx = find(r >= edges(1:end-1) & r < edges(2:end), 1);
end
function centers_idx = federated_kmeans_pp_init(X_parts, k, random_state)

    rng(random_state);
    n_samples = size(X_parts{1}, 1);
    centers_idx = randi(n_samples); 

    for i = 2:k
        D_m_list = cellfun(@(X_m) client_compute_distances(X_m, centers_idx), X_parts, 'UniformOutput', false);
        D_total = zeros(n_samples, length(centers_idx));
        for m = 1:length(D_m_list)
            D_total = D_total + D_m_list{m};
        end

        new_center = server_select_next_center(D_total, centers_idx);
        centers_idx = [centers_idx, new_center];
    end
end
function labels = federated_assign_labels(X_parts, centers_idx)

    D_m_list = cellfun(@(X_m) client_compute_distances(X_m, centers_idx), X_parts, 'UniformOutput', false);
    D_total = zeros(size(D_m_list{1}));
    for m = 1:length(D_m_list)
        D_total = D_total + D_m_list{m};
    end
    [~, labels] = min(D_total, [], 2);
end

dataset_name = '1-Crime';  

if strcmp(dataset_name, '1-Crime') || strcmp(dataset_name, '2-MTG') || strcmp(dataset_name, '3-census1990') || ...
   strcmp(dataset_name, '4-Game') || strcmp(dataset_name, '5-NYC')
    file_path = strcat('dataset/output_final/', dataset_name, '.csv');
    X = csvread(file_path, 1, 1); 
    [dimension, l] = size(X);
    if l >= 5000
        X = X(:, 1:5000);
    end
    if dimension > 2000
        X = X(1:2000, :);
    end
    fprintf('稀疏矩阵维度: %d x %d\n', size(X, 1), size(X, 2));
else
    error('未知数据集名称');
end
n_samples = size(X,1)

n_features = size(X,2)

k = 4;


X1 = X(:, 1:64);     
X2 = X(:, 65:end);  


selected_centers_idx = federated_kmeans_pp_init({X1, X2}, k, 0);
disp("Selected center indices:");
disp(selected_centers_idx);


cluster_labels = federated_assign_labels({X1, X2}, selected_centers_idx);
disp("Cluster labels size:");
disp(size(cluster_labels));
disp("First 100 cluster labels:");
disp(cluster_labels(1:100));
function [X, labels] = make_blobs(n_samples, n_features, n_centers)
    centers = randn(n_centers, n_features) * 10;
    X = [];
    labels = [];
    samples_per_center = floor(n_samples / n_centers);
    for i = 1:n_centers
        Xi = bsxfun(@plus, randn(samples_per_center, n_features), centers(i, :));
        X = [X; Xi];
        labels = [labels; i * ones(samples_per_center, 1)];
    end

    if size(X, 1) < n_samples
        extra = n_samples - size(X, 1);
        Xi = bsxfun(@plus, randn(extra, n_features), centers(1, :));
        X = [X; Xi];
        labels = [labels; ones(extra, 1)];
    end
end
