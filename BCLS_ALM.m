%
function [ID, Y, SSE, BalanceLoss, runtime, clusterSizes] = BCLS_ALM(X, Y, ITER, gamma, lam, mu)
% BCLS_ALM
% min_Y,W,b ||X'W+1b'-Y||^2 + gamma*||W||^2 + lam*Tr(Z'11'Z) + mu/2*||Y-Z + 1/mu*Lambda||^2
% INPUT:
% X: data matrix (d by n), already processed by PCA with 80%~90% information preserved
% Y: randomly initialized label matrix (n by c)
% Parameters: gamma and lam are the parameters respectively corresponding to Eq.(13) in the paper
% OUTPUT:
% ID: indicator vector (n by 1)
% Y: generated label matrix (b by c)


start_time = tic;

% ITER = 1200;
[dim, n] = size(X);

% H = eye(n) - 1/n*ones(n);
% X = X*H;
meanX = mean(X);

% 中心化矩阵
X = X - meanX;


c = size(Y,2);   % number of clusters
Lambda = zeros(n,c);
rho = 1.005;
P = eye(dim)/(X*X'+gamma*eye(dim));

Obj2 = zeros(ITER, 1);

for iter = 1:ITER

%     display(['Solving alternatively...',num2str(iter)]);

    % Solve W and b
    W = P*(X*Y);
    b = mean(Y)';
    E = X'*W + ones(n,1)*b' - Y;

    % Solve Z
    %     Z = (mu*eye(n)+2*lam*ones(n))\(mu*Y + Lambda);   % original solution - O(n^3)
        Z = (-2*lam*ones(n)+(mu+2*n*lam)*eye(n))/(mu^2+2*n*lam*mu)*(mu*Y+Lambda);  % new solution - O(n^2)
    % 使用分块或逐元素计算来避免大规模矩阵
    %     Z_diag = (-2*lam + (mu + 2*n*lam)) / (mu^2 + 2*n*lam*mu); % 计算对角线元素的系数
    %     Z = Z_diag * (mu*Y + Lambda); % 逐元素操作，而不是构建完整的矩阵

%     % 计算常量部分
%     scalar1 = -2 * lam;
%     scalar2 = (mu + 2 * n * lam);
% 
%     % 计算分母
%     denom = mu^2 + 2 * n * lam * mu;
% 
%     % 初始化结果矩阵 Z
%     Z = zeros(size(Y)); % Z 的大小与 Y 相同
% 
%     % 逐元素计算 Z
%     for i = 1:n
%         Z(i, :) = (scalar1 + scalar2) / denom * (mu * Y(i, :) + Lambda(i, :));
%     end


    % Solve Y
    V = 1/(2+mu)*(2*X'*W + 2*ones(n,1)*b' + mu*Z - Lambda);
    [~, ind] = max(V,[],2);
    Y = zeros(n,c);
    Y((1:n)' + n*(ind-1)) = 1;

    % Update Lambda and mu according to ALM
    Lambda = Lambda + mu*(Y-Z);
    mu = min(mu*rho, 10^5);

    % Objective value
    sum_Y = sum(Y(:)); % 计算矩阵 Y 所有元素的和
    Obj(iter) = trace(E'*E) + gamma*trace(W'*W) + lam*(sum_Y^2);

    % SSE Calculation
    clusterCenters = (X * Y) ./ sum(Y, 1); % c * d (each row is a cluster center)
    SSE(iter) = 0;
    for i = 1:n
        clusterIdx = ind(i);
        SSE(iter) = SSE(iter) + norm(X(:, i) - clusterCenters(:, clusterIdx))^2;
    end
    
    % Balance Loss Calculation
    clusterSizes = sum(Y, 1);
    idealSize = n / c;
    BalanceLoss(iter) = sum((clusterSizes - idealSize).^2);
%     fprintf('iter: %d, sse: %.4f, balance loss: %.4f\n', iter, SSE(iter), BalanceLoss(iter));

end

[~,ID] = max(Y,[],2);
% plot(Obj);

runtime = toc(start_time);
minO = SSE(end);
fprintf('BCLS runtime: %.4f seconds, sse: %.4f, balance loss: %.4f\n', runtime, minO, BalanceLoss(end));
% disp(clusterSizes);
end



