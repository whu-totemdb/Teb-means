
function [Y, minO, iter_num, obj, elapsed_time] = BANCDKM_Ji(X, label,c, rho)
fprintf("BANCDKM_Ji\n");
% Input
% X d*n data
% label is initial label n*1
% c is the number of clusters
% code for F. Nie, J. Xue, D. Wu, R. Wang, H. Li, and X. Li,
%¡°Coordinate descent method for k-means,¡± IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021
% Output
% Y is the label vector n*1
% minO is the Converged objective function value
% iter_num is the number of iteration
% obj is the objective function value
% It is written by Jingjing Xue

% 记录每次运行的开始时间
run_time = tic;

[~,n] = size(X);z    
F = sparse(1:n,label,1,n,c,n);  % transform label into indicator matrix
last = 0;
iter_num = 0;
% rho = 0.1;
%% compute Initial objective function value
for ii=1:c
    idxi = find(label==ii);
    Xi = X(:,idxi);
    ceni = mean(Xi,2);
    center(:,ii) = ceni;
    c2 = ceni'*ceni;
    d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
    sumd(ii,1) = sum(d2c);
end

% obj(1)= sum(sumd);    % Initial objective function value
%% store once
for i=1:n
    XX(i)=X(:,i)'* X(:,i);
end
BB = X*F;
aa=sum(F,1);% diag(F'*F) ;
FXXF=BB'*BB;% F'*X'*X*F;
% while any(label ~= last)

% Clusters = [580, 465, 755, 385, 625, 980, 410, 900, 450, 540, 760, 370, 425, 570, 805, 390, 720, 560, 395, 960]; 
% Clusters = [640, 525, 655, 505, 685, 920, 510, 820, 550, 480, 820, 345, 485, 530, 865, 450, 660, 620, 335, 980];
% Clusters = [1500, 45, 1780, 250, 1100, 25, 1600, 60, 1350, 75, 900, 35, 1850, 95, 500, 120, 1400, 85, 650, 80];
%Clusters = [1700, 50, 1550, 100, 1300, 200, 1200, 55, 1450, 150, 650, 75, 1900, 40, 800, 30, 1350, 90, 550, 60];
Clusters = [450, 350, 250, 480, 380, 280, 460, 360, 260, 450, 550, 650, 750, 520, 620, 720, 540, 640, 740, 550];
rho

% Clusters = [1100, 900, 800, 1200, 700, 1300, 950, 1050, 850, 1150];

for iter = 1:100
    % while any(label ~= last)
    last = label;

    
    local_limit = zeros(1, c);
    for i = 1:n
        m = label(i);
        if aa(m) == 1
            continue;
        end

        for k = 1:c
            if k == m
                V1(k) = FXXF(k, k) - 2 * X(:, i)' * BB(:, k) + XX(i);
                delta(k) = FXXF(k, k) / aa(k) - V1(k) / (aa(k) - 1) - 2*(500 / Clusters(k))*(aa(k) - Clusters(k))  + (500 / Clusters(k));
%                 delta(k) = FXXF(k, k) / aa(k) - V1(k) / (aa(k) - 1) - 2*rho*(aa(k)-n/c)  + rho;
            else
                V2(k) = FXXF(k, k) + 2 * X(:, i)' * BB(:, k) + XX(i);
                delta(k) = V2(k) / (aa(k) + 1) - FXXF(k, k) / aa(k) - 2*(500 / Clusters(k))*(aa(k) - Clusters(k))  - (500 / Clusters(k));
%                 delta(k) = V2(k) / (aa(k) + 1) - FXXF(k, k) / aa(k) - 2*rho*(aa(k)-n/c)  - rho;
            end
            local_limit(k) = (aa(k) - Clusters(k))^2;
        end

        [~, q] = max(delta);
        if m ~= q
            BB(:, q) = BB(:, q) + X(:, i); % Update BB for new cluster
            BB(:, m) = BB(:, m) - X(:, i); % Update BB for old cluster
            aa(q) = aa(q) + 1;
            aa(m) = aa(m) - 1;
            FXXF(m, m) = V1(m);
            FXXF(q, q) = V2(q);
            label(i) = q;
        end
    end
    iter_num = iter_num+1;

    %% compute objective function value
    for ii=1:c
        idxi = find(label==ii);
        Xi = X(:,idxi);
        ceni = mean(Xi,2);
        center1(:,ii) = ceni;
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
        sumd(ii,1) = sum(d2c)  + (500 / Clusters(ii)) * local_limit(ii);

    end
%     obj(iter_num+1) = sum(sumd) +  rho * sum(local_limit);    %  objective function value
     obj(iter_num) = sum(sumd) ;    %  objective function value
end

cluster_size = zeros(1, c);
for ii = 1:c
    cluster_size(ii) = sum(label == ii);
end
disp(cluster_size);
sum(cluster_size)
elapsed_time = toc(run_time);
disp(['Elapsed time: ', num2str(elapsed_time)]);
minO=min(obj);
Y=label;
end


