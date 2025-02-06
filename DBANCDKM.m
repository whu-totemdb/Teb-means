% t小n（10）大比较差
function [Y, minO, iter_num, sse, obj, balance_loss, runtime] = DBANCDKM(X, label, c, numWorkers, t, iter_rounds)
% Input
%   X: data matrix (d*n)
%   label: the initial assignment label (n*1)
%   c: the number of clusters
%   numWorkers:
%   t:
%   iter_rounds: the maximum iteration number
% Output
%   Y: the final assignment label vector (n*1)
%   minO: the objective function value when converged
%   iter_num: the number of iteration
%   obj: the objective function value in each iteration
%   runtime: the total runtime of CDKM

fprintf("DBANCDKM\n");

parpool('local', numWorkers);

loss_t = zeros(1, iter_rounds);
start_time = tic;
[~,n] = size(X);
F = sparse(1:n,label,1,n,c,n);     % transform label into indicator matrix (n*k)

iter_num = 0;

%% compute initial objective function value
for ii = 1:c
    idxi = find(label==ii);
    Xi = X(:,idxi);
    m = size(Xi, 2);
    ceni = mean(Xi,2);
    center(:,ii) = ceni;
    c2 = ceni'*ceni;
    d2c = sum(Xi.^2) + m*c2 - 2*ceni'*Xi;
    sumd(ii,1) = sum(d2c); 
end

rho = sum(sumd) / (t*(n/c));
partitionSize = ceil(n / numWorkers);

b = (n/c) * 0.001;
alpha = n/(c) - b;
beta = n/(c) + b;

limit_total_t = zeros(c, 1);
limit_total = zeros(c, 1);
thetas = (n / c) * ones(1, c);
lambdas = zeros(1, c);

for iter = 1:iter_rounds
    thetas_t = thetas / numWorkers;
%     lambdas_t = lambdas / numWorkers;
%     thetas_t = thetas;
    lambdas_t = lambdas / numWorkers;

    % 1.calculate each thread separately
    spmd(numWorkers)
    %for labindex = 1:numWorkers
        penaltys = zeros(1, c);
        penaltys_t = zeros(1, c);
        
        % 1.1 set data number as local_n in current thread
        local_n = partitionSize;
        if labindex == numWorkers
            local_n = n - partitionSize * (numWorkers - 1);
        end

        % 1.2 get local_X (local data matrix) and local_F (local indicator matrix)
        start = ( (labindex - 1) * partitionSize + 1 );
        local_X = X(:, start: start + local_n - 1);
        local_F = F( start: start + local_n - 1 , :);

        % 1.3 get the local parameter for CDKM
        local_BB = local_X * local_F;
        local_FXXF = local_BB' * local_BB;
        local_aa = full(sum(local_F,1));

        V1 = zeros(1, c);
        V2 = zeros(1, c);
        local_delta = zeros(1, c);
        local_XX = zeros(1, local_n);

        for i = 1:local_n
            local_XX(i) = local_X(:,i)' * local_X(:,i);
        end

        local_label = label(start : start + local_n - 1 , :);

        % 1.4 assign each local data point to a cluster that minimizes obj
        for i = 1:local_n
            p = local_label(i);
            if local_aa(p) == 1
                continue;
            end
            for k = 1:c
                if k == p
                    V1(k) = local_FXXF(k,k) - 2 * local_X(:,i)' * local_BB(:,k) + local_XX(i);
%                     penaltys_t(k) =  lambdas_t(k) - rho * (2 * local_aa(k) + 1 - 2 * thetas_t(k));
                    penaltys_t(k) = 0;
                    local_delta(k) = local_FXXF(k,k) / local_aa(k) - V1(k) / (local_aa(k) -1) + penaltys_t(k);
                else
                    V2(k) =(local_FXXF(k,k)  + 2 * local_X(:,i)'* local_BB(:,k) + local_XX(i));
%                     penaltys_t(k) =  lambdas_t(k) - rho * (2 * local_aa(k) - 1 - 2 * thetas_t(k));      
                    penaltys_t(k) = 0;
                    local_delta(k) = V2(k) / (local_aa(k) +1) - local_FXXF(k,k)  / local_aa(k) + penaltys_t(k);
                end
            end
            [~,q] = max(local_delta);
            if p ~= q
                local_BB(:,q) = local_BB(:,q) + local_X(:,i); % local_BB(:,p)=local_X*local_F(:,p);
                local_BB(:,p) = local_BB(:,p) - local_X(:,i); % local_BB(:,m)=local_X*local_F(:,m);
                local_aa(q) = local_aa(q) + 1; %  FF(p,p)=F(:,p)'*local_F(:,p);
                local_aa(p) = local_aa(p) - 1; %  FF(m,m)=F(:,m)'*local_F(:,m)
                local_FXXF(p,p) = V1(p);
                local_FXXF(q,q) = V2(q);
                local_label(i) = q;
            end
        end
    end

    % 2.combine each assignment result to the global label
    for workerIdx = 1:numWorkers
        start_l = (workerIdx - 1) * partitionSize + 1;
        end_l = min(workerIdx * partitionSize, n);
        label(start_l : end_l, :) = local_label{workerIdx};
    end

    % 3.update the global indicator matrix
    F = sparse(1:n,label,1,n,c,n);
    aa=sum(F,1);% diag(F'*F) ;

    for k = 1:c
        thetas(k) = aa(k) - lambdas(k) / (2 * rho);
        thetas(k) = min(max(thetas(k), alpha), beta);
        lambdas(k) = lambdas(k) + rho * (thetas(k) - aa(k));
        
        limit_total(k) = lambdas(k) * (thetas(k)  - aa(k)) + rho * (thetas(k)  - aa(k))^2;
    end
    

    for ii = 1:c
        idxi = find(label == ii);
        Xi = X(:,idxi);
        m = size(Xi, 2);
        ceni = mean(Xi,2);
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + m*c2 - 2*ceni'*Xi;
        sumd(ii,1) = sum(d2c);
    end
    sse(iter) = sum(sumd) ;     %  objective function value
    rho = sse(iter) / (t*(n / c));
    balance_loss = sum(limit_total);
    obj(iter) =  sse(iter) + balance_loss;
    balance_loss = sum(limit_total);
    iter_num = iter_num + 1;
end


runtime = toc(start_time);
disp(['Elapsed time: ', num2str(runtime)]);
delete(gcp('nocreate'));
cluster_size = zeros(1, c);
for ii = 1:c
    cluster_size(ii) = sum(label == ii);
end
 sse(iter) 
disp(cluster_size);
minO=min(obj);
disp(minO);
Y=label;
loss = loss_t;

% 
% figure;
% plot(loss);
% drawnow; % 强制刷新图形
% file_name = sprintf('DBANCDKM_%.2f_obj.png', rho);
% saveas(gcf, file_name);
% 
% figure;
% plot(obj);
% drawnow; % 强制刷新图形
% file_name = sprintf('DBANCDKM_%.2f_sse.png', rho);
% saveas(gcf, file_name);


end
