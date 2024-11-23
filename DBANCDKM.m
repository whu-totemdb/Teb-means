% t小n（10）大比较差
function [Y, minO, iter_num, sse, obj, balance_loss, elapsed_time] = DBANCDKM(X, label,c, numWorkers, t,iter_rounds)

fprintf("DBANCDKM\n");


parpool('local', numWorkers);

loss_t = zeros(1, iter_rounds);
run_time = tic;
[~,n] = size(X);
F = sparse(1:n,label,1,n,c,n);  % transform label into indicator matrix

iter_num = 0;

for ii=1:c
        idxi = find(label==ii);
        Xi = X(:,idxi);     
        ceni = mean(Xi,2); 
        center(:,ii) = ceni;
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
        sumd(ii,1) = sum(d2c); 
end
rho = sum(sumd) /(t*(n / c));

partitionSize = ceil(n / numWorkers);

b = (n/c) * 0.001;
alpha = n/(c) - b;
beta = n/(c) + b;

limit_total_t = zeros(c, 1);
limit_total = zeros(c, 1);
thetas = (n / c) * ones(1, c);
lambdas = zeros(1, c);

for iter =1:iter_rounds
    % while any(label ~= last)
    thetas_t = thetas / numWorkers;
%     lambdas_t = lambdas / numWorkers;
%     thetas_t = thetas;
    lambdas_t = lambdas / numWorkers;%     
    spmd(numWorkers)
        penaltys = zeros(1, c);
        penaltys_t = zeros(1, c);

        local_n = partitionSize;
        if labindex == numWorkers
            local_n = n - partitionSize * (numWorkers - 1);
        end
        start = ( (labindex - 1) * partitionSize + 1);
        local_X = X(:, start: start + local_n - 1);
        local_F = F( start: start + local_n - 1 , :);

        local_BB = local_X * local_F;
        local_FXXF = local_BB' * local_BB;
        local_aa = full(sum(local_F,1));

        V1= zeros(1, c);
        V2= zeros(1, c);
        local_delta = zeros(1, c);
        local_XX = zeros(1, local_n);

%         

        for i=1:local_n
            local_XX(i)=local_X(:,i)'* local_X(:,i);
        end

        local_label = label(start:start + local_n - 1 ,:);

        for i = 1:local_n
            m = local_label(i) ;
            if local_aa(m)==1
                continue;
            end
            for k = 1:c
                if k == m
                    V1(k) = local_FXXF(k,k)- 2 * local_X(:,i)'* local_BB(:,k) + local_XX(i);
                    penaltys_t(k) =  lambdas_t(k) - rho * (2 * local_aa(k) + 1 - 2 * thetas_t(k));
                    local_delta(k) = local_FXXF(k,k) / local_aa(k) - V1(k) / (local_aa(k) -1)  + penaltys_t(k);
                    
                else
                    V2(k) =(local_FXXF(k,k)  + 2 * local_X(:,i)'* local_BB(:,k) + local_XX(i));

                    penaltys_t(k) =  lambdas_t(k) - rho * (2 * local_aa(k) - 1 - 2 * thetas_t(k));      
                    local_delta(k) = V2(k) / (local_aa(k) +1) -  local_FXXF(k,k)  / local_aa(k)  + penaltys_t(k);
                end
                

            end
            [~,q] = max(local_delta);
            if m~=q
                local_BB(:,q)=local_BB(:,q)+local_X(:,i); % local_BB(:,p)=local_X*local_F(:,p);
                local_BB(:,m)=local_BB(:,m)-local_X(:,i); % local_BB(:,m)=local_X*local_F(:,m);
                local_aa(q)= local_aa(q) + 1; %  FF(p,p)=F(:,p)'*local_F(:,p);
                local_aa(m)= local_aa(m) - 1; %  FF(m,m)=F(:,m)'*local_F(:,m)
                local_FXXF(m,m)=V1(m);
                local_FXXF(q,q)=V2(q);
                local_label(i)=q;
            end
        end

%         local_aa
    end

    for workerIdx = 1:numWorkers
        start_l = (workerIdx - 1) * partitionSize + 1;
        end_l = min(workerIdx * partitionSize, n);
        label(start_l:end_l, :) = local_label{workerIdx};
    end


%     lambdas_Workers
    F = sparse(1:n,label,1,n,c,n);
    aa=sum(F,1);% diag(F'*F) ;

    for k = 1:c
        thetas(k) = aa(k) - lambdas(k) / (2 * rho);
        thetas(k) = min(max(thetas(k), alpha), beta);
        lambdas(k) = lambdas(k) + rho * (thetas(k) - aa(k));
        
        limit_total(k) = lambdas(k) * (thetas(k)  - aa(k)) + rho * (thetas(k)  - aa(k))^2;

    end
    

    for ii=1:c
        idxi = find(label==ii);
        Xi = X(:,idxi);
        ceni = mean(Xi,2);
        %         center1(:,ii) = ceni;
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
        sumd(ii,1) = sum(d2c);
        
    end
    sse(iter) = sum(sumd) ;     %  objective function value
    rho = sse(iter) /(t*(n / c));
    balance_loss = sum(limit_total);
    obj(iter) =  sse(iter) + balance_loss;
    balance_loss = sum(limit_total);
    iter_num = iter_num + 1;
end


elapsed_time = toc(run_time);
disp(['Elapsed time: ', num2str(elapsed_time)]);
delete(gcp('nocreate'));
cluster_size = zeros(1, c);
for ii = 1:c
    cluster_size(ii) = sum(label == ii);
end

disp(cluster_size);
minO=min(obj);
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
