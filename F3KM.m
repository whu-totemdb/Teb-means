
function [Y, minO, iter_num, sse, obj, balance_loss, elapsed_time, cluster_size] = F3KM(X, label,c, block_size, rho, numWorkers, max_iters)


% parpool("local",numWorkers);
run_time = tic;
[~,n] = size(X);
F = sparse(1:n,label,1,n,c,n);
iter_num = 0;
for i=1:n
    XX(i)=X(:,i)'* X(:,i);
end
XF = X*F;
FF=sum(F,1);    % diag(F'*F) ;
FXXF=XF'*XF;    % F'*X'*X*F;

blocks = partitionNumbers(n,block_size);
par_time_t = zeros(max_iters, length(blocks));

b = (n/c) * 0.003;
alpha = n/(c) - b;
beta = n/(c) + b;
limit_total = zeros(c, 1);
thetas = (n / c) * ones(1, c);
lambdas = zeros(1, c);


for iter_t = 1:max_iters
    iter_num = iter_num+1;
    phi=zeros(1,c);
    v1_t=zeros(1,c);
    v2_t=zeros(1,c);
    V1_all = zeros(length(blocks), c);
    V2_all = zeros(length(blocks), c);
    %% Solve F
    for blockid = 1:length(blocks)
        block = blocks{blockid};
        m = label;
%         elapsed_time_1 = toc(run_time_1);
        %         parfor idx = 1:length(block)

        par_time = tic;
        for idx = 1:length(block)    
            i = block(idx);
            for k = 1:c
                if k == m(i,:)
                    V1 = FXXF(k,k)- 2 * X(:,i)'* XF(:,k)+ XX(i);
                    U1 = V1/ (FF(k) -1) - FXXF(k,k) / FF(k);

                    phi1(idx, k) = U1 - (lambdas(k) - rho * (2 * FF(k) + 1 - 2 * thetas(k)));
                    V1_all(idx, k) = V1;  % Store V1
                else
                    V2 =(FXXF(k,k)  + 2 * X(:,i)'* XF(:,k)+ XX(i));
                    U2 = FXXF(k,k)/ FF(k) -  V2 / (FF(k) +1);

                    phi1(idx, k) = U2 - (lambdas(k) - rho * (2 * FF(k) - 1 - 2 * thetas(k)));
                    V2_all(idx, k) = V2;  % Store V2
                end
            end
        end
        
        par_time_t(iter_t,blockid) = toc(par_time);

        v1_t(block,:)=V1_all(1:length(block),:);
        v2_t(block,:)=V2_all(1:length(block),:);
        phi(block,:) = phi1(1:length(block),:);
        [~,label_update] = min(phi,[],2);
        q = find(m(1:block(end))~=label_update)';
        for j = q
            XF(:,label_update(j))=XF(:,label_update(j))+X(:,j);
            XF(:,m(j))=XF(:,m(j))-X(:,j);
            FF(label_update(j))= FF(label_update(j)) +1;
            FF(m(j))= FF(m(j)) -1;
            %              FXXF(m(j), m(j)) = v1_t(block, m(j));
            %              FXXF(label_update(j), label_update(j)) = V2_all(block, label_update(j));
        end
        label(1:block(end),:)=label_update;
        FXXF=XF'*XF;


        %         F = sparse(1:n,label,1,n,c,n);
    end
    for k = 1:c
        thetas(k) = FF(k) - lambdas(k) / (2 * rho);
        thetas(k) = min(max(thetas(k), alpha), beta);
        lambdas(k) = lambdas(k) + rho * (thetas(k) - FF(k));
    end
    
    %% compute objective function value
    for ii=1:c
        idxi = label==ii;
        Xi = X(:,idxi);
        m = size(Xi, 2);
        ceni = mean(Xi,2);
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
        sumd(ii,1) = sum(d2c);
%         balance_loss_t(ii) = lambdas(k) * (thetas(k) - FF(k)) + rho * (thetas(k) - FF(k))^2;

        cluster_size(ii) = sum(label == ii);
        balance_loss_t(ii) = (cluster_size(ii) - n/c)^2;
    end
    sse(iter_num) = sum(sumd);
    %     rho = sse(iter) / (t*(n / c));
    balance_loss(iter_num) = sum(balance_loss_t);
    obj(iter_num) = sse(iter_num) + sum(balance_loss_t);
%     fprintf('obj=%f\n',obj(iter_num))

end

minO=obj(iter_num);
Y=label;
elapsed_time = toc(run_time);
% row_max_time = max(par_time_t, [], 2);  % max(matrix, [], 2) 返回每一行的最大值
% sum_of_max_time = sum(row_max_time);
% total_sum = sum(sum(par_time_t));
% elapsed_time = elapsed_time - total_sum + sum_of_max_time;

fprintf('F3KM runtime: %.4f seconds, sse: %.4f, balance loss: %.4f\n', elapsed_time, sse(end), mean(balance_loss(end-4:end)));

% disp(['Elapsed time: ', num2str(elapsed_time), ' seconds']);
% delete(gcp('nocreate'))

% cluster_size = zeros(1, c);
% for ii = 1:c
%     cluster_size(ii) = sum(label == ii);
% end

% disp(cluster_size);
minO=min(obj);
Y=label;


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
