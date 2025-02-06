
function [Y, minO, iter_num, sse, obj, balance_loss, runtime, cluster_size] = Teb(X, label, c, block_size, eta, max_iters)

% parpool("local",4);

start_time = tic;

[~,n] = size(X);
F = sparse(1:n,label,1,n,c,n);
iter_num = 0;

% rho = 0.20;
rho = (1-eta)/eta;

%% store once
for i=1:n
    XX(i) = X(:,i)'* X(:,i);
end
XF = X*F;
FF = sum(F,1);    % diag(F'*F) ;
FXXF = XF'*XF;    % F'*X'*X*F;

iter=0;
blocks = partitionNumbers(n, block_size);
for iter_t = 1:max_iters

    iter = iter + 1;
    phi = zeros(1,c);
    v1_t = zeros(1,c);
    v2_t = zeros(1,c);
    V1_all = zeros(length(blocks), c);
    V2_all = zeros(length(blocks), c);
%% Solve F
    for blockid = 1:length(blocks)
        block = blocks{blockid};
        m = label;   
        for idx = 1:length(block)
            i = block(idx);
            for k = 1:c
                if k == m(i,:)   
                    V1 = FXXF(k,k)- 2 * X(:,i)'* XF(:,k) + XX(i);
                    U1 = V1/ (FF(k) -1) - FXXF(k,k) / FF(k);

                    phi1(idx, k) = U1 + 2 * rho * (FF(k) - n/c) - rho;
                    V1_all(idx, k) = V1;  % Store V1
                else  
                    V2 =(FXXF(k,k)  + 2 * X(:,i)'* XF(:,k)+ XX(i));
                    U2 = FXXF(k,k)/ FF(k) -  V2 / (FF(k) +1);

                    phi1(idx, k) = U2 + 2 * rho * (FF(k) - n/c) + rho;
                    V2_all(idx, k) = V2;  % Store V2
                end 
            end
        end
        v1_t(block,:) = V1_all(1:length(block),:);
        v2_t(block,:) = V2_all(1:length(block),:);
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

%% compute objective function value
    for ii=1:c
        idxi = label==ii;
        Xi = X(:,idxi);     
        ceni = mean(Xi,2);   
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi; 
        sumd(ii) = sum(d2c);
%         balance_loss_t(ii) = rho * (FF(ii) - n/c)^2;
        balance_loss_t(ii) = (FF(ii) - n/c)^2;
    end
    iter_num = iter_num + 1;
    sse(iter_t) = sum(sumd);
    balance_loss(iter_t) = sum(balance_loss_t);
    obj(iter_t) = eta*sse(iter_t) + (1-eta)*balance_loss(iter_t);
%     obj(iter_t) = sse(iter_t)^2 + rho*sum(balance_loss_t);
    % fprintf('sse=%f, ',sse(iter_num)^2)
    % fprintf('block=%f\n',sum(balance_loss_t))
    fprintf('obj = %f, balance loss = %f\n', obj(iter_t), balance_loss(iter_t))
end

minO=obj(iter_num)^2;
Y=label;


runtime = toc(start_time);
for ii = 1:c
    cluster_size(ii) = sum(label == ii);
end
% fprintf('Teb runtime: %.4f seconds, sse: %.4f, balance loss: %.4f\n', runtime, sse(end), mean(balance_loss(end-4:end)));

% delete(gcp('nocreate'))
end
