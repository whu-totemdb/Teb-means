
function [Y, minO, iter_num, sse, obj, balance_loss, runtime, cluster_size] = Federated_Teb(X, label, c, block_size, eta, max_iters)

% parpool("local",4);

start_time = tic;

[features_num,n] = size(X);
F = sparse(1:n,label,1,n,c,n);
iter_num = 0;

% rho = 0.20;
rho = (1-eta)/eta;

% set the number of clients;
clients_num = 2;
%% store once
% for i=1:n
%     XX(i) = X(:,i)'* X(:,i);
% end
% XF = X*F;
% FF = sum(F,1);    % diag(F'*F) ;
% FXXF = XF'*XF;    % F'*X'*X*F;

iter=0;
blocks = partitionNumbers(n, block_size); % partition blocks;

features_clients = partitionFeatures(features_num, clients_num); % partition features;


% store Xm, XmXm, XmF, FXmXmF for each client;
% FF will be updated globally;
Xm_cell = cell(1, clients_num);
XmXm_cell = cell(1, clients_num); 
XmF_cell = cell(1, clients_num);
FXmXmF_cell = cell(1, clients_num);
FF = sum(F,1);    % diag(F'*F) ;

for i = 1:clients_num
    Xm = X(features_clients{i}(1):features_clients{i}(end),:);  
    Xm_cell{i} = Xm;
    
    XmXm = zeros(1,n);
    for j = 1:n
        XmXm(j) = Xm(:,j)'* Xm(:,j);
    end
    XmXm_cell{i} = XmXm;
    
    XmF = Xm*F;
    XmF_cell{i} = XmF;
    
    FXmXmF = XmF'*XmF;
    FXmXmF_cell{i} = FXmXmF;
end


for iter_t = 1:max_iters

    iter = iter + 1;
    phi = zeros(1,c);
    phi1 = zeros(block_size,c, clients_num);
    phi1_t = zeros(block_size, c);
    
    v1_t = zeros(1,c);
    v2_t = zeros(1,c);
    V1_all = zeros(block_size, c, clients_num);
    V2_all = zeros(block_size, c, clients_num);
    
    %% Solve F
    for blockid = 1:length(blocks)
        block = blocks{blockid};
        m = label;   
        for idx = 1:length(block)
            i = block(idx);
            for k = 1:c
                for clientid = 1:clients_num
                    Xm = Xm_cell{clientid};
                    XmXm = XmXm_cell{clientid};
                    XmF = XmF_cell{clientid};
                    FXmXmF = FXmXmF_cell{clientid};
                    if k == m(i,:)
                        V1 = FXmXmF(k,k)- 2 * Xm(:,i)'* XmF(:,k) + XmXm(i);
                        U1 = V1/ (FF(k) -1) - FXmXmF(k,k) / FF(k);
                        phi1(idx, k, clientid) = U1 + (2 * rho * (FF(k) - n/c) - rho)/clients_num;
                        V1_all(idx, k, clientid) = V1;
                    else  
                        V2 =(FXmXmF(k,k) + 2 * Xm(:,i)'* XmF(:,k)+ XmXm(i));
                        U2 = FXmXmF(k,k)/ FF(k) - V2 / (FF(k) +1);
                        phi1(idx, k, clientid) = U2 + (2 * rho * (FF(k) - n/c) + rho)/clients_num;
                        V2_all(idx, k, clientid) = V2;
                    end
                end
                phi1_t(idx,k) = sum(phi1(idx,k,:));
            end
        end
        
        phi(block,:) = phi1_t(1:length(block),:);
        % fprintf('phi size: %d x %d\n', size(phi,1), size(phi,2));  % 添加
        % disp(['block ' num2str(blockid) ', iter ' num2str(iter_t) ...
        %      ', phi sum: ' num2str(sum(phi(block,:)),'%.4f')]);
        [~,label_update] = min(phi,[],2);
        q = find(m(1:block(end))~=label_update)';
        for j = q
            for clientid = 1:clients_num
                Xm = Xm_cell{clientid};
                XmF = XmF_cell{clientid};
                XmF(:,label_update(j))=XmF(:,label_update(j))+Xm(:,j);
                XmF(:,m(j))=XmF(:,m(j))-Xm(:,j);
                XmF_cell{clientid} = XmF;
                FXmXmF_cell{clientid} = XmF'*XmF;
            end
            FF(label_update(j))= FF(label_update(j)) +1; 
            FF(m(j))= FF(m(j)) -1;
        end
        
        label(1:block(end),:)=label_update;
    end
    
    %% compute objective function value
    for ii=1:c
        idxi = label==ii;
        Xi = X(:,idxi);     
        ceni = mean(Xi,2);   
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi; 
        sumd(ii) = sum(d2c);
        balance_loss_t(ii) = (FF(ii) - n/c)^2;
    end
    
    iter_num = iter_num + 1;
    sse(iter_t) = sum(sumd);
    balance_loss(iter_t) = sum(balance_loss_t);
    obj(iter_t) = eta*sse(iter_t) + (1-eta)*balance_loss(iter_t);
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
