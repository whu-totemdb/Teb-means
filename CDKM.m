% code for F. Nie, J. Xue, D. Wu, R. Wang, H. Li, and X. Li,
%¡°Coordinate descent method for k-means,¡± IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021


function [Y, minO, iter_num, obj, elapsed_time] = CDKM(X, label, k)
% Input
%   X: data matrix (d*n)
%   label: the initial assignment label (n*1)
%   k: the number of clusters
% Output
%   Y: the final assignment label vector (n*1)
%   minO: the objective function value when converged
%   iter_num: the number of iteration
%   obj: the objective function value

fprintf("CDKM\n");

start_time = tic;

[~,n] = size(X);
F = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix 
last = 0;
iter_num = 0;
%% compute Initial objective function value
for ii=1:k
        idxi = find(label==ii);
        Xi = X(:,idxi);     
        ceni = mean(Xi,2); 
        center(:,ii) = ceni;
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
        sumd(ii,1) = sum(d2c); 
end
 obj(1)= sum(sumd);    % Initial objective function value
%% store once
for i=1:n
    XX(i)=X(:,i)'* X(:,i);
end    
BB = X*F;
aa=sum(F,1);% diag(F'*F) ;
FXXF=BB'*BB;% F'*X'*X*F;



for iter =1:200
% while any(label ~= last)
    last = label;
 for i = 1:n   
     m = label(i) ;
    if aa(m)==1
        continue;  
    end 
    for k = 1:k        
        if k == m   
           V1(k) = FXXF(k,k)- 2 * X(:,i)'* BB(:,k) + XX(i);
           delta(k) = FXXF(k,k) / aa(k) - V1(k) / (aa(k) -1); 
        else  
           V2(k) =(FXXF(k,k)  + 2 * X(:,i)'* BB(:,k) + XX(i));
           delta(k) = V2(k) / (aa(k) +1) -  FXXF(k,k)  / aa(k); 
        end         
    end  
    [~,q] = max(delta);     
    if m~=q        
         BB(:,q)=BB(:,q)+X(:,i); % BB(:,p)=X*F(:,p);
         BB(:,m)=BB(:,m)-X(:,i); % BB(:,m)=X*F(:,m);
         aa(q)= aa(q) +1; %  FF(p,p)=F(:,p)'*F(:,p);
         aa(m)= aa(m) -1; %  FF(m,m)=F(:,m)'*F(:,m)
         FXXF(m,m)=V1(m); 
         FXXF(q,q)=V2(q);
         label(i)=q;
    end
 end 
    iter_num = iter_num+1;
    %% compute objective function value
    for ii=1:k
        idxi = find(label==ii);
        Xi = X(:,idxi);
        ceni = mean(Xi,2);
        center1(:,ii) = ceni;
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
        sumd(ii,1) = sum(d2c);
    end
    obj(iter_num+1) = sum(sumd) ;     %  objective function value
end
elapsed_time = toc(start_time);

disp(['Elapsed time: ', num2str(elapsed_time)]);
minO=min(obj);
Y=label;
end
