fileNames = dir('dataset/output/');
result = zeros(length(fileNames), 5);
for i = 7:length(fileNames)
    dataName = fileNames(i).name;
    disp(dataName);
    dataTable = readtable(strcat('dataset/output/', dataName));
    new_fea = dataTable{:,1:5};
    % K = length(unique(new_gnd));
    K = 3;
    times = 1;
    temp_result = zeros(times, 5);
    for t = 1 : times
        t0 = cputime;
        index = FCFC(new_fea, K, 100);
        temp_result(t,5) = cputime - t0;
        counts = hist(index,1:K)

%         [VIn, VDn, Rn, NMI] = exMeasure(index, new_gnd);
%         counts = hist(index,1:K);
%         temp_result(i,1) = Rn;
%         temp_result(i,2) = NMI;
%         temp_result(i,3) = std(counts)/mean(counts);
%         temp = counts/sum(counts);
%         temp_result(i,4) = -1/(log(K)) * sum(temp.*log(temp));
    end
    
    result(i,:) = temp_result(find(temp_result(:,1) == max(temp_result(:,1)),1),:);
    save('result_BKM.mat','result')
    disp(i);
end