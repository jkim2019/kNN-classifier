function labels = kmeans(X,test,linfun)
% ADDME given labeled training data X, predict classifications for test data
% uses 5-fold validation to determine optimal k
%   X is (m x n) matrix of labeled data
%   test is (m x n) test matrix of labeled data
%   linfun is an anonymous linear functional: scalar = linfun(vector)
    % [m,n] = size(X);

    % % divide X into 5 batches
    % b_size = ceil(m/5);
    % X_batches = mat2cell(X,[b_size*ones(1,4), m-4*b_size],n);

    % % interchange validation & training sets, determine accuracy across various k
    % K = [1,2,3];
    % batch_accuracies = zeros(numel(K),5);

    % for k=1:numel(K)
    %     for b=1:5
    %         % create validation set
    %         X_val = X_batches{b,1};
            
    %         % create training set as all batches that are not the validation set
    %         % train_batches = find(~(1:5 == b));
    %         X_train = vertcat(X_batches{1:5~=b,1});

    %         % divide image data and labels
    %         y_train = X_train(:,end);
    %         X_train = X_train(:,1:end-1);
    %         y_val = X_val(:,end);
    %         X_val = X_val(:,1:end-1);
    %         y_pred = zeros(size(y_val));
            
    %         % predict on validation set
    %         [m_val, ~] = size(X_val);
    %         for i=1:m_val
    %             % calculate distance between validation image and each image in X_train
    %             distances = cellfun(linfun, num2cell(X_train-X_val(i,:),2));
                
    %             % obtain indices of k smallest elements
    %             [~,indices] = mink(distances,K(k));

    %             % obtain classifications and determine most popular
    %             candidates = y_train(indices);
    %             candidate_votes = sum(candidates == candidates');
    %             [~, idx] = max(candidate_votes);
    %             y_pred(i) = candidates(idx);
    %         end
            
    %         batch_accuracies(k,b) = sum(y_pred == y_val) / numel(y_val);
    %     end
    % end
    
    % % plot batch accuracies, compute mean row-wise
    % k_accuracies = mean(batch_accuracies,2);
    % [~, idx] = max(k_accuracies);
    % opt_k = K(idx);
    % fprintf('optimal k: %d\n', opt_k);

    % acc_vec = batch_accuracies';
    % acc_vec = acc_vec(:);

    % figure; scatter(repelem(K,5),acc_vec); title('batch accuracies');

    opt_k = 1;

    % predict on test with opt_k
    % divide training & test sets
    y = X(:,end);
    X = X(:,1:end-1);
    
    y_test = test(:,end);
    X_test = test(:,1:end-1);

    [m_test, ~] = size(test);
    labels = zeros(m_test,1);
    idx_counts = zeros(m_test,1);
    for i=1:m_test
        % calculate distances between test image and each image in X_train
        distances = X - X_test(i,:);
        distances = linfun(distances');

        % obtain indices of k_opt smallest elements
        [~,indices] = mink(distances,opt_k);

        % obtain classifications and determine most popular
        candidates = y(indices);
        candidate_votes = sum(candidates == candidates');
        [~, idx] = max(candidate_votes);
        labels(i) = candidates(idx);
        
        % change to indices if using kNN
        idx_counts(indices) = idx_counts(indices) + 1;
    end
    figure; plot(idx_counts);

    % calculate accuracy
    acc = sum(labels == y_test) / numel(y_test);
    fprintf('accuracy: %2f\n', acc);
end