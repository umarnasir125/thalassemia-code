% Extract the feature columns and the label column
complete_vars = Betathalassemiamain(:, 4:14);
last_vars = Betathalassemiamain(:, 15);

% Convert to arrays if needed
if istable(complete_vars)
    complete_vars = table2array(complete_vars);
end

if istable(last_vars)
    last_vars = table2array(last_vars);
end

% Split the data into training (70%) and testing (30%)
cv = cvpartition(size(complete_vars, 1), 'HoldOut', 0.3);
idx = cv.test;

% Separate to training and testing data
training_complete_vars = complete_vars(~idx, :);
training_last_vars = last_vars(~idx, :);
testing_complete_vars = complete_vars(idx, :);
testing_last_vars = last_vars(idx, :);

% Train the k-NN model
knn_model = fitcknn(training_complete_vars, training_last_vars, ...
    'NumNeighbors', 2, ...
    'NSMethod', 'exhaustive', ...
    'Distance', 'minkowski', ...
    'Standardize', 1);

% Predict using the k-NN model for the training data
predicted_class_nameknn_training = predict(knn_model, training_complete_vars);

% Calculate training accuracy
accuracyknn_training = (sum(predicted_class_nameknn_training == training_last_vars) / numel(training_last_vars)) * 100;
fprintf('Training Accuracy: %.2f%%\n', accuracyknn_training);

% Display confusion chart for training data
figure;
confusionchart(training_last_vars, predicted_class_nameknn_training);
title('Confusion Matrix for Training Data');

% Predict using the k-NN model for the testing data
predicted_class_nameknn_testing = predict(knn_model, testing_complete_vars);

% Calculate testing accuracy
accuracyknn_testing = (sum(predicted_class_nameknn_testing == testing_last_vars) / numel(testing_last_vars)) * 100;
fprintf('Testing Accuracy: %.2f%%\n', accuracyknn_testing);

% Display confusion chart for testing data
figure;
confusionchart(testing_last_vars, predicted_class_nameknn_testing);
title('Confusion Matrix for Testing Data');
