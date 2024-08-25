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

% Train the SVM model
svm_model = fitcsvm(training_complete_vars, training_last_vars, ...
    'KernelFunction', 'linear', ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);

% Predict using the SVM model for the training data
predicted_class_name_svm_training = predict(svm_model, training_complete_vars);

% Calculate training accuracy
accuracy_svm_training = (sum(predicted_class_name_svm_training == training_last_vars) / numel(training_last_vars)) * 100;
fprintf('Training Accuracy: %.2f%%\n', accuracy_svm_training);

% Display confusion chart for training data
figure;
confusionchart(training_last_vars, predicted_class_name_svm_training);
title('Confusion Matrix for Training Data');

% Predict using the SVM model for the testing data
predicted_class_name_svm_testing = predict(svm_model, testing_complete_vars);

% Calculate testing accuracy
accuracy_svm_testing = (sum(predicted_class_name_svm_testing == testing_last_vars) / numel(testing_last_vars)) * 100;
fprintf('Testing Accuracy: %.2f%%\n', accuracy_svm_testing);

% Display confusion chart for testing data
figure;
confusionchart(testing_last_vars, predicted_class_name_svm_testing);
title('Confusion Matrix for Testing Data');