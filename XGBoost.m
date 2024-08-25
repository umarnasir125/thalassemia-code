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

% Ensure the response variable is categorical
if ~islogical(last_vars) && ~isnumeric(last_vars)
    last_vars = categorical(last_vars);
end

% Split the data into training (70%) and testing (30%)
cv = cvpartition(size(complete_vars, 1), 'HoldOut', 0.3);
idx = cv.test;

% Separate to training and testing data
training_complete_vars = complete_vars(~idx, :);
training_last_vars = last_vars(~idx, :);
testing_complete_vars = complete_vars(idx, :);
testing_last_vars = last_vars(idx, :);

% Train the Gradient Boosting Machine model using fitcensemble
gbm_model = fitcensemble(training_complete_vars, training_last_vars, ...
    'Method', 'LogitBoost', ...
    'NumLearningCycles', 100, ...
    'Learners', 'Tree');  % Using default decision tree

% Predict using the GBM model for the training data
predicted_class_name_training = predict(gbm_model, training_complete_vars);

% Calculate training accuracy
accuracy_training = (sum(predicted_class_name_training == training_last_vars) / numel(training_last_vars)) * 100;
fprintf('Training Accuracy: %.2f%%\n', accuracy_training);

% Display confusion chart for training data
confusionchart(training_last_vars, predicted_class_name_training);
title('Confusion Matrix for Training Data');

% Predict using the GBM model for the testing data
predicted_class_name_testing = predict(gbm_model, testing_complete_vars);

% Calculate testing accuracy
accuracy_testing = (sum(predicted_class_name_testing == testing_last_vars) / numel(testing_last_vars)) * 100;
fprintf('Testing Accuracy: %.2f%%\n', accuracy_testing);

% Display confusion chart for testing data
confusionchart(testing_last_vars, predicted_class_name_testing);
title('Confusion Matrix for Testing Data');