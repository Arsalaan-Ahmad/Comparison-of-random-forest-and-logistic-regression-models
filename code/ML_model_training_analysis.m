%%loading the data from CSV file
data=readtable("diabetes.csv");

%data Overview
disp(head(data));

% Extract Column Headings
columnNames = data.Properties.VariableNames;

% Display the Column Headings
disp(columnNames);


% Display Table Information
disp('Table Information:');
disp(['Number of Rows: ', num2str(height(data))]);
disp(['Number of Columns: ', num2str(width(data))]);

% Creating a table with equal-length columns
columnNames = data.Properties.VariableNames';
dataTypes = varfun(@class, data, 'OutputFormat', 'cell')';

% Display Column Names and Data Types
infoTable = table(columnNames, dataTypes, 'VariableNames', {'ColumnName', 'DataType'});
disp(infoTable);

  % Compute Descriptive Statistics
    summaryStats = table;
    summaryStats.Mean = mean(data{:,:})';     % Mean of each column
    summaryStats.Median = median(data{:,:})'; % Median of each column
    summaryStats.Min = min(data{:,:})';       % Minimum of each column
    summaryStats.Max = max(data{:,:})';       % Maximum of each column
    summaryStats.Std = std(data{:,:})';       % Standard deviation of each column

    % Assign Column Names
    summaryStats.Properties.RowNames = data.Properties.VariableNames;

    % Display Summary Table
    disp(summaryStats);

    % Count Missing (NaN) Values
missingCounts = sum(ismissing(data));

% Display Missing Counts as a Table
missingTable = table(data.Properties.VariableNames', missingCounts', ...
    'VariableNames', {'ColumnName', 'MissingValues'});

disp(missingTable);

%% Replace 0 values with NaN:(because  In this dataset, missing values are encoded as 0.)

% Create a deep copy
data_copy = data;

% Replace zeros with NaN in specific columns
columnsToReplace = {'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'};

for i = 1:numel(columnsToReplace)
    colName = columnsToReplace{i};
    data_copy.(colName)(data_copy.(colName) == 0) = NaN;
end

% Count and Display Missing Values
missingCounts = sum(ismissing(data_copy));

% Display Missing Counts as a Table
missingTable = table(data_copy.Properties.VariableNames', missingCounts', ...
    'VariableNames', {'ColumnName', 'MissingValues'});

disp(missingTable);

%% Create Histograms for Each Column
figure('Units', 'normalized', 'Position', [0 0 0.8 0.8]); % Full-screen figure

numCols = width(data); % Number of columns

for i = 1:numCols
    subplot(ceil(sqrt(numCols)), ceil(sqrt(numCols)), i);
    histogram(data{:, i}); % Plot histogram
    title(columnNames{i}, 'Interpreter', 'none'); % Add column title
    xlabel('Values');
    ylabel('Frequency');
end

%% Fill missing values for each column
data_copy.Glucose = fillmissing(data_copy.Glucose, 'constant', mean(data_copy.Glucose, 'omitnan')); % filling in mean values
data_copy.BloodPressure = fillmissing(data_copy.BloodPressure, 'constant', mean(data_copy.BloodPressure, 'omitnan'));  % filling in mean values
data_copy.SkinThickness = fillmissing(data_copy.SkinThickness, 'constant', median(data_copy.SkinThickness, 'omitnan'));  % filling in median values
data_copy.Insulin = fillmissing(data_copy.Insulin, 'constant', median(data_copy.Insulin, 'omitnan')); % filling in median values
data_copy.BMI = fillmissing(data_copy.BMI, 'constant', median(data_copy.BMI, 'omitnan')); % filling in median values

% Display the updated table
disp(head(data_copy));

%% Create Histograms for Each Column
figure('Units', 'normalized', 'Position', [0 0 0.8 0.8]); % Full-screen figure

numCols = width(data_copy); % Number of columns

for i = 1:numCols
    subplot(ceil(sqrt(numCols)), ceil(sqrt(numCols)), i);
    histogram(data_copy{:, i}); % Plot histogram
    title(columnNames{i}, 'Interpreter', 'none'); % Add column title
    xlabel('Values');
    ylabel('Frequency');
end

%% Count and Display Missing Values
missingCounts = sum(ismissing(data_copy));

% Display Missing Counts as a Table
missingTable = table(data_copy.Properties.VariableNames', missingCounts', ...
    'VariableNames', {'ColumnName', 'MissingValues'});

disp(missingTable);

%% Count the occurrences of 0 and 1 in the 'Output' column
outputCounts = countcats(categorical(data.Outcome));

% Create a bar plot for the 'Output' column with teal color
figure;
bar([0 1], outputCounts, 'FaceColor', [0, 99/255, 128/255], 'FaceAlpha', 0.7); % Teal color
xlabel('Output Values');
ylabel('Count');
title('Distribution of Output (0 and 1)');
xticks([0 1]);
xticklabels({'0', '1'}); % Label the x-axis with 0 and 1

%%  Correlation Heatmap with numbers

% Ensure all data columns are numeric
numericData = table2array(data_copy);

corrMatrix = corr(numericData, 'Rows', 'complete');
figure;
heatmapHandle = heatmap(columnNames, columnNames, corrMatrix, ...
    'Colormap', jet, ...
    'ColorbarVisible', 'on');
heatmapHandle.Title = 'Correlation Heatmap';
heatmapHandle.CellLabelFormat = '%.2f'; % Display numbers with 2 decimal places

%% Normalize the predictor using min max normalization

% Separate predictors and target variable
X = data_copy{:, 1:8};
y = data_copy{:, 9};

% Normalize the predictor variables between 0 and 1
X = (X - min(X)) ./ (max(X) - min(X));

%% Split the data into training, validation, and testing sets (60% train, 20% validation, 20% test)

% Define the split ratios
trainRatio = 0.6;
valRatio = 0.2;
testRatio = 0.2;

% Shuffle the data
numRows = size(X, 1);
randomIdx = randperm(numRows);

% Determine the indices for each split
trainSize = round(trainRatio * numRows);
valSize = round(valRatio * numRows);

trainIdx = randomIdx(1:trainSize);
valIdx = randomIdx(trainSize+1:trainSize+valSize);
testIdx = randomIdx(trainSize+valSize+1:end);

% Split the data into training, validation, and test sets
XTrain = X(trainIdx, :);
yTrain = y(trainIdx);
XVal = X(valIdx, :);
yVal = y(valIdx);
XTest = X(testIdx, :);
yTest = y(testIdx);

% Display the sizes of each set
fprintf('Training Set Size: %d\n', size(XTrain, 1));
fprintf('Validation Set Size: %d\n', size(XVal, 1));
fprintf('Test Set Size: %d\n', size(XTest, 1));

%% Initialize models

% Create and train the logistic regression model
logisticModel = fitglm(XTrain, yTrain, 'Distribution', 'binomial', 'Link', 'logit');

% Create and train the Random Forest model
numTrees = 200; % Number of estimators (trees)
randomForestModel = TreeBagger(numTrees, XTrain, yTrain, 'Method', 'classification', 'OOBPrediction', 'on','OOBPredictorImportance','on');

%% Predict and evaluate logistic regression model
yPredLogisticTrain = round(predict(logisticModel, XTrain));
yPredLogisticVal = round(predict(logisticModel, XVal));
yPredLogisticTest = round(predict(logisticModel, XTest));

logisticTrainAcc = sum(yPredLogisticTrain == yTrain) / length(yTrain);
logisticValAcc = sum(yPredLogisticVal == yVal) / length(yVal);
logisticTestAcc = sum(yPredLogisticTest == yTest) / length(yTest);

% Calculate precision, recall, F1 score, and AUC for logistic regression
[cmLogistic, ~] = confusionmat(yTest, yPredLogisticTest);
logisticPrecision = cmLogistic(2,2) / sum(cmLogistic(:,2));
logisticRecall = cmLogistic(2,2) / sum(cmLogistic(2,:));
logisticF1 = 2 * (logisticPrecision * logisticRecall) / (logisticPrecision + logisticRecall);
[~, ~, ~, logisticAUC] = perfcurve(yTest, predict(logisticModel, XTest), 1);

% Predict and evaluate random forest model
[yPredRFTrain, ~] = predict(randomForestModel, XTrain);
[yPredRFVal, ~] = predict(randomForestModel, XVal);
[yPredRFTest, scoresRFTest] = predict(randomForestModel, XTest);

yPredRFTrain = str2double(yPredRFTrain);
yPredRFVal = str2double(yPredRFVal);
yPredRFTest = str2double(yPredRFTest);

rfTrainAcc = sum(yPredRFTrain == yTrain) / length(yTrain);
rfValAcc = sum(yPredRFVal == yVal) / length(yVal);
rfTestAcc = sum(yPredRFTest == yTest) / length(yTest);

% Calculate precision, recall, F1 score, and AUC for random forest
[cmRF, ~] = confusionmat(yTest, yPredRFTest);
rfPrecision = cmRF(2,2) / sum(cmRF(:,2));
rfRecall = cmRF(2,2) / sum(cmRF(2,:));
rfF1 = 2 * (rfPrecision * rfRecall) / (rfPrecision + rfRecall);
[~, ~, ~, rfAUC] = perfcurve(yTest, scoresRFTest(:,2), 1);

%% Display results
fprintf('(1) Initial Model performance comparison before hyperparameter tuning and cross validation\n');
fprintf('Logistic Regression Training Accuracy: %.4f\n', logisticTrainAcc);
fprintf('Logistic Regression Validation Accuracy: %.4f\n', logisticValAcc);
fprintf('Logistic Regression Testing Accuracy: %.4f\n', logisticTestAcc);
fprintf('Logistic Regression Precision: %.4f\n', logisticPrecision);
fprintf('Logistic Regression Recall: %.4f\n', logisticRecall);
fprintf('Logistic Regression F1 Score: %.4f\n', logisticF1);
fprintf('Logistic Regression AUC: %.4f\n', logisticAUC);

fprintf('Random Forest Training Accuracy: %.4f\n', rfTrainAcc);
fprintf('Random Forest Validation Accuracy: %.4f\n', rfValAcc);
fprintf('Random Forest Testing Accuracy: %.4f\n', rfTestAcc);
fprintf('Random Forest Precision: %.4f\n', rfPrecision);
fprintf('Random Forest Recall: %.4f\n', rfRecall);
fprintf('Random Forest F1 Score: %.4f\n', rfF1);
fprintf('Random Forest AUC: %.4f\n', rfAUC);

%% Confusion matrices with additional matrix on the right side showing percentages
function plotConfusionMatrixWithPercentages(cm)
    % Create confusion matrix heatmap for Logistic Regression
    figure;
    subplot(1, 2, 1);
    heatmap(cm, 'XLabel', 'Predicted', 'YLabel', 'Actual', ...
        'XData', {'Non-Diabetic', 'Diabetic'}, 'YData', {'Non-Diabetic', 'Diabetic'}, ...
        'Colormap', parula, 'ColorbarVisible', 'off');
    title('LogR Confusion Matrix');
    
    % Calculate and plot the percentage confusion matrix
    percentagesCM = cm ./ sum(cm, 2) * 100;
    subplot(1, 2, 2);
    heatmap(percentagesCM, 'XLabel', 'Predicted', 'YLabel', 'Actual', ...
        'XData', {'Non-Diabetic', 'Diabetic'}, 'YData', {'Non-Diabetic', 'Diabetic'}, ...
        'Colormap', parula, 'ColorbarVisible', 'off');
    title('Percentages');
end

function plotConfusionMatrixWithPercentages1(cm)
    % Create confusion matrix heatmap for Random Forest
    figure;
    subplot(1, 2, 1);
    heatmap(cm, 'XLabel', 'Predicted', 'YLabel', 'Actual', ...
        'XData', {'Non-Diabetic', 'Diabetic'}, 'YData', {'Non-Diabetic', 'Diabetic'}, ...
        'Colormap', parula, 'ColorbarVisible', 'off');
    title('RF Confusion Matrix');
    
    % Calculate and plot the percentage confusion matrix
    percentagesCM = cm ./ sum(cm, 2) * 100;
    subplot(1, 2, 2);
    heatmap(percentagesCM, 'XLabel', 'Predicted', 'YLabel', 'Actual', ...
        'XData', {'Non-Diabetic', 'Diabetic'}, 'YData', {'Non-Diabetic', 'Diabetic'}, ...
        'Colormap', parula, 'ColorbarVisible', 'off');
    title('Percentages');
end


plotConfusionMatrixWithPercentages(cmLogistic);
plotConfusionMatrixWithPercentages1(cmRF);

%% features importance for RF 

% Get feature importances (OOB Permuted Predictor Delta Error)
featureImportances = randomForestModel.OOBPermutedPredictorDeltaError;

% Display feature importances
disp('Feature Importances:');
disp(featureImportances);

% Create a bar chart
figure;
bar(featureImportances, 'FaceColor', [0.2 0.6 0.6], 'EdgeColor', 'black');

% Set x-axis ticks and labels to correspond to the feature names
xticks(1:length(columnNames));  % Set tick positions to match the number of features
xticklabels(columnNames);       % Label each tick with the feature name
xtickangle(45);                 % Rotate x-axis labels for better visibility

% Title and labels
title('Feature Importances For RF');
xlabel('Feature');
ylabel('Importance Value');

%% features importance for LR 

% Extract the coefficients (Beta values)
coefficients = logisticModel.Coefficients.Estimate;

% Display feature importances
disp(coefficients);

% Create a bar chart
figure;
bar(coefficients, 'FaceColor', [0.2 0.6 0.6], 'EdgeColor', 'black');

% Set x-axis ticks and labels to correspond to the feature names
xticks(1:length(columnNames));  % Setting tick positions to match the number of features
xticklabels(columnNames);       % Labeling each tick with the feature name
xtickangle(45);                 % Rotating x-axis labels for better visibility

% Title and labels
title('Feature Importances For LR');
xlabel('Feature');
ylabel('Importance Value');

%% Initialize cross-validation

numFolds = 10;
cv = cvpartition(size(XTrain, 1), 'KFold', numFolds);

% Initialize arrays to store results
logisticAUC = zeros(numFolds, 1);
logisticError = zeros(numFolds, 1);
logisticTrainTime = zeros(numFolds, 1);
logisticPredTime = zeros(numFolds, 1);

logisticPrecision = zeros(numFolds, 1);
logisticRecall = zeros(numFolds, 1);
logisticF1 = zeros(numFolds, 1);
logisticAccuracy = zeros(numFolds, 1);

randomForestAUC = zeros(numFolds, 1);
randomForestError = zeros(numFolds, 1);
randomForestTrainTime = zeros(numFolds, 1);
randomForestPredTime = zeros(numFolds, 1);

randomForestPrecision = zeros(numFolds, 1);
randomForestRecall = zeros(numFolds, 1);
randomForestF1 = zeros(numFolds, 1);
randomForestAccuracy = zeros(numFolds, 1);

%% 10-fold Cross-validation

for i = 1:numFolds
    % Split the data into training and testing sets
    trainIdx = cv.training(i);
    testIdx = cv.test(i);

    XTrainFold = XTrain(trainIdx, :);
    yTrainFold = yTrain(trainIdx);

    XTestFold = XTrain(testIdx, :);
    yTestFold = yTrain(testIdx);

    % --- Logistic Regression ---
    tic;
    logisticModel = fitglm(XTrainFold, yTrainFold, 'Distribution', 'binomial', 'Link', 'logit');
    logisticTrainTime(i) = toc;

    tic;
    logisticPred = predict(logisticModel, XTestFold); % Predicted probabilities
    logisticPredTime(i) = toc;

    % Metrics for Logistic Regression
    logisticPredLabels = double(logisticPred > 0.5);
    [logisticPrecision(i), logisticRecall(i), logisticF1(i), logisticAccuracy(i)] = computeMetrics(yTestFold, logisticPredLabels);

    [~, ~, ~, logisticAUC(i)] = perfcurve(yTestFold, logisticPred, 1);
    logisticError(i) = mean(logisticPredLabels ~= yTestFold);

    % --- Random Forest Model ---
    tic;
    randomForestModel = TreeBagger(10, XTrainFold, yTrainFold, 'Method', 'classification', ...
        'MinLeafSize', 30, 'MaxNumSplits', 150, 'OOBPrediction', 'on');
    randomForestTrainTime(i) = toc;

    tic;
    [randomForestPred, scores] = predict(randomForestModel, XTestFold);
    randomForestPredTime(i) = toc;

    % Convert to numeric for evaluation
    randomForestPred = str2double(randomForestPred);

    % Metrics for Random Forest
    [randomForestPrecision(i), randomForestRecall(i), randomForestF1(i), randomForestAccuracy(i)] = computeMetrics(yTestFold, randomForestPred);

    [~, ~, ~, randomForestAUC(i)] = perfcurve(yTestFold, scores(:, 2), 1);
    randomForestError(i) = mean(randomForestPred ~= yTestFold);
end

% Average results
avgLogisticResults = [mean(logisticAUC), mean(logisticError), mean(logisticTrainTime), mean(logisticPredTime), ...
    mean(logisticPrecision), mean(logisticRecall), mean(logisticF1), mean(logisticAccuracy)];

avgRandomForestResults = [mean(randomForestAUC), mean(randomForestError), mean(randomForestTrainTime), mean(randomForestPredTime), ...
    mean(randomForestPrecision), mean(randomForestRecall), mean(randomForestF1), mean(randomForestAccuracy)];

% Display results
fprintf('(2) Model performance comparison before hyperparameter tuning and after cross validation\n');
fprintf('Logistic Regression - AUC: %.4f, Error: %.4f, Train Time: %.4f sec, Predict Time: %.4f sec\n', ...
    avgLogisticResults(1), avgLogisticResults(2), avgLogisticResults(3), avgLogisticResults(4));
fprintf('Precision: %.4f, Recall: %.4f, F1-Score: %.4f, Accuracy: %.4f\n\n', ...
    avgLogisticResults(5), avgLogisticResults(6), avgLogisticResults(7), avgLogisticResults(8));

fprintf('Random Forest - AUC: %.4f, Error: %.4f, Train Time: %.4f sec, Predict Time: %.4f sec\n', ...
    avgRandomForestResults(1), avgRandomForestResults(2), avgRandomForestResults(3), avgRandomForestResults(4));
fprintf('Precision: %.4f, Recall: %.4f, F1-Score: %.4f, Accuracy: %.4f\n\n', ...
    avgRandomForestResults(5), avgRandomForestResults(6), avgRandomForestResults(7), avgRandomForestResults(8));

% function for for calculating precision, recall,f1 and accuracy
function [precision, recall, f1, accuracy] = computeMetrics(yTrue, yPred)
    confMat = confusionmat(yTrue, yPred);
    tp = confMat(2, 2);
    fp = confMat(1, 2);
    fn = confMat(2, 1);
    tn = confMat(1, 1);

    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = 2 * (precision * recall) / (precision + recall);
    accuracy = (tp + tn) / sum(confMat(:));
end

%% Hyperparameter Tuning Using Cross-validation LR

% Definining a range of lambda values (regularization strength)
lambdaRange = logspace(-4, 4, 9);

% Initialize variables to store results
bestLambda = 0;
bestAccuracy = 0;

% Number of folds for cross-validation
k = 10;

% Perform grid search with cross-validation for Logistic Regression
for lambda = lambdaRange
    % Initialize variable to store the cross-validation accuracy for this lambda
    cvAccuracy = zeros(k, 1);

    % Perform k-fold cross-validation manually
    cv = cvpartition(length(yVal), 'KFold', k);

    for i = 1:k
        % Get the training and validation data for the current fold
        trainIdx = training(cv, i);
        valIdx = test(cv, i);

        XTrainFold = XVal(trainIdx, :);
        yTrainFold = yVal(trainIdx);
        XValFold = XVal(valIdx, :);
        yValFold = yVal(valIdx);

        % Creating the logistic regression model with the current lambda using L2 regularization
        logisticModel = fitclinear(XTrainFold, yTrainFold, 'Learner', 'logistic', 'Regularization', 'ridge', 'Lambda', lambda);

        % Predict the validation set labels
        logisticPred = predict(logisticModel, XValFold);

        % Calculate the accuracy for the current fold
        cvAccuracy(i) = sum(logisticPred == yValFold) / length(yValFold);
    end

    % Calculate the mean cross-validation accuracy for this lambda
    meanAccuracy = mean(cvAccuracy);

    % Update the best hyperparameter based on accuracy
    if meanAccuracy > bestAccuracy
        bestAccuracy = meanAccuracy;
        bestLambda = lambda;
    end
end

disp(['Best Lambda for Logistic Regression: ', num2str(bestLambda)]);
disp(['Best Cross-validated Accuracy for Logistic Regression: ', num2str(bestAccuracy)]);

%% Hyperparameter Tuning Using Cross-validation RF

% Define ranges of hyperparameters for Random Forest
%numTreesRange = [100, 200, 300];   % Number of trees in the forest
%minLeafSizeRange = [1, 5, 10];     % Minimum number of samples in a leaf node
%maxSplitsRange = [10, 50, 100];    % Maximum number of splits per tree

numTreesRange = [50, 100, 150, 200, 300];           % Number of trees in the forest
minLeafSizeRange = [1, 5, 10, 20, 30, 50, 80, 100]; % Minimum number of samples in a leaf node
maxSplitsRange = [5, 10, 30, 40, 50, 80, 100];      % Maximum number of splits per tree
% Initialize variables to store results
bestNumTrees = 0;
bestMaxSplits = 0;
bestMinLeafSize = 0;
bestRFAccuracy = 0;

% Perform grid search with cross-validation for Random Forest
for numTrees = numTreesRange
    for maxSplits = maxSplitsRange
        for minLeafSize = minLeafSizeRange
            % Initialize cross-validation accuracy storage
            foldAccuracies = zeros(10, 1);
            
            % Perform 10-fold cross-validation manually
            cv = cvpartition(yTrain, 'KFold', 10);  % 10-fold cross-validation
            
            for fold = 1:cv.NumTestSets
                % Separate the training and validation sets for this fold
                XTrainFold = XTrain(training(cv, fold), :);
                yTrainFold = yTrain(training(cv, fold));
                XValFold = XTrain(test(cv, fold), :);
                yValFold = yTrain(test(cv, fold));
                
                % Train a random forest model with current hyperparameters
                randomForestModel = TreeBagger(numTrees, XTrainFold, yTrainFold, 'Method', 'classification', ...
                    'OOBPrediction', 'on', 'OOBPredictorImportance', 'on', 'MaxNumSplits', maxSplits, ...
                    'MinLeafSize', minLeafSize);
                
                % Get predictions for the validation fold
                rfPred = predict(randomForestModel, XValFold);
                rfPred = str2double(rfPred);  % Convert string predictions to numeric
                
                % Calculate the accuracy for this fold
                foldAccuracies(fold) = sum(rfPred == yValFold) / length(yValFold);
            end
            
            % Calculate the average accuracy across all folds
            avgAccuracy = mean(foldAccuracies);
            
            % Update the best hyperparameters based on accuracy
            if avgAccuracy > bestRFAccuracy
                bestRFAccuracy = avgAccuracy;
                bestNumTrees = numTrees;
                bestMaxSplits = maxSplits;
                bestMinLeafSize = minLeafSize;
            end
        end
    end
end

% Display the best hyperparameters and accuracy
disp(['Best Number of Trees for Random Forest: ', num2str(bestNumTrees)]);
disp(['Best MaxNumSplits for Random Forest: ', num2str(bestMaxSplits)]);
disp(['Best MinLeafSize for Random Forest: ', num2str(bestMinLeafSize)]);
disp(['Best Cross-validated Accuracy for Random Forest: ', num2str(bestRFAccuracy)]);


%% model post_hyper parameter tuning performance test

% logistic Regression
% Fit the final logistic regression model with regularization (L2 regularization)
finalLogisticModel = fitclinear(XTrain, yTrain, 'Learner', 'logistic', 'Regularization', 'ridge', 'Lambda', bestLambda);

% Evaluate the final model on the test set
logisticTestPred = predict(finalLogisticModel, XTest);
logisticTestAccuracy = sum(logisticTestPred == yTest) / length(yTest);
disp(['Logistic Regression - Final Test Accuracy: ', num2str(logisticTestAccuracy)]);


% Random Forest
% Train the final random forest model with the best hyperparameters
finalRandomForestModel = TreeBagger(bestNumTrees, XTrain, yTrain, 'Method', 'classification', 'OOBPrediction', 'on', 'OOBPredictorImportance', 'on','MinLeafSize', bestMinLeafSize, 'MaxNumSplits', bestMaxSplits);

% Evaluate the final model on the test set
rfTestPred = predict(finalRandomForestModel, XTest);
rfPred = str2double(rfTestPred);  % Convert string predictions to numeric
rfTestAccuracy = sum(rfPred == yTest) / length(yTest);

%rfTestAccuracy = sum(strcmp(rfTestPred, yTest)) / length(yTest);
disp(['Random Forest - Final Test Accuracy: ', num2str(rfTestAccuracy)]);

%% Predict and evaluate logistic regression model

logisticModel=finalLogisticModel;
RandomForestModel =finalRandomForestModel ;

yPredLogisticTrain = round(predict(logisticModel, XTrain));
yPredLogisticVal = round(predict(logisticModel, XVal));
yPredLogisticTest = round(predict(logisticModel, XTest));

logisticTrainAcc = sum(yPredLogisticTrain == yTrain) / length(yTrain);
logisticValAcc = sum(yPredLogisticVal == yVal) / length(yVal);
logisticTestAcc = sum(yPredLogisticTest == yTest) / length(yTest);

% Calculate precision, recall, F1 score, and AUC for logistic regression
[cmLogistic, ~] = confusionmat(yTest, yPredLogisticTest);
logisticPrecision = cmLogistic(2,2) / sum(cmLogistic(:,2));
logisticRecall = cmLogistic(2,2) / sum(cmLogistic(2,:));
logisticF1 = 2 * (logisticPrecision * logisticRecall) / (logisticPrecision + logisticRecall);
[XLogistic, YLogistic, ~, logisticAUC] = perfcurve(yTest, predict(logisticModel, XTest), 1);

% Predict and evaluate random forest model
[yPredRFTrain, scoresRFTrain] = predict(randomForestModel, XTrain);
[yPredRFVal, scoresRFVal] = predict(randomForestModel, XVal);
[yPredRFTest, scoresRFTest] = predict(randomForestModel, XTest);

yPredRFTrain = str2double(yPredRFTrain);
yPredRFVal = str2double(yPredRFVal);
yPredRFTest = str2double(yPredRFTest);

rfTrainAcc = sum(yPredRFTrain == yTrain) / length(yTrain);
rfValAcc = sum(yPredRFVal == yVal) / length(yVal);
rfTestAcc = sum(yPredRFTest == yTest) / length(yTest);

% Calculate precision, recall, F1 score, and AUC for random forest
[cmRF, ~] = confusionmat(yTest, yPredRFTest);
rfPrecision = cmRF(2,2) / sum(cmRF(:,2));
rfRecall = cmRF(2,2) / sum(cmRF(2,:));
rfF1 = 2 * (rfPrecision * rfRecall) / (rfPrecision + rfRecall);
[XRF, YRF, ~, rfAUC] = perfcurve(yTest, scoresRFTest(:,2), 1);

%% Display results
fprintf('(3) Model performance comparison after hyperparameter tuning\n');
fprintf('Logistic Regression Training Accuracy: %.4f\n', logisticTrainAcc);
fprintf('Logistic Regression Validation Accuracy: %.4f\n', logisticValAcc);
fprintf('Logistic Regression Testing Accuracy: %.4f\n', logisticTestAcc);
fprintf('Logistic Regression Precision: %.4f\n', logisticPrecision);
fprintf('Logistic Regression Recall: %.4f\n', logisticRecall);
fprintf('Logistic Regression F1 Score: %.4f\n', logisticF1);
fprintf('Logistic Regression AUC: %.4f\n', logisticAUC);

fprintf('Random Forest Training Accuracy: %.4f\n', rfTrainAcc);
fprintf('Random Forest Validation Accuracy: %.4f\n', rfValAcc);
fprintf('Random Forest Testing Accuracy: %.4f\n', rfTestAcc);
fprintf('Random Forest Precision: %.4f\n', rfPrecision);
fprintf('Random Forest Recall: %.4f\n', rfRecall);
fprintf('Random Forest F1 Score: %.4f\n', rfF1);
fprintf('Random Forest AUC: %.4f\n', rfAUC);

%% Confusion matrices with additional matrix on the right side showing percentages
plotConfusionMatrixWithPercentages(cmLogistic);
plotConfusionMatrixWithPercentages1(cmRF);

%% ROC curves
figure;
plot(XLogistic, YLogistic);
hold on;
plot(XRF, YRF);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC Curves');
legend('Logistic Regression', 'Random Forest');
hold off;

%% Save the model to a .mat file  here i change the file name to they dont change the data used for testing
save('randomForestModel1.mat', 'randomForestModel');
save('logisticModel1.mat', 'logisticModel');
% Save yTest to a .csv file 
csvwrite('yTest1.csv', yTest);
csvwrite('XTest1.csv', XTest);
csvwrite('yTrain1.csv', yTrain);
csvwrite('XTrain1.csv', XTrain);
csvwrite('XVal1.csv', XVal);
csvwrite('yVal1.csv', yVal);



