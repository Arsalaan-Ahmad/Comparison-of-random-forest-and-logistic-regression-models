% Load the saved model
loadedModelLR = load('randomForestModel.mat');
randomForestModel = loadedModelLR.randomForestModel;

loadedModelRF = load('logisticModel.mat');
logisticModel = loadedModelRF.logisticModel;
 
% Load data from CSV
XTest = readmatrix('XTest.csv');
yTest = readmatrix('yTest.csv');
XTrain = readmatrix('XTrain.csv');
yTrain = readmatrix('yTrain.csv');
XVal = readmatrix('XVal.csv');
yVal= readmatrix('yVal.csv');

%% functions

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
yPredLogisticVal = round(predict(logisticModel, XVal));
yPredLogisticTest = round(predict(logisticModel, XTest));

logisticValAcc = sum(yPredLogisticVal == yVal) / length(yVal);
logisticTestAcc = sum(yPredLogisticTest == yTest) / length(yTest);

% Confusion matrices with additional matrix on the right side showing percentages
function plotConfusionMatrixWithPercentagesLR(cm)
    % confusion matrix heatmap for Logistic Regression
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

function plotConfusionMatrixWithPercentagesRF(cm)
    % confusion matrix heatmap for Random Forest
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

%% Comparison and statistics

fprintf('(1) final performance comparison after hyperparameter tuning and cross validation');
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
[XLogistic, YLogistic, ~, logisticAUC] = perfcurve(yTest, predict(logisticModel, XTest), 1);

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
[XRF, YRF, ~, rfAUC] = perfcurve(yTest, scoresRFTest(:,2), 1);

%% Display results
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
plotConfusionMatrixWithPercentagesRF(cmLogistic);
plotConfusionMatrixWithPercentagesLR(cmRF);

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

%% Measure runtime for Logistic Regression
numRuns = 10;
logisticTimes = zeros(1, numRuns);
rfTimes = zeros(1, numRuns);

for i = 1:numRuns
    tic;
    yPredLogisticTest = round(predict(logisticModel, XTest));
    logisticTimes(i) = toc;
end

for i = 1:numRuns
    tic;
    [yPredRFTest, ~] = predict(randomForestModel, XTest);
    rfTimes(i) = toc;
end

avgLogisticTime = mean(logisticTimes);
avgRFTime = mean(rfTimes);

fprintf('Average Logistic Regression Runtime: %.4f seconds\n', avgLogisticTime);
fprintf('Average Random Forest Runtime: %.4f seconds\n', avgRFTime);


% Bar Graph: Precision, Recall, F1 Scores
figure;
barData = [logisticPrecision, logisticRecall, logisticF1; ...
           rfPrecision, rfRecall, rfF1];
bar(barData);
title('Model Performance Metrics');
xlabel('Models');
ylabel('Scores');
xticks(1:2);
xticklabels({'Logistic Regression', 'Random Forest'});
legend('Precision', 'Recall', 'F1 Score', 'Location', 'northwest');
grid on;

% Training vs Testing Accuracy Plot
figure;
trainAcc = [logisticValAcc, rfValAcc];
testAcc = [logisticTestAcc, rfTestAcc];
modelNames = {'Logistic Regression', 'Random Forest'};

plot(1:2, trainAcc, '-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(1:2, testAcc, '-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Models');
ylabel('Accuracy');
xticks(1:2);
xticklabels(modelNames);
legend('Validation Accuracy', 'Testing Accuracy', 'Location', 'southeast');
title('Training vs Testing Accuracy');
grid on;
hold off;

%% Verifing True Positive Rate (TPR) and False Positive Rate (FPR) for Logistic Regression and Random Forest

% TPR = TP / (TP + FN)
% FPR = FP / (FP + TN)

% Calculating TPR and FPR for Logistic Regression
tpLogistic = cmLogistic(2, 2);
fpLogistic = cmLogistic(1, 2);
fnLogistic = cmLogistic(2, 1);
tnLogistic = cmLogistic(1, 1);

logisticTPR = tpLogistic / (tpLogistic + fnLogistic);
logisticFPR = fpLogistic / (fpLogistic + tnLogistic);

% Calculating TPR and FPR for Random Forest
tpRF = cmRF(2, 2);
fpRF = cmRF(1, 2);
fnRF = cmRF(2, 1);
tnRF = cmRF(1, 1);

rfTPR = tpRF / (tpRF + fnRF);
rfFPR = fpRF / (fpRF + tnRF);

% Display TPR and FPR for both models
fprintf('Logistic Regression True Positive Rate (TPR): %.4f\n', logisticTPR);
fprintf('Logistic Regression False Positive Rate (FPR): %.4f\n', logisticFPR);

fprintf('Random Forest True Positive Rate (TPR): %.4f\n', rfTPR);
fprintf('Random Forest False Positive Rate (FPR): %.4f\n', rfFPR);

% Visualize TPR and FPR Comparison
figure;
barDataTPR_FPR = [logisticTPR, logisticFPR; rfTPR, rfFPR];
bar(barDataTPR_FPR);
title('TPR and FPR Comparison');
xlabel('Models');
ylabel('Rates');
xticks(1:2);
xticklabels({'Logistic Regression', 'Random Forest'});
legend('True Positive Rate (TPR)', 'False Positive Rate (FPR)', 'Location', 'northwest');
grid on;
