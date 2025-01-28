% Load the prediction result
result = load('prediction_result.mat');

% Display the predicted label and probabilities
disp(['Predicted Index: ', num2str(result.predicted_index)]);
disp(['Predicted Label: ', result.predicted_label]);

% Display class probabilities
cifar10Labels = string(result.cifar10_labels);
figure;
bar(categorical(cifar10Labels), result.predictions);
xlabel('Class');
ylabel('Probability');
title(['Predicted Label: ', result.predicted_label]);
