clear ; close all; clc

% setup the parameters
input_layer_size  = 1119; %  
hidden_layer_size = 35;   % hidden units
num_labels = 1;           % 0 or 1   

% load training data
load('X_training.mat');   %7162 x 1119
load('Y_training.mat');   %7162 x 1
load('X_test.mat');       %74 x 1119
load('Y_test.mat');       %74 x 1

[m, n] = size(X_training); %m = 7162, n = 1119
[a, b] = size(X_test);     %a = 74, b = 119

% normalize data
for i = 1:m
  X_training(i,:) = [X_training(i,:) - mean(X_training(i,:))] ./ std(X_training(i,:));
end
for i = 1:a
  X_test(i,:) = [X_test(i,:) - mean(X_test(i,:))] ./ std(X_test(i,:));
end

for i = [1 2 3 4 5]
fprintf('round %f\n', i);
% initialze thetas to random values
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta3 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta4 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta5 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:); initial_Theta2(:); initial_Theta3(:); initial_Theta4(:); initial_Theta5(:)];

%% =================== Part 8: Training NN ===================
options = optimset('MaxIter', 300);

% hyperparameter
[lambda cv_acc AUC] = crossval2(X_training, Y_training, initial_nn_params, input_layer_size, ...
       hidden_layer_size, num_labels);

lambda       
fprintf('\nCV Set:\n acc: %f\n sp: %f\n se: %f\n', cv_acc);
fprintf('AUC: %f\n', AUC);

%lambda

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction2(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_training, Y_training, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:(hidden_layer_size * (input_layer_size + 1))), ...
                hidden_layer_size, (input_layer_size + 1));

k = (hidden_layer_size * (input_layer_size + 1));
Theta2 = reshape(nn_params((1+k):(k + hidden_layer_size * (hidden_layer_size+1))), ...
                hidden_layer_size, (hidden_layer_size + 1));

k = (k + hidden_layer_size * (hidden_layer_size+1));         
Theta3 = reshape(nn_params((1+k):(k + hidden_layer_size * (hidden_layer_size+1))), ...
                hidden_layer_size, (hidden_layer_size + 1));           

k = (k + hidden_layer_size * (hidden_layer_size+1));         
Theta4 = reshape(nn_params((1+k):(k + hidden_layer_size * (hidden_layer_size+1))), ...
                hidden_layer_size, (hidden_layer_size + 1));               
               
Theta5 = reshape(nn_params(1+(k + hidden_layer_size * (hidden_layer_size+1)):end), ...
                num_labels, (hidden_layer_size + 1));

%% ================= Part 10: Implement Predict =================

t = 0.76
pred = predictNN2(Theta1, Theta2, Theta3, Theta4, Theta5, X_training, t);
disp('Training Set: ');  
[a, sp, se] = getAcc(pred, Y_training)
    
RES = roc(pred, Y_training);
plot(RES.FPR, RES.TPR);
fprintf('AUC: %f\n', RES.AUC);
fprintf('g mean: %f\n', sqrt(sp*se));

    %-------------test set-----------------
pred = predictNN2(Theta1, Theta2, Theta3, Theta4, Theta5, X_test, t);
disp('Test Set: ');
[a, sp, se] = getAcc(pred, Y_test)

RES = roc(pred, Y_test);
figure;
plot(RES.FPR, RES.TPR);
fprintf('AUC: %f\n', RES.AUC);
fprintf('g mean: %f\n\n', sqrt(sp*se));

end