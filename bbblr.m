clear ; close all; clc

% load training data
load('X_training.mat'); %7162 x 1119
load('Y_training.mat'); %7162 x 2
load('X_test.mat'); %74 x 1119
load('Y_test.mat'); %74 x 2

[m, n] = size(X_training); %m = 7162, n = 1119

%add bias unit
X = [ones(m, 1) X_training];

% initialize parameters
initial_theta = zeros(size(X, 2), 1);
lambda = 1;

% optimize theta
options = optimset('GradObj', 'on', 'MaxIter', 200);
[theta, J, exit_flag] = ...
	fminunc(@(t)(lrCostFunctionReg(t, X, Y_training, lambda)), initial_theta, options);
 
% find accuracy on training set
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == Y_training)) * 100);
fprintf('tp: %f\n', sum(Y_training(find(Y_training == 1)) == p(find(Y_training == 1))));
fprintf('fp: %f\n', sum(Y_training(find(Y_training == 0)) != p(find(Y_training == 0))));




%test set
X_test = [ones(size(X_test,1),1) X_test];
p2 = predict(theta, X_test);
fprintf('Test Accuracy: %f\n', mean(double(p2 == Y_test)) * 100);

%split training set into csv set?