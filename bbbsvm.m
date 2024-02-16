clear ; close all; clc

% load training data
load('X_training.mat');   %7162 x 1119
load('Y_training.mat');   %7162 x 1
load('X_test.mat');       %74 x 1119
load('Y_test.mat');       %74 x 1
load('X_test_padel.mat'); %74 x 1544


%cross validation set
X_cv = X_training(7001:7162,:);  %162 x 119
Y_cv = Y_training(7001:7162,:);

X_train = X_training(1:7000,:);  %7000 x 119
Y_train = Y_training(1:7000,:);

[m, n] = size(X_train); %m = 7000, n = 1119
[a, b] = size(X_test);     %a = 74, b = 119
[c, d] = size(X_cv);
[u, v] = size(X_training);

% normalize data
for i = 1:u
  X_training(i,:) = [X_training(i,:) - mean(X_training(i,:))] ./ std(X_training(i,:));
end
for i = 1:m
  X_train(i,:) = [X_train(i,:) - mean(X_train(i,:))] ./ std(X_train(i,:));
end
for i = 1:c
  X_cv(i,:) = [X_cv(i,:) - mean(X_cv(i,:))] ./ std(X_cv(i,:));
end
for i = 1:a
  X_test(i,:) = [X_test(i,:) - mean(X_test(i,:))] ./ std(X_test(i,:));
end
##for i = 1:a
##  X_test_padel(i,:) = [X_test_padel(i,:) - mean(X_test_padel(i,:))] ./ std(X_test_padel(i,:));
##end

%% used LIBSVM from https://www.csie.ntu.edu.tw/~cin/libsvm/#download 

##[C gamma auc] = findParams(X_train, Y_train, X_cv, Y_cv);
##fprintf('C: %f\n', C);
##fprintf('gamma: %f\n', gamma);
##fprintf('\nCV Set AUC: %f\n ', auc);

C = 32;
gamma = 2;

model = svmtrain(Y_training, X_training, ['-q -t 2 -b 1 -c ', num2str(C), ' -g ',  num2str(gamma)]);

%---------------------training set
[pred, accuracy, prob_est] = svmpredict(Y_training, X_training, model, '-b 1');
display('training set: ');
[acc, sp, se] = getAcc(pred, Y_training)
RES = roc(pred, Y_training);
fprintf('AUC: %f\n', RES.AUC);

%---------------------test set
##maybe use prob est?
[pred, accuracy, prob_est] = svmpredict(Y_test, X_test, model, '-b 1');
display('test set: ');
[acc, sp, se] = getAcc(pred, Y_test)
RES = roc(pred, Y_test);
fprintf('AUC: %f\n', RES.AUC);