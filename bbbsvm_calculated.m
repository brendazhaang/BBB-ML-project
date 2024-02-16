clear ; close all; clc

%load data
load('Y_training.mat');   %7162 x 1
load('Y_test.mat');   

load('cdk_select_train.mat'); %7162 x 72
load('cdk_select_test.mat'); %74 x 1

load('cdk_all_train.mat'); %7162 x 221
load('cdk_all_test.mat'); 

%cross validation set
X_cv = cdk_all_train(7001:7162,:);  %162 x 193
Y_cv = Y_training(7001:7162,:);

X_train = cdk_all_train(1:7000,:);  %7000 x 193
Y_train = Y_training(1:7000,:);

[u, v] = size(cdk_select_train);
[m, n] = size(X_train); %m = 7000,
[a, b] = size(cdk_all_test);  %a = 74,
[c, d] = size(X_cv);

% normalize data
for i = 1:u
  cdk_select_train(i,:) = [cdk_select_train(i,:) - mean(cdk_select_train(i,:))] ./ std(cdk_select_train(i,:));
  cdk_all_train(i,:) = [cdk_all_train(i,:) - mean(cdk_all_train(i,:))] ./ std(cdk_all_train(i,:),'fro');
end
for i = 1:m
  X_train(i,:) = [X_train(i,:) - mean(X_train(i,:))] ./ std(X_train(i,:));
end
for i = 1:c
  X_cv(i,:) = [X_cv(i,:) - mean(X_cv(i,:))] ./ std(X_cv(i,:));
end
for i = 1:a
  cdk_select_test(i,:) = [cdk_select_test(i,:) - mean(cdk_select_test(i,:))] ./ std(cdk_select_test(i,:));
  cdk_all_test(i,:) = [cdk_all_test(i,:) - mean(cdk_all_test(i,:))] ./ std(cdk_all_test(i,:),'fro');
end

%----cross val
%% used LIBSVM from https://www.csie.ntu.edu.tw/~cin/libsvm/#download 
##[C gamma auc] = findParams(X_train, Y_train, X_cv, Y_cv);
##fprintf('C: %f\n', C);
##fprintf('gamma: %f\n', gamma);
##fprintf('\nCV Set AUC: %f\n ', auc);

C = 8;
gamma = 8;

%-------------------------calculated set
model = svmtrain(Y_training, cdk_select_train, ['-q -t 2 -b 1 -c ', num2str(C), ' -g ',  num2str(gamma)]);
display('calculated descriptors (72)');

[pred, accuracy, prob_est] = svmpredict(Y_training, cdk_select_train, model, '-b 1');
display('training set: ');
[acc, sp, se] = getAcc(pred, Y_training)
RES = roc(pred, Y_training);
fprintf('AUC: %f\n', RES.AUC);

%test set
[pred, accuracy, prob_est] = svmpredict(Y_test, cdk_select_test, model, '-b 1');
display('test set: ');
[acc, sp, se] = getAcc(pred, Y_test)

%figure;
RES = roc(pred, Y_test);
%plot(RES.FPR, RES.TPR);
fprintf('AUC: %f\n', RES.AUC);


%---------------not working:
display('calculated descriptors (221)');

model = svmtrain(Y_training, cdk_all_train, ['-q -t 2 -b 1 -c ', num2str(C), ' -g ',  num2str(gamma)]);

[pred, accuracy, prob_est] = svmpredict(Y_training, cdk_all_train, model, '-b 1');
display('training set: ');
[acc, sp, se] = getAcc(pred, Y_training)
RES = roc(pred, Y_training);
fprintf('AUC: %f\n', RES.AUC);

%test set
[pred, accuracy, prob_est] = svmpredict(Y_test, cdk_all_test, model, '-b 1');
display('test set: ');
[acc, sp, se] = getAcc(pred, Y_test)

figure;
RES = roc(pred, Y_test);
plot(RES.FPR, RES.TPR);
fprintf('AUC: %f\n', RES.AUC);
