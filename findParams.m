function [C gamma auc] = findParams(X, Y)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
gamma = 0.3;

% ====================== YOUR CODE HERE ======================
k = idivide(7162, int32 (3), 'fix'); %2387

X1 = X(1:k,:);
Y1 = Y(1:k,:);

X2 = X(k+1:2*k+1,:);
Y2 = Y(k+1:2*k+1,:);

X3 = X(2*k+2:size(X,1),:);
Y3 = Y(2*k+2:size(X,1),:);

auc_vec = zeros(1,1)#(6,8);

C_vec = [2^-5 2^-3 0.5 2 2^3 2^5]';
Gamma_vec = [2^-13 2^-10 2^-7 2^-5 2^-3 0.5 2 2^3]';

for i = 1:length(C_vec)
  c_t = C_vec(i);
  for j = 1:length(Gamma_vec)
    g_t = Gamma_vec(j);
    
    fprintf('C: %f  gamma: %f\n', c_t, g_t);
    
    model = svmtrain([Y1;Y2], [X1;X2], ['-q -t 2 -c ', num2str(c_t), ' -g ',  num2str(g_t)]);
    pred = svmpredict(Y3, X3, model);
    RES = roc(pred, Y3);
    auc1 = RES.AUC
    [acc, sp, se] = getAcc(pred, Y3)
    
    model = svmtrain([Y1;Y3], [X1;X3], ['-q -t 2 -c ', num2str(c_t), ' -g ', num2str(g_t)]);
    pred = svmpredict(Y2, X2, model);
    RES = roc(pred, Y2);
    auc2 = RES.AUC
    [acc, sp, se] = getAcc(pred, Y2)
    
    model = svmtrain([Y2;Y3], [X2;X3], ['-q -t 2 -c ', num2str(c_t), ' -g ', num2str(g_t)]);
    pred = svmpredict(Y1, X1, model);
    RES = roc(pred, Y1);
    auc3 = RES.AUC
    [acc, sp, se] = getAcc(pred, Y1)
    
    auc_vec(i, j) = (auc1 + auc2 + auc3)/ 3.0;

  end
end


[m, g_i] = max(max(auc_vec)); %column
[m, C_i] = max(max(auc_vec')); %row

gamma = Gamma_vec(g_i);
C = C_vec(C_i);
auc = m;

% =========================================================================

end
