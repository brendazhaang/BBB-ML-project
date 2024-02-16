## Copyright (C) 2022 brend
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {} {@var{retval} =} crossval (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: brend <brend@BRENDA-LENOVO2>
## Created: 2022-06-30

function [lambda acc AUC] = crossval(X, Y, initial_nn_params, input_layer_size, ...
                  hidden_layer_size, num_labels)
t = 0.76;
L_vec = 0.1#[0.01 0.03 0.06 0.1 0.3 0.5 0.6 0.8 1]';
acc_vec = zeros(size(L_vec), 3);
auc_vec = zeros(size(L_vec));

%split data into 3 folds
k = idivide(7162, int32 (3), 'fix'); %2387

X1 = X(1:k,:);
Y1 = Y(1:k,:);

X2 = X(k+1:2*k+1,:);
Y2 = Y(k+1:2*k+1,:);

X3 = X(2*k+2:size(X,1),:);
Y3 = Y(2*k+2:size(X,1),:);

for i = 1 : length(L_vec)
  l = L_vec(i)
 
  options = optimset('MaxIter', 80);

  ###train on sets 1& 2, test set 3
  % Create "short hand" for the cost function to be minimized
  costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, [X1;X2], [Y1;Y2], l);

  [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

  % Obtain Theta1 and Theta2 back from nn_params
  Theta1 = reshape(nn_params(1:(hidden_layer_size * (input_layer_size + 1))), ...
                hidden_layer_size, (input_layer_size + 1));

  k = (hidden_layer_size * (input_layer_size + 1));
  Theta2 = reshape(nn_params((1+k):(k + hidden_layer_size * (hidden_layer_size+1))), ...
                hidden_layer_size, (hidden_layer_size + 1));
                
  Theta3 = reshape(nn_params(1+(k + hidden_layer_size * (hidden_layer_size+1)):end), ...
                num_labels, (hidden_layer_size + 1));
  pred = predictNN(Theta1, Theta2, Theta3, X3, t);
  [a1, sp1, se1] = getAcc(pred, Y3);
  RES = roc(pred, Y3);
  auc1 = RES.AUC;
  
  ###train on sets 1&3, test set 2
  % Create "short hand" for the cost function to be minimized
  costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, [X1;X3], [Y1;Y3], l);

  [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

  % Obtain Theta1 and Theta2 back from nn_params
  Theta1 = reshape(nn_params(1:(hidden_layer_size * (input_layer_size + 1))), ...
                hidden_layer_size, (input_layer_size + 1));

  k = (hidden_layer_size * (input_layer_size + 1));
  Theta2 = reshape(nn_params((1+k):(k + hidden_layer_size * (hidden_layer_size+1))), ...
                hidden_layer_size, (hidden_layer_size + 1));
                
  Theta3 = reshape(nn_params(1+(k + hidden_layer_size * (hidden_layer_size+1)):end), ...
                num_labels, (hidden_layer_size + 1));
  
  pred = predictNN(Theta1, Theta2, Theta3, X2, t);
  [a2, sp2, se2] = getAcc(pred, Y2);
  RES = roc(pred, Y2);
  auc2 = RES.AUC;
  
  ###train sets 2&3, test set 1
  % Create "short hand" for the cost function to be minimized
  costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, [X2;X3], [Y2;Y3], l);

  [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

  % Obtain Theta1 and Theta2 back from nn_params
  Theta1 = reshape(nn_params(1:(hidden_layer_size * (input_layer_size + 1))), ...
                hidden_layer_size, (input_layer_size + 1));

  k = (hidden_layer_size * (input_layer_size + 1));
  Theta2 = reshape(nn_params((1+k):(k + hidden_layer_size * (hidden_layer_size+1))), ...
                hidden_layer_size, (hidden_layer_size + 1));
                
  Theta3 = reshape(nn_params(1+(k + hidden_layer_size * (hidden_layer_size+1)):end), ...
                num_labels, (hidden_layer_size + 1));
                
  pred = predictNN(Theta1, Theta2, Theta3, X1, t);
  [a3, sp3, se3] = getAcc(pred, Y1);
  RES = roc(pred, Y1);
  auc3 = RES.AUC;
  
  a = (a1+a2+a3) / 3.0;
  sp = (sp1+sp2+sp3) / 3.0;
  se = (se1+se2+se3) / 3.0;
 
  auc_vec(i) = (auc1+auc2+auc3) / 3.0;
  acc_vec(i,:) = [a sp se];
  
end

[AUC, index] = max(auc_vec); 
lambda = L_vec(index);
acc = acc_vec(index,:);

endfunction
