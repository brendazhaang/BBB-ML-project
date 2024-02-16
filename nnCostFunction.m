function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:(hidden_layer_size * (input_layer_size + 1))), ...
                hidden_layer_size, (input_layer_size + 1));

k = (hidden_layer_size * (input_layer_size + 1));
Theta2 = reshape(nn_params((1+k):(k + hidden_layer_size * (hidden_layer_size+1))), ...
                hidden_layer_size, (hidden_layer_size + 1));
                
Theta3 = reshape(nn_params((1+ k + hidden_layer_size * (hidden_layer_size+1)):end), ...
                num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

% ====================== YOUR CODE HERE ======================

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%add bias unit
X = [ones(m, 1) X]; %7162 x 1120

deltas3 = zeros(size(Theta3)); %1 x 46
deltas2 = zeros(size(Theta2)); %45 x 46
deltas1 = zeros(size(Theta1)); %45 x 1120

delta2 = 0; %hidden layer
delta3 = 0; %hidden layer
delta4 = 0; %output layer

for i = 1:m
  %forward prop
  a1 = X(i,:)'; %1120 x 1
  a2 = sigmoid(Theta1 * a1); %45 x 1
  a2 = [1 ; a2]; %46 x 1
  
  a3 = sigmoid(Theta2 * a2); %45 x 1
  a3 = [1; a3]; %46 x 1
  
  h = sigmoid(Theta3 * a3); %1 x 1
  
  %cost
  J +=  -y(i,:) * log(h) - [(1 - y(i,:)) * log(1 - h)]; 
  
  %backprop
  delta4 = h - y(i,:)'; %1 x 1
  delta3 = Theta3' * delta4 .* [1; sigmoidGradient(Theta2 * a2)]; %46 x 1
  delta3 = delta3(2:end);
  delta2 = Theta2' * delta3 .* [1; sigmoidGradient(Theta1 * a1)]; 
  delta2 = delta2(2:end); %45 x 1
  
  deltas3 += delta4 * a3';
  deltas2 += delta3 * a2'; 
  deltas1 += delta2 * a1'; 

  end

J /= m;

Theta1_grad = deltas1 ./ m;
Theta2_grad = deltas2 ./ m;
Theta3_grad = deltas3 ./ m;

Theta1_grad(:, 2:end) += lambda * Theta1(:, 2:end) / m;
Theta2_grad(:, 2:end) += lambda * Theta2(:, 2:end) / m;
Theta3_grad(:, 2:end) += lambda * Theta3(:, 2:end) / m;


%regularization
t1 = Theta1'(2:end,:)';  
t2 = Theta2'(2:end,:);  
t3 = Theta3'(2:end,:);
J2 = 0;

for i = 1:size(t1,1)
  J2 += sum(t1(i,:) .^2) + sum(t2(i,:) .^2) + sum(t3(i,:) .^2);
end

J = J + J2 * lambda / (2 * m);

% -------------------------------------------------------------



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end
