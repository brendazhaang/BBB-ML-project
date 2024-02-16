function [J, grad] = lrCostFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hyp = sigmoid(X * theta);

J = sum(-y' * log(hyp) - (1 - y)' * log(1 - hyp)) / m + lambda * sum(theta(2:end,:) .^2) / (2 * m);

% grad for theta0 
grad(1) = X(:,1)' * (hyp - y) / m;

grad(2:end) = X(:,2:end)' *(hyp - y) ./ m + lambda .* theta(2:end) ./ m;


% =============================================================

end
