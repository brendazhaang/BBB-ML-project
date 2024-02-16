function p = predictNN(Theta1, Theta2, Theta3, X, threshold)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta3, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
h3 = sigmoid([ones(m, 1) h2] * Theta3');


%%notes: high recall low precision; raise threshold?

p(find(h3 >= threshold)) = 1;
p(find(h3 < threshold)) = 0;

%[dummy, p] = max(h2, [], 2) - 1;

% =========================================================================


end
