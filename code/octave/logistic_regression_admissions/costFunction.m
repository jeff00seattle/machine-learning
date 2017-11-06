function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

grad = zeros(size(theta));
xt = X * theta;
prediction = sigmoid(xt);

% plotSigmoidSave(prediction, xt);

J = -(1 / m) * sum( (y .* log(prediction)) + ((1 - y) .* log(1 - prediction)) );
% fprintf(' %f \n', J);

for i = 1 : size(theta, 1)
    grad(i) = (1 / m) * sum( (prediction - y) .* X(:, i) );
end

end
