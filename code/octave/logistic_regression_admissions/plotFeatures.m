function plotFeatures(X, y)
%PLOTDATA Plots the data points X and y into a new figure
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% Find Indices of Positive and Negative Examples
pos = find(y == 1);
neg = find(y == 0);

X1 = X(:,1);
X2 = X(:,2);

% Plot Examples
plot(X1(pos, 1), y(pos, 1), 'ko', 'MarkerFaceColor', 'g', 'MarkerSize', 7);
plot(X1(neg, 1), y(neg, 1), 'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
plot(X2(pos, 1), y(pos, 1), 'k^', 'MarkerEdgeColor', 'g', 'MarkerFaceColor', 'none', 'MarkerSize', 7);
plot(X2(neg, 1), y(neg, 1), 'kv', 'MarkerEdgeColor', 'r', 'MarkerSize', 7);

hold off;

end