function plotLinearRegressionLine(X, y, theta)

figure; hold on;
plot(X(:,2), X*theta, '-', X(:,2), y, 'rx', 'MarkerSize', 10);
hold off;

end