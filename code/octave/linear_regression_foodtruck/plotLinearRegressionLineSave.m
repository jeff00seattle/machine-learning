function plotLinearRegressionLineSave(X, y, theta)

pause(1)

figure; hold on;
clf()
f = figure(1);
plot(X(:,2), X*theta, '-', X(:,2), y, 'rx', 'MarkerSize', 10);
t = int32(time());
pathfig=sprintf('output/%d.png',t);
saveas(f, pathfig)
hold off;

end