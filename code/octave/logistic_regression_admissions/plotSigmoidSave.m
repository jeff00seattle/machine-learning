function plotSigmoidSave(prediction, xt)

pause(1)

figure; hold on;
clf()
f = figure(1);
plot(xt, prediction, 'k.', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none', 'MarkerSize', 7);
t = int32(time());
pathfig=sprintf('output/%d.png',t);
saveas(f, pathfig)
hold off;

end