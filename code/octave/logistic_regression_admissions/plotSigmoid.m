function plotSigmoid(prediction, xt)

figure; hold on;
figure('Position',[0,0,300,300]);
plot(xt, prediction, 'k.', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none', 'MarkerSize', 7);
hold off;

end
