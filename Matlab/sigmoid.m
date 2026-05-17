x = linspace(-20, 20, 2000);

S = 1 ./ (1 + exp(-x));

figure;
plot(x, S, 'b-', 'LineWidth', 2);

grid on;

ylim([-0.1 1.1]);

xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$S(x)$', 'Interpreter', 'latex', 'FontSize', 12);

hold on;
yline(0, 'k--', 'LineWidth', 1);
yline(1, 'k--', 'LineWidth', 1);
hold off;