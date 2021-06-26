function [mse, rmse, r2, mae] = fit_error(tru, fitted)
s = sum((tru-fitted).^2);
n = length(tru);

mse = s/n;

rmse = sqrt(s/n);

r2 = 1 - (s/sum((tru-mean(tru)).^2));

mae = sum(abs(fitted-tru))/n;

end