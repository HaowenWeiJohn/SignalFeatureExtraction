
function X_norm = featureNormalize2(X, method)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
epsilon = 0.0001;
if method == "Zscale"
    temp = bsxfun(@minus, X, mean(X));
    X_norm = bsxfun(@rdivide, temp, (std(X)+epsilon));
elseif method == "MinMax"
     X_norm = (X - min(X))./(max(X) - min(X));
elseif method == "Log"
    X_norm = log10(X);
else
    disp('No scaling method selected')
end
% ============================================================

end
