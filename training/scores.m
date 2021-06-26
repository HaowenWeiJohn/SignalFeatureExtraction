function [F1,BA] = scores(groundTruth, prediction)
% n = size(groundTruth,2);
% FN = zeros(1,n);
% TP = zeros(1,n);
% FP = zeros(1,n);
% TN = zeros(1,n);
% TPR = zeros(1,n);
% TNR = zeros(1,n);
% prec = zeros(1,n);
% BA = zeros(1,n);
% F1 = zeros(1,n);
% for ii=1:n
%     [TP(ii), FP(ii), TN(ii), FN(ii)] = calError(groundTruth(:,ii),prediction(:,ii));
%     TPR(ii) = TP(ii)/(TP(ii)+FN(ii));
%     TNR(ii) = TN(ii)/(TN(ii)+FP(ii));
%     prec(ii) = TP(ii)/(TP(ii)+FP(ii));
%     BA(ii) = (TPR(ii)+TNR(ii))/2;
%     F1(ii) = (2*TPR(ii)*prec(ii))/(TPR(ii)+prec(ii));
% end
epsilon = 0.0001;
[TP, FP, TN, FN] = calError(groundTruth,prediction);
TPR = TP./(TP+FN+epsilon);
TNR = TN./(TN+FP+epsilon);
prec = TP./(TP+FP+epsilon);
BA = (TPR+TNR)./2;
F1 = (2.*TPR.*prec)./(TPR+prec+epsilon);

end