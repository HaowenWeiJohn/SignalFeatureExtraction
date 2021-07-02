clear
%-----
% 7/2/2021 no Bluetooth
% 
load('train_acc_user11cf.mat') 
load('train_gyro_user11cf.mat') 
acc_train_range= acc_data_proc.ax_range;
gyro_train_range = gyro_data_proc.gyro_range;  

load('test_acc_user11cf.mat') 
load('test_gyro_user11cf.mat') 
acc_test_range = acc_data_proc.ax_range;
gyro_test_range = gyro_data_proc.gyro_range;

% load('../feature_data/test_acc_user11cff2.mat') 
load('../feature_data/all_train_features.mat')
load('../feature_data/all_test_features.mat')

% %randomly shufle the data
acc_dat_test = table(acc_x_test, acc_y_test, acc_z_test);
gyro_dat_test = table(gyro_x_test, gyro_y_test, gyro_z_test);

acc_dat_train = table(acc_x_train, acc_y_train, acc_z_train);
gyro_dat_train = table(gyro_x_train, gyro_y_train, gyro_z_train);

h = height(acc_dat_test);
idx = randperm(h);
acc_data_test_rand = acc_dat_test(idx,:);
gyro_data_test_rand = gyro_dat_test(idx,:);


%-------------------------------
h = height(acc_dat_train);
idx = randperm(h);
acc_data_train_rand = acc_dat_train(idx,:);
gyro_data_train_rand = gyro_dat_train(idx,:);

%accelerometer
acc_norm_train = featureNormalize2(acc_data_train_rand, "Zscale");
acc_norm_test = featureNormalize2(acc_data_test_rand, "Zscale");
  
%gyroscope
gyro_norm_train = featureNormalize2(gyro_data_train_rand, "Zscale");
gyro_norm_test = featureNormalize2(gyro_data_test_rand, "Zscale");

%----------------------------------------------------------------%
%regrassion on range 
% predNames = {'Mean', 'STD', 'M3', 'M4', '25%', '50%', '75%',...
%     'Value Entropy', 'Time Entropy', 'Autocorelation', 'Autocovariance',...
%     'X Mean', 'Y Mean', 'Z Mean', 'X STD', 'Y STD', 'Z STD', ...
%     'Autocorelation XY', 'Autocorelation XZ', 'Autocorelation YZ'};
% predNamesBT = {'RX Mean', 'TX Mean', 'Advertiser Time', 'Delta Mean',...
%     'Delta STD', 'M3', 'M4', '25%', '50%', '75%',...
%     'Value Entropy', 'Autocorelation', 'RX STD', 'TX STD'};
predNames = {'Mean', 'STD', 'M3', 'M4', '25%', '50%', '75%',...
    'Value Entropy', 'Time Entropy', 'Autocorelation', 'Autocovariance',...
    'X Mean', 'Y Mean', 'Z Mean', 'X STD', 'Y STD', 'Z STD', ...
    'Autocorelation XY', 'Autocorelation XZ', 'Autocorelation YZ', ...
    'Environment', 'Room Size', 'Location in Room',...
    'Pose', 'On body location'};
% predNamesBT = {'RX Mean', 'TX Mean', 'Advertiser Time','Delta Mean',...
%     'Delta STD', 'M3', 'M4', '25%', '50%', '75%',...
%     'Value Entropy', 'Autocorelation', 'Autocovariance',...
%     'RX STD', 'TX STD', 'Yp2p', 'Rayleigh', ... 
%     'Fade Duration', 'Level Crossing',...
%     'Energy', 'Laplacian Best Fit', 'RMS Doppler', ...
%     'Laplacian Fit2', 'Gaussian Fit', 'Polynomial Fit', 'Doppler Peak',...
%     'Doppler Mean',...
%     'Environment', 'Room Size', 'Location in Room',...
%     'Pose', 'On body location'}; 
predNamesBT = {'RX Mean', 'TX Mean',  'Advertiser Time','Delta Mean',...
    'Delta STD', 'M3', 'M4', '25%', '50%', '75%',...
    'Value Entropy', 'Autocorelation', 'Autocovariance',...
    'RX STD', 'TX STD', 'Yp2p', 'Rayleigh', ... 
    'Fade Duration', 'Level Crossing',...
    'Energy',  'RMS Doppler', ...
    'Doppler Peak',...
    'Doppler Mean',...
    'AT mean', 'AT std', 'AT 25%', 'AT 50%', 'AT75%', 'AT_M3', 'AT_M4', ... 
    'Environment', 'Room Size', 'Location in Room',...
    'Pose', 'On body location'}; % 

%range train  
t2 = templateTree('MinLeafSize', 1); 
Mdl_acc_reg = fitrensemble(acc_norm_train, acc_train_range, ...
    'Method','Bag','NumLearningCycles', 265,'Learners',t2 ); %try 265 Nlerncycles , 'PredictorNames', predNames
%fitted_acc= kfoldPredict(Mdl_acc_reg); ,  'PredictorNames', predNames

Mdl_gyro_reg = fitrensemble(gyro_norm_train, gyro_train_range, ...
    'Method','Bag','NumLearningCycles',265,'Learners',t2); %try 265 Nlerncycles , 'PredictorNames', predNames
%, 'PredictorNames', predNames 
 
%----------------------------------------------------------------%
%test
fitted_acc = predict(Mdl_acc_reg, acc_norm_test);
fitted_gyro = predict(Mdl_gyro_reg, gyro_norm_test);

%----------------------------------------------------------------%
[mse_acc, rmse_acc, r2_acc, mae_acc] = fit_error(acc_test_range, fitted_acc);
[mse_gyro, rmse_gyro, r2_gyro, mae_gyro] = fit_error(gyro_test_range, fitted_gyro);


methods = ["MSE" "RMSE" "R2" "MAE"];
results_reg = table(methods', [mse_acc rmse_acc r2_acc mae_acc]',...
    [mse_gyro rmse_gyro r2_gyro mae_gyro]');

writetable(results_reg, 'results_pact_LOO_manyFeat_user11.xlsx');

[F1_acc, BA_acc] = scores((acc_test_range >= 6),(fitted_acc >= 6));
[F1_gyro, BA_gyro] = scores((gyro_test_range >= 6),(fitted_gyro >= 6));


b_methods = ["F1" "BA"];

results_b = table(b_methods', [F1_acc BA_acc]', [F1_gyro BA_gyro]',...
   'VariableNames',{'Metric' 'Accelerometer' 'Gyroscope'});

writetable(results_b, 'results_pact_LOO_manyFeat_6feetBest_user11.xlsx');

