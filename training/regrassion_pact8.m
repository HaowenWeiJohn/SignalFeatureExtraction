clear
%-----
% 6/26/21 modified to work with large feature set
% 
load('train_acc_user11cff2.mat') 
load('train_gyro_user11cff2.mat') 
load('train_bt_user11cff2.mat') 

load('test_acc_user11cff2.mat') 
load('test_gyro_user11cff2.mat') 
load('test_bt_user11cff2.mat') 
% %randomly shufle the data
h = height(acc_dat_test);
idx = randperm(h);
acc_data_test_rand = acc_dat_test(idx,:);
gyro_data_test_rand = gyro_dat_test(idx,:);

h1 = height(bt_dat_test);
idx1 = randperm(h1);
bt_data_test_rand = bt_dat_test(idx1,:);
%-------------------------------
h = height(acc_dat_train);
idx = randperm(h);
acc_data_train_rand = acc_dat_train(idx,:);
gyro_data_train_rand = gyro_dat_train(idx,:);

h1 = height(bt_dat_train);
idx1 = randperm(h1);
bt_data_train_rand = bt_dat_train(idx1,:);

%accelerometer
acc_norm_train = featureNormalize2(acc_data_train_rand.acc_stats, "Zscale");
acc_norm_test = featureNormalize2(acc_data_test_rand.acc_stats, "Zscale");

acc_train_range= acc_data_train_rand.acc_range;
acc_test_range= acc_data_test_rand.acc_range;   
%gyroscope
gyro_norm_train = featureNormalize2(gyro_data_train_rand.gyro_stats, "Zscale");
gyro_norm_test = featureNormalize2(gyro_data_test_rand.gyro_stats, "Zscale");

gyro_train_range = gyro_data_train_rand.gyro_range;
gyro_test_range = gyro_data_test_rand.gyro_range;  
%bluetooth
bt_norm_train = featureNormalize2(bt_data_train_rand.bt_stats, "Zscale"); %(:,1:23)
bt_norm_test = featureNormalize2(bt_data_test_rand.bt_stats, "Zscale"); %(:,1:23)

bt_train_range = bt_data_train_rand.bt_range;
bt_test_range = bt_data_test_rand.bt_range;
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
 
Mdl_bt_reg = fitrtree(bt_norm_train, bt_train_range,'MinLeafSize', 7, ...
    'Surrogate', 'off'); %, 'PredictorNames', predNamesBT
%----------------------------------------------------------------%
%test
fitted_acc = predict(Mdl_acc_reg, acc_norm_test);
fitted_gyro = predict(Mdl_gyro_reg, gyro_norm_test);
fitted_bt = predict(Mdl_bt_reg, bt_norm_test);
%----------------------------------------------------------------%
[mse_acc, rmse_acc, r2_acc, mae_acc] = fit_error(acc_test_range, fitted_acc);
[mse_gyro, rmse_gyro, r2_gyro, mae_gyro] = fit_error(gyro_test_range, fitted_gyro);
[mse_bt, rmse_bt, r2_bt, mae_bt] = fit_error(bt_test_range, fitted_bt);

methods = ["MSE" "RMSE" "R2" "MAE"];
results_reg = table(methods', [mse_acc rmse_acc r2_acc mae_acc]',...
    [mse_gyro rmse_gyro r2_gyro mae_gyro]',...
    [mse_bt rmse_bt r2_bt mae_bt]');

writetable(results_reg, 'results_pact_LOO_manyFeat_user11.xlsx');

[F1_acc, BA_acc] = scores((acc_test_range >= 6),(fitted_acc >= 6));
[F1_gyro, BA_gyro] = scores((gyro_test_range >= 6),(fitted_gyro >= 6));
[F1_bt, BA_bt] = scores((bt_test_range >= 6),(fitted_bt >= 6));

b_methods = ["F1" "BA"];

results_b = table(b_methods', [F1_acc BA_acc]', [F1_gyro BA_gyro]',...
    [F1_bt BA_bt]','VariableNames',{'Metric' 'Accelerometer' 'Gyroscope' 'Bluetooth'});

writetable(results_b, 'results_pact_LOO_manyFeat_6feetBest_user11.xlsx');

