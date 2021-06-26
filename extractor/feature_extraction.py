import tsfel
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import pickle
import numpy, scipy.io


acc_data = np.loadtxt(open("../original_data/acc_data.csv", "rb"), delimiter=",", skiprows=1)
gyro_data = np.loadtxt(open("../original_data/gyro_data.csv", "rb"), delimiter=",", skiprows=1)
bt_data = np.loadtxt(open("../original_data/bt_data.csv", "rb"), delimiter=",", skiprows=1)

data_acc_test = pd.DataFrame(acc_data[:,0:3], columns=["acc_x", "acc_y", "acc_z"])
data_gyro_test = pd.DataFrame(gyro_data[:,0:3], columns=["gyro_x", "gyro_y", "gyro_z"])
data_bt_test = pd.DataFrame(bt_data[:,1:4], columns=["bt_rx", "bt_tx", "bt_time"])


acc_x_test = extract_features(data_acc_test, column_id="acc_x") #, column_id="id", column_sort="time"
acc_y_test = extract_features(data_acc_test, column_id="acc_y")
acc_z_test = extract_features(data_acc_test, column_id="acc_z")
impute(acc_x_test)
impute(acc_y_test)
impute(acc_z_test)


gyro_x_test = extract_features(data_gyro_test, column_id="gyro_x")
gyro_y_test = extract_features(data_gyro_test, column_id="gyro_y")
gyro_z_test = extract_features(data_gyro_test, column_id="gyro_z")
impute(gyro_x_test)
impute(gyro_y_test)
impute(gyro_z_test)

bt_rx_test = extract_features(data_bt_test, column_id="bt_rx")
bt_tx_test = extract_features(data_bt_test, column_id="bt_tx")
bt_time_test = extract_features(data_bt_test, column_id="bt_time")
impute(bt_rx_test)
impute(bt_tx_test)
impute(bt_time_test)


# test_mdict={ 'acc_x_test': np.array(acc_x_test),
#         'acc_y_test': np.array(acc_y_test),
#         'acc_z_test': np.array(acc_z_test),
#         'gyro_x_test': np.array(gyro_x_test),
#         'gyro_y_test': np.array(gyro_y_test),
#         'gyro_z_test': np.array(gyro_z_test),
#         'bt_rx_test': np.array(bt_rx_test),
#         'bt_tx_test': np.array(bt_tx_test),
#         'bt_time_test': np.array(bt_time_test)
#        }

# scipy.io.savemat('../feature_data/all_test_features.mat', test_mdict, long_field_names=True)


test_pdict={ 'acc_x_test': acc_x_test,
        'acc_y_test': acc_y_test,
        'acc_z_test': acc_z_test,
        'gyro_x_test': gyro_x_test,
        'gyro_y_test': gyro_y_test,
        'gyro_z_test': gyro_z_test,
        'bt_rx_test': bt_rx_test,
        'bt_tx_test': bt_tx_test,
        'bt_time_test': bt_time_test
       }
with open('../feature_data/all_test_features.pickle', 'wb') as handle:
    pickle.dump(test_pdict, handle, protocol=4)




acc_data_train = np.loadtxt(open("../original_data/acc_data_train.csv", "rb"), delimiter=",", skiprows=1)
gyro_data_train = np.loadtxt(open("../original_data/gyro_data_train.csv", "rb"), delimiter=",", skiprows=1)
bt_data_train = np.loadtxt(open("../original_data/bt_data_train.csv", "rb"), delimiter=",", skiprows=1)

data_acc_train = pd.DataFrame(acc_data_train[:,0:3], columns=["acc_x", "acc_y", "acc_z"])
data_gyro_train = pd.DataFrame(gyro_data_train[:,0:3], columns=["gyro_x", "gyro_y", "gyro_z"])
data_bt_train = pd.DataFrame(bt_data_train[:,1:4], columns=["bt_rx", "bt_tx", "bt_time"])


acc_x_train = extract_features(data_acc_train, column_id="acc_x") #, column_id="id", column_sort="time"
acc_y_train = extract_features(data_acc_train, column_id="acc_y")
acc_z_train = extract_features(data_acc_train, column_id="acc_z")
impute(acc_x_train)
impute(acc_y_train)
impute(acc_z_train)

gyro_x_train = extract_features(data_gyro_train, column_id="gyro_x")
gyro_y_train = extract_features(data_gyro_train, column_id="gyro_y")
gyro_z_train = extract_features(data_gyro_train, column_id="gyro_z")
impute(gyro_x_train)
impute(gyro_y_train)
impute(gyro_z_train)

bt_rx_train = extract_features(data_bt_train, column_id="bt_rx")
bt_tx_train = extract_features(data_bt_train, column_id="bt_tx")
bt_time_train = extract_features(data_bt_train, column_id="bt_time")
impute(bt_rx_train)
impute(bt_tx_train)
impute(bt_time_train)



# train_mdict={ 'acc_x_train': np.array(acc_x_train),
#         'acc_y_train': np.array(acc_y_train),
#         'acc_z_train': np.array(acc_z_train),
#         'gyro_x_train': np.array(gyro_x_train),
#         'gyro_y_train': np.array(gyro_y_train),
#         'gyro_z_train': np.array(gyro_z_train),
#         'bt_rx_train': np.array(bt_rx_train),
#         'bt_tx_train': np.array(bt_tx_train),
#         'bt_time_train': np.array(bt_time_train)
#        }

# scipy.io.savemat('../feature_data/all_train_features.mat', train_mdict, long_field_names=True)


train_pdict={ 'acc_x_train': acc_x_train,
        'acc_y_train': acc_y_train,
        'acc_z_train': acc_z_train,
        'gyro_x_train': gyro_x_train,
        'gyro_y_train': gyro_y_train,
        'gyro_z_train': gyro_z_train,
        'bt_rx_train': bt_rx_train,
        'bt_tx_train': bt_tx_train,
        'bt_time_train': bt_time_train
       }

with open('../feature_data/all_train_features.pickle', 'wb') as handle:
    pickle.dump(train_pdict, handle, protocol=4)



