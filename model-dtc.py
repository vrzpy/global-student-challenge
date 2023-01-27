import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import FastICA

# Reading the file
train_data = pd.read_csv('./train_data.csv', index_col="job_id")
test_data = pd.read_csv("./test_data_unlabeled.csv", index_col="job_id")

# Defining X and Y
X = train_data[['memory_GB', 'network_log10_MBps',
          'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]  
Y = train_data['failed']
test_X = test_data[['memory_GB', 'network_log10_MBps',
                   'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]

train_x, test_x, train_y,test_y = train_test_split(X, Y, train_size=0.5, test_size=0.5, random_state=0)

#FastICA for feature engineering
fica = FastICA(n_components=4, random_state=0)
train_x = fica.fit_transform(train_x)
test_x = fica.transform(test_x)

# Model
myModel = DecisionTreeClassifier(max_depth=None, min_samples_split=2)
myModel.fit(X, Y)
predicted_y = myModel.predict(test_X)

#Confusion Matrix
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(test_y, predicted_y)
print(cm)

#Absolute Error
print(
    mean_absolute_error(test_y, predicted_y))

# Balanced Accuracy
from sklearn.metrics import balanced_accuracy_score
print(balanced_accuracy_score(test_y, predicted_y))

# Writing to csv
output = pd.DataFrame({'job_id': test_X.index,
                       'failed': predicted_y})
output.to_csv('model_complete_test.csv', index=False)

