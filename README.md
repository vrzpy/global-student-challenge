
# Python Code for Model Complete

The following is the readme file explaining how to run the code.

## Importing Libraries

```
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import FastICA
```
The above-mentioned commands import necessary modules from pandas and sklearn libraries.

## Reading the file

```
train_data = pd.read_csv('./train_data.csv', index_col="job_id")
test_data = pd.read_csv("./test_data_unlabeled.csv", index_col="job_id")
```
Read the train data and test data from the train_data.csv and test_data_unlabeled files respectively through the command ```pd.read_csv```.

## Defining X and Y
```
X = train_data[['memory_GB', 'network_log10_MBps',
          'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]  
Y = train_data['failed']
test_X = test_data[['memory_GB', 'network_log10_MBps',
                   'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]

train_x, test_x, train_y,test_y = train_test_split(X, Y, train_size=0.5, test_size=0.5, random_state=0)
```

Define the variables X and Y by taking the resources provided in the training data. Split the train and test data using ```train_test_split``` into equal sizes of 0.5.

## Feature Engineering
```
fica = FastICA(n_components=4, random_state=0)
train_x = fica.fit_transform(train_x)
test_x = fica.transform(test_x)
```

Transform the train and test data using FastICA, by taking all four components using feature engineering.  

## Model : DecisionTreeClassifier 
```
myModel = DecisionTreeClassifier(max_depth=None, min_samples_split=2)
myModel.fit(X, Y)
predicted_y = myModel.predict(test_X)
```

Use the DecisionTreeClassifier with parameters max_depth=None, min_samples_split=2 to fit the values of X and Y, & predict the test data.

## Confusion Matrix

```
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(test_y, predicted_y)
print(cm)
```
Use the sklearn module ```confusion_matrix``` to print the confusion matrix.

## Balanced Accuracy

```
from sklearn.metrics import balanced_accuracy_score
print(balanced_accuracy_score(test_y, predicted_y))
```
Use the sklearn module ```balanced_accuracy_score``` to print the balanced accuracy.

## Writing to CSV (Output)

```
output = pd.DataFrame({'job_id': test_X.index,
                       'failed': predicted_y})
output.to_csv('model_complete_test.csv', index=False)
```

Use the pandas commands ```Dataframe``` and ```.to_csv``` to output the columns ```job_id``` and ```failed``` based on the model predictions to the file model_complete_test.csv.


