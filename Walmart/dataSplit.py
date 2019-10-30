import numpy as np;
import pandas as pd

import os
from sklearn.model_selection import train_test_split

print(os.listdir("./data"))
# ['sampleSubmission.csv', 'test.csv', 'train.csv', 'features.csv', 'stores.csv']


train = pd.read_csv('./data/train_old.csv')
features = pd.read_csv('./data/features.csv')
stores = pd.read_csv('./data/stores.csv')

print(train.size)
print(train.head())

print(train.columns)
# Index(['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday'], dtype='object')

X = pd.DataFrame(train, columns=['Store', 'Dept', 'Date', 'IsHoliday'])
y = pd.DataFrame(train, columns=['Weekly_Sales'])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=X['Store'])

train_new = pd.concat([X_train, y_train], axis=1)
train_new.sort_index(inplace=True)


test_new = pd.concat([X_test, y_test], axis=1)
test_new.sort_index(inplace=True)
test_new['Id'] = test_new['Store'].map(str) + '_' + test_new['Dept'].map(str) + '_' + test_new['Date'].map(str)

test_final = pd.DataFrame(test_new, columns=['Store', 'Dept', 'Date', 'IsHoliday'])
true_label = pd.DataFrame(test_new, columns=['Id', 'Weekly_Sales', 'IsHoliday'])

sample_submission = true_label.copy()
sample_submission['Weekly_Sales'] = 0


print(train_new.size)
print(test_new.size)

output_path = './data_new/'
train_new.to_csv(output_path + 'train.csv', index=False)

cols_test = ['Store', 'Dept', 'Date', 'IsHoliday']
test_new[cols_test].to_csv(output_path + 'test.csv', index=False)

true_label.to_csv(output_path + 'correct_submission.csv', index=False)

col_sample = ['Id', 'Weekly_Sales']
sample_submission[col_sample].to_csv(output_path + 'sample_submission.csv', index=False)









