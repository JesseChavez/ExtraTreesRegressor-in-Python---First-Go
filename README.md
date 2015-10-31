# ExtraTreesRegressor in Python, First Go

* Data source here: http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity
* Target column (what want to predict) is last column "shares"
* Removed url column and saved as new file
* This code is a work in progress (I'm new to Python)


```python

import csv
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import train_test_split

def import_dataset(filename, all_data):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;with open(filename,'rb') as csvfile:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;csv_has_header = csv.Sniffer().has_header(csvfile.read(10*1024))
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;csvfile.seek(0)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if csv_has_header:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;csvfile.next()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;csvlines = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dataset=list(csvlines)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for x in range (len(dataset)):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;all_data.append(dataset[x])

def export_dataset(filename, prediction_result):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;with open(filename, 'wb') as output:
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mywriter = csv.writer(output, lineterminator='\n')
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for val in prediction_result:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mywriter.writerow(val)

def main():
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;all_data = []
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;split = 0.30
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;export_file_predictions = '/home/becky/Documents/OnlineNewsPopularity_predictions.csv'
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;import_dataset('/home/becky/Documents/OnlineNewsPopularity_fixed_nourl.csv', all_data)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dataset = np.array(all_data)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;training_data, test_data = train_test_split(dataset, test_size = split)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;regression_task = ExtraTreesRegressor(n_estimators=15, max_features="auto", max_depth=None, min_samples_split=2)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;regression_task.fit(training_data, training_data[:,59], sample_weight=None)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;predicted_values = regression_task.predict(test_data)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;actual_and_test = np.column_stack([test_data, predicted_values])
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;export_dataset(export_file_predictions, actual_and_test)

main()

```