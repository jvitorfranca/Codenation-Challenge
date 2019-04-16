import autosklearn.regression as asc
import sklearn as sk
import pandas as pd
import numpy as np
import pickle

from utilities import process_data

def main(time=360):

	df_train = pd.read_csv('data/train.csv')
	df_test = pd.read_csv('data/test.csv')
	df_answer = pd.DataFrame()

	df_train, df_test, df_answer = process_data(df_train, df_test, df_answer)

	label = df_train['NU_NOTA_MT']

	df_train.drop(['NU_NOTA_MT'], axis=1, inplace=True)

	classifier = asc.AutoSklearnRegressor(
	    time_left_for_this_task=time+10,
	    per_run_time_limit=time,
	    initial_configurations_via_metalearning=0,
	    resampling_strategy='cv',
	    resampling_strategy_arguments={'folds': 5},
	)

	classifier.fit(df_train.copy(), label.copy())

	classifier.refit(df_train.copy(), label.copy())

	pickle.dump(classifier, open('models/regressor.pkl', 'wb'))

	predictions = classifier.predict(df_test)

	df_answer['NU_NOTA_MT'] = np.around(predictions,2)

	df_answer.to_csv('data/answer.csv', index=False, header=True)

if __name__ == "__main__":
	main(time=5400)