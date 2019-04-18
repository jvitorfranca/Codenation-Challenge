from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

import autosklearn.regression as asc
import sklearn as sk
import pandas as pd
import numpy as np
import pickle

from utilities import process_data
from utilities import convert_to_zero
from utilities import ft_importance

def main(time, df_train, label, df_test, df_answer):

	classifier = asc.AutoSklearnRegressor(
	    time_left_for_this_task=time+10,
	    per_run_time_limit=time,
	    initial_configurations_via_metalearning=0,
	    resampling_strategy='cv',
	    resampling_strategy_arguments={'folds': 5},
	)

	classifier.fit(df_train.copy(), label.copy())

	classifier.refit(df_train.copy(), label.copy())

	pickle.dump(classifier, open('regressor.pkl', 'wb'))

	predictions = classifier.predict(df_test)

	df_answer['NU_NOTA_MT'] = np.around(predictions,2)

	df_answer.to_csv('automl_answer.csv', index=False, header=True)

def random_search(df_train, label, df_test, df_answer):

	# Number of trees in random forest
	n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
	# Number of features to consider at every split
	max_features = ['auto', 'sqrt']
	# Maximum number of levels in tree
	max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth.append(None)
	# Minimum number of samples required to split a node
	min_samples_split = [2, 5, 10]
	# Minimum number of samples required at each leaf node
	min_samples_leaf = [1, 2, 4]
	# Method of selecting samples for training each tree
	bootstrap = [True, False]
	# Create the random grid
	random_grid = {'n_estimators': n_estimators,
	               'max_features': max_features,
	               'max_depth': max_depth,
	               'min_samples_split': min_samples_split,
	               'min_samples_leaf': min_samples_leaf,
	               'bootstrap': bootstrap}
	print(random_grid)

	# Use the random grid to search for best hyperparameters
	# First create the base model to tune
	rf = RandomForestRegressor()
	# Random search of parameters, using 3 fold cross validation, 
	# search across 100 different combinations, and use all available cores
	rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=3, random_state=42, n_jobs = -2)

	# Fit the random search model
	rf_random.fit(df_train, label)

	bestRF = rf_random.best_estimator_
	
	print(bestRF)
	
	predictions = bestRF.predict(df_test)

	df_answer['NU_NOTA_MT'] = np.around(predictions,2)

	df_answer.to_csv('regressor_answer.csv', index=False, header=True)

if __name__ == "__main__":
	
	df_train = pd.read_csv('data/train.csv')
	df_test = pd.read_csv('data/test.csv')
	df_answer = pd.DataFrame()

	df_train, df_test, df_answer = process_data(df_train, df_test, df_answer)

	label = df_train['NU_NOTA_MT']

	df_train.drop(['NU_NOTA_MT'], axis=1, inplace=True)

	df = ft_importance(k=87)

	features = df['Specs'].tolist()[:50]

	df_train = df_train[features]

	df_test = df_test[features]

	main(60*60+60*30, df_train, label, df_test, df_answer)

	data = convert_to_zero('automl_answer')