import autosklearn.regression as asc
import sklearn as sk
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


class Regressor:
	def __init__(self, df_train, df_test, df_answer, label):
		self.df_train = df_train
		self.df_test = df_test
		self.df_answer = df_answer
		self.label = label
		self.model = None
		self.predictions = None

	def auto_sklearn(self, time=60, estimators=None, resampling_strategy='cv'):

		if self.model is None:

			self.model = asc.AutoSklearnRegressor(
				time_left_for_this_task=time+10,
				per_run_time_limit=time,
				initial_configurations_via_metalearning=0,
				include_estimators=estimators,
				resampling_strategy=resampling_strategy,
				resampling_strategy_arguments={'folds': 5}
			)

			self.model.fit(self.df_train.copy(), self.label.copy())

			if resampling_strategy == 'cv':

				self.model.refit(self.df_train.copy(), self.label.copy())

	def tpot_regressor(self, generations=5, population_size=10):

		if self.model is None:

			self.model = tpot.TPOTRegressor(verbosity=3,
                      random_state=25,
                      n_jobs=2,
                      generations=8,
                      population_size=40,
                      early_stop = 5,
                      memory = None)

			self.model.fit(self.df_train.copy(), self.label.copy())

			self.model.export('tpot_ames.py')

	def random_search(self):

		if self.model is None:

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

			random_grid = {'n_estimators': n_estimators,
			               'max_features': max_features,
			               'max_depth': max_depth,
			               'min_samples_split': min_samples_split,
			               'min_samples_leaf': min_samples_leaf,
			               'bootstrap': bootstrap}

			rf = RandomForestRegressor()

			self.model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=3, random_state=42, n_jobs = -2)

			self.model.fit(self.df_train, self.label)

			bestRF = self.model.best_estimator_
		
			return random_grid, bestRF

	def prediction(self):

		if self.model is not None:

			self.predictions = self.model.predict(self.df_test)


	def save_answer(self, filename='no_processed_answer'):

		if self.predictions is not None:

			self.df_answer[self.label.name] = np.around(self.predictions, 2)

			self.df_answer.to_csv(filename+'.csv', index=False, header=True)

	def save_model(self, filename='regressor'):

		if self.model is not None:

			pickle.dump(self.model, open(filename+'.pkl', 'wb'))
