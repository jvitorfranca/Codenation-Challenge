import sys
import regressor as rg
import pandas as pd
import numpy as np
from utilities import process_data
from utilities import convert_to_zero
from utilities import ft_importance

# Argument 1: time (for auto-sklearn)

def main(argv):

	df_train = pd.read_csv('data/train.csv')
	df_test = pd.read_csv('data/test.csv')
	df_answer = pd.DataFrame()

	df_train, df_test, df_answer = process_data(df_train, df_test, df_answer)

	label = df_train['NU_NOTA_MT']

	df_train.drop(['NU_NOTA_MT'], axis=1, inplace=True)

	regression_model = rg.Regressor(df_train, df_test, df_answer, label)

	regression_model.auto_sklearn(time=int(sys.argv[1]))

	regression_model.prediction()

	regression_model.save_model('my_model')

	regression_model.save_answer('automl_answer')

	convert_to_zero('automl_answer')

if __name__ == "__main__":

	main(sys.argv)