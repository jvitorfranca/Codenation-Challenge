import pandas as pd
import numpy as np

def process_data(df_train, df_test, df_answer):

	features = list(df_test)
	features.append('NU_NOTA_MT')

	df_answer['NU_INSCRICAO'] = df_test['NU_INSCRICAO']

	df_train = df_train[features]

	# Dropando tabelas que não serão utilizadas

	df_train.drop(['NU_INSCRICAO', 'SG_UF_RESIDENCIA', 'TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC', 'TP_PRESENCA_CH', 'Q027', 'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT'], 1, inplace=True)

	df_test.drop(['NU_INSCRICAO', 'SG_UF_RESIDENCIA', 'TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC', 'TP_PRESENCA_CH', 'Q027', 'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT'], 1, inplace=True)

	# Processado dados categóricos

	df_train['TP_SEXO'].replace('M', 0, inplace=True)
	df_train['TP_SEXO'].replace('F', 1, inplace=True)

	df_test['TP_SEXO'].replace('M', 0, inplace=True)
	df_test['TP_SEXO'].replace('F', 1, inplace=True)

	# Coloquei as notas como zero pq acredito que representam a nota dos alunos que faltaram.

	df_train['NU_NOTA_CN'].fillna(0, inplace=True)
	df_train['NU_NOTA_CH'].fillna(0, inplace=True)
	df_train['NU_NOTA_LC'].fillna(0, inplace=True)
	df_train['NU_NOTA_MT'].fillna(0, inplace=True)
	df_train['NU_NOTA_COMP1'].fillna(0, inplace=True)
	df_train['NU_NOTA_COMP2'].fillna(0, inplace=True)
	df_train['NU_NOTA_COMP3'].fillna(0, inplace=True)
	df_train['NU_NOTA_COMP4'].fillna(0, inplace=True)
	df_train['NU_NOTA_COMP5'].fillna(0, inplace=True)
	df_train['NU_NOTA_REDACAO'].fillna(0, inplace=True)

	df_test['NU_NOTA_CN'].fillna(0, inplace=True)
	df_test['NU_NOTA_CH'].fillna(0, inplace=True)
	df_test['NU_NOTA_LC'].fillna(0, inplace=True)
	df_test['NU_NOTA_COMP1'].fillna(0, inplace=True)
	df_test['NU_NOTA_COMP2'].fillna(0, inplace=True)
	df_test['NU_NOTA_COMP3'].fillna(0, inplace=True)
	df_test['NU_NOTA_COMP4'].fillna(0, inplace=True)
	df_test['NU_NOTA_COMP5'].fillna(0, inplace=True)
	df_test['NU_NOTA_REDACAO'].fillna(0, inplace=True)

	# Coloquei como 8 pq acredito que represente uma situação neutra

	df_train['TP_STATUS_REDACAO'].fillna(8, inplace=True)

	df_test['TP_STATUS_REDACAO'].fillna(8, inplace=True)

	# One_hot_enconding em variáveis categóricas

	df_train = pd.get_dummies(df_train, prefix='Q', columns=['Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'Q047'])

	df_test = pd.get_dummies(df_test, prefix='Q', columns=['Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'Q047'])

	return df_train, df_test, df_answer