import pandas as pd
import numpy as np

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_resposta = pd.DataFrame()

df_train.drop(["Unnamed: 0", "NU_INSCRICAO", "NU_ANO", "NO_MUNICIPIO_RESIDENCIA", "SG_UF_RESIDENCIA", "NO_MUNICIPIO_NASCIMENTO",
			   "SG_UF_NASCIMENTO", "NO_MUNICIPIO_ESC", "SG_UF_ESC", "TX_RESPOSTAS_CN", "TX_RESPOSTAS_CH", "TX_RESPOSTAS_LC",
			   "TX_RESPOSTAS_MT", "TX_GABARITO_CN", "TX_GABARITO_CH", "TX_GABARITO_LC", "TX_GABARITO_MT", "Q041", "TP_ENSINO"], 1, inplace=True)

df_train['TP_SEXO'].replace('M', 0, inplace=True)
df_train['TP_SEXO'].replace('F', 1, inplace=True)

df_train['NU_NOTA_CN'].fillna(df_train['NU_NOTA_CN'].mean(), inplace=True)
df_train['NU_NOTA_CH'].fillna(df_train['NU_NOTA_CH'].mean(), inplace=True)
df_train['NU_NOTA_REDACAO'].fillna(df_train['NU_NOTA_REDACAO'].mean(), inplace=True)
df_train['NU_NOTA_LC'].fillna(df_train['NU_NOTA_LC'].mean(), inplace=True)
df_train['NU_NOTA_MT'].fillna(df_train['NU_NOTA_MT'].mean(), inplace=True)
df_test['NU_NOTA_CN'].fillna(df_train['NU_NOTA_CN'].mean(), inplace=True)
df_test['NU_NOTA_CH'].fillna(df_train['NU_NOTA_CH'].mean(), inplace=True)
df_test['NU_NOTA_REDACAO'].fillna(df_train['NU_NOTA_REDACAO'].mean(), inplace=True)
df_test['NU_NOTA_LC'].fillna(df_train['NU_NOTA_LC'].mean(), inplace=True)