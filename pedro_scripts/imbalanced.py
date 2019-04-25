from sklearn.ensemble import RandomForestRegressor

from imblearn.under_sampling import TomekLinks
import sklearn as sk
import pandas as pd
import numpy as np
import pickle

from utilities import convert_to_zero
from utilities import ft_importance


def p_data(df_train, df_test, true_test, df_answer):

    features = list(true_test)
    features.append('NU_NOTA_MT')

    df_answer['NU_INSCRICAO'] = df_test['NU_INSCRICAO']

    df_train = df_train[features]

    # Dropando tabelas que não serão utilizadas

    df_train.drop(['NU_INSCRICAO', 'SG_UF_RESIDENCIA', 'TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC', 'TP_PRESENCA_CH',
                   'Q027', 'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT'], 1, inplace=True)

    df_test.drop(['NU_INSCRICAO', 'SG_UF_RESIDENCIA', 'TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC', 'TP_PRESENCA_CH',
                  'Q027', 'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT'], 1, inplace=True)

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
    df_test['NU_NOTA_MT'].fillna(0, inplace=True)
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

    df_train = pd.get_dummies(df_train, prefix=['Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'Q047'], columns=[
                              'Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'Q047'])

    df_test = pd.get_dummies(df_test, prefix=['Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'Q047'], columns=[
                             'Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'Q047'])

    # Utilizando Estatística Descritiva

    notas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO']

    df_train['KURTOSIS_NOTAS'] = df_train[notas].kurtosis(axis=1)
    df_train['MEAN_NOTAS'] = df_train[notas].mean(axis=1)
    df_train['MEDIAN_NOTAS'] = df_train[notas].median(axis=1)
    df_train['MAD_NOTAS'] = df_train[notas].mad(axis=1)
    df_train['QUANTILE_NOTAS'] = df_train[notas].quantile(axis=1)
    df_train['SEM_NOTAS'] = df_train[notas].sem(axis=1)
    df_train['SKEW_NOTAS'] = df_train[notas].skew(axis=1)
    df_train['STD_NOTAS'] = df_train[notas].std(axis=1)
    df_train['VAR_NOTAS'] = df_train[notas].var(axis=1)
    df_train['AMP_NOTAS'] = df_train[notas].max(
        axis=1) - df_train[notas].min(axis=1)

    df_train['MEAN_NOTA_CH_LC'] = df_train[[
        'NU_NOTA_CH', 'NU_NOTA_LC']].mean(axis=1)
    df_train['MEAN_NOTA_CH_CN'] = df_train[[
        'NU_NOTA_CH', 'NU_NOTA_CN']].mean(axis=1)
    df_train['MEAN_NOTA_CH_REDACAO'] = df_train[[
        'NU_NOTA_CH', 'NU_NOTA_REDACAO']].mean(axis=1)
    df_train['MEAN_NOTA_CN_LC'] = df_train[[
        'NU_NOTA_CN', 'NU_NOTA_LC']].mean(axis=1)
    df_train['MEAN_NOTA_CN_REDACAO'] = df_train[[
        'NU_NOTA_CN', 'NU_NOTA_REDACAO']].mean(axis=1)
    df_train['MEAN_NOTA_LC_REDACAO'] = df_train[[
        'NU_NOTA_LC', 'NU_NOTA_REDACAO']].mean(axis=1)

    notas_red = ['NU_NOTA_COMP1', 'NU_NOTA_COMP2',
                 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5']

    df_train['KURTOSIS_NOTAS_COMP'] = df_train[notas_red].kurtosis(axis=1)
    df_train['MEAN_NOTAS_COMP'] = df_train[notas_red].mean(axis=1)
    df_train['MEDIAN_NOTAS_COMP'] = df_train[notas_red].median(axis=1)
    df_train['MAD_NOTAS_COMP'] = df_train[notas_red].mad(axis=1)
    df_train['QUANTILE_NOTAS_COMP'] = df_train[notas_red].quantile(axis=1)
    df_train['SEM_NOTAS_COMP'] = df_train[notas_red].sem(axis=1)
    df_train['STD_NOTAS_COMP'] = df_train[notas_red].std(axis=1)
    df_train['VAR_NOTAS_COMP'] = df_train[notas_red].var(axis=1)
    df_train['AMP_NOTAS_COMP'] = df_train[notas_red].max(
        axis=1) - df_train[notas_red].min(axis=1)

    df_train['MEAN_NOTA_COMP1_COMP2'] = df_train[[
        'NU_NOTA_COMP1', 'NU_NOTA_COMP2']].mean(axis=1)
    df_train['MEAN_NOTA_COMP1_COMP3'] = df_train[[
        'NU_NOTA_COMP1', 'NU_NOTA_COMP3']].mean(axis=1)
    df_train['MEAN_NOTA_COMP1_COMP4'] = df_train[[
        'NU_NOTA_COMP1', 'NU_NOTA_COMP4']].mean(axis=1)
    df_train['MEAN_NOTA_COMP1_COMP5'] = df_train[[
        'NU_NOTA_COMP1', 'NU_NOTA_COMP5']].mean(axis=1)
    df_train['MEAN_NOTA_COMP2_COMP3'] = df_train[[
        'NU_NOTA_COMP2', 'NU_NOTA_COMP3']].mean(axis=1)
    df_train['MEAN_NOTA_COMP2_COMP4'] = df_train[[
        'NU_NOTA_COMP2', 'NU_NOTA_COMP4']].mean(axis=1)
    df_train['MEAN_NOTA_COMP2_COMP5'] = df_train[[
        'NU_NOTA_COMP2', 'NU_NOTA_COMP5']].mean(axis=1)
    df_train['MEAN_NOTA_COMP3_COMP4'] = df_train[[
        'NU_NOTA_COMP3', 'NU_NOTA_COMP4']].mean(axis=1)
    df_train['MEAN_NOTA_COMP3_COMP5'] = df_train[[
        'NU_NOTA_COMP3', 'NU_NOTA_COMP5']].mean(axis=1)
    df_train['MEAN_NOTA_COMP4_COMP5'] = df_train[[
        'NU_NOTA_COMP4', 'NU_NOTA_COMP5']].mean(axis=1)

    df_test['KURTOSIS_NOTAS'] = df_test[notas].kurtosis(axis=1)
    df_test['MEAN_NOTAS'] = df_test[notas].mean(axis=1)
    df_test['MEDIAN_NOTAS'] = df_test[notas].median(axis=1)
    df_test['MAD_NOTAS'] = df_test[notas].mad(axis=1)
    df_test['QUANTILE_NOTAS'] = df_test[notas].quantile(axis=1)
    df_test['SEM_NOTAS'] = df_test[notas].sem(axis=1)
    df_test['SKEW_NOTAS'] = df_test[notas].skew(axis=1)
    df_test['STD_NOTAS'] = df_test[notas].std(axis=1)
    df_test['VAR_NOTAS'] = df_test[notas].var(axis=1)
    df_test['AMP_NOTAS'] = df_test[notas].max(
        axis=1) - df_test[notas].min(axis=1)

    df_test['MEAN_NOTA_CH_LC'] = df_test[[
        'NU_NOTA_CH', 'NU_NOTA_LC']].mean(axis=1)
    df_test['MEAN_NOTA_CH_CN'] = df_test[[
        'NU_NOTA_CH', 'NU_NOTA_CN']].mean(axis=1)
    df_test['MEAN_NOTA_CH_REDACAO'] = df_test[[
        'NU_NOTA_CH', 'NU_NOTA_REDACAO']].mean(axis=1)
    df_test['MEAN_NOTA_CN_LC'] = df_test[[
        'NU_NOTA_CN', 'NU_NOTA_LC']].mean(axis=1)
    df_test['MEAN_NOTA_CN_REDACAO'] = df_test[[
        'NU_NOTA_CN', 'NU_NOTA_REDACAO']].mean(axis=1)
    df_test['MEAN_NOTA_LC_REDACAO'] = df_test[[
        'NU_NOTA_LC', 'NU_NOTA_REDACAO']].mean(axis=1)

    df_test['KURTOSIS_NOTAS_COMP'] = df_test[notas_red].kurtosis(axis=1)
    df_test['MEAN_NOTAS_COMP'] = df_test[notas_red].mean(axis=1)
    df_test['MEDIAN_NOTAS_COMP'] = df_test[notas_red].median(axis=1)
    df_test['MAD_NOTAS_COMP'] = df_test[notas_red].mad(axis=1)
    df_test['QUANTILE_NOTAS_COMP'] = df_test[notas_red].quantile(axis=1)
    df_test['SEM_NOTAS_COMP'] = df_test[notas_red].sem(axis=1)
    df_test['STD_NOTAS_COMP'] = df_test[notas_red].std(axis=1)
    df_test['VAR_NOTAS_COMP'] = df_test[notas_red].var(axis=1)
    df_test['AMP_NOTAS_COMP'] = df_test[notas_red].max(
        axis=1) - df_test[notas_red].min(axis=1)

    df_test['MEAN_NOTA_COMP1_COMP2'] = df_test[[
        'NU_NOTA_COMP1', 'NU_NOTA_COMP2']].mean(axis=1)
    df_test['MEAN_NOTA_COMP1_COMP3'] = df_test[[
        'NU_NOTA_COMP1', 'NU_NOTA_COMP3']].mean(axis=1)
    df_test['MEAN_NOTA_COMP1_COMP4'] = df_test[[
        'NU_NOTA_COMP1', 'NU_NOTA_COMP4']].mean(axis=1)
    df_test['MEAN_NOTA_COMP1_COMP5'] = df_test[[
        'NU_NOTA_COMP1', 'NU_NOTA_COMP5']].mean(axis=1)
    df_test['MEAN_NOTA_COMP2_COMP3'] = df_test[[
        'NU_NOTA_COMP2', 'NU_NOTA_COMP3']].mean(axis=1)
    df_test['MEAN_NOTA_COMP2_COMP4'] = df_test[[
        'NU_NOTA_COMP2', 'NU_NOTA_COMP4']].mean(axis=1)
    df_test['MEAN_NOTA_COMP2_COMP5'] = df_test[[
        'NU_NOTA_COMP2', 'NU_NOTA_COMP5']].mean(axis=1)
    df_test['MEAN_NOTA_COMP3_COMP4'] = df_test[[
        'NU_NOTA_COMP3', 'NU_NOTA_COMP4']].mean(axis=1)
    df_test['MEAN_NOTA_COMP3_COMP5'] = df_test[[
        'NU_NOTA_COMP3', 'NU_NOTA_COMP5']].mean(axis=1)
    df_test['MEAN_NOTA_COMP4_COMP5'] = df_test[[
        'NU_NOTA_COMP4', 'NU_NOTA_COMP5']].mean(axis=1)

    return df_train, df_test, df_answer


if __name__ == "__main__":

    df_train = pd.read_csv('data/treino_dividido/new_TRAIN.csv')
    df_test = pd.read_csv('data/treino_dividido/new_TEST.csv')

    test_aux = pd.read_csv('data/test.csv')

    df_answer = pd.DataFrame()

    df_train, df_test, df_answer = p_data(
        df_train, df_test, test_aux, df_answer)

    label = df_train['NU_NOTA_MT']

    df_train.drop(['NU_NOTA_MT'], axis=1, inplace=True)

    tl = TomekLinks()

    newX, newY = tl.fit_resample(df_train.values, label.values)

    print(len(df_train), len(newX))

    # model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
    #                               max_features='auto', max_leaf_nodes=None,
    #                               min_impurity_decrease=0.0, min_impurity_split=None,
    #                               min_samples_leaf=4, min_samples_split=5,
    #                               min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=-2,
    #                               oob_score=False, random_state=None, verbose=0, warm_start=False)

    # model.fit(df_train, label)

    # #predictions = model.predict(df_test)

    # df_answer['NU_NOTA_MT'] = np.around(predictions, 2)

    # df_answer.to_csv('unbalanced.csv', index=False, header=True)

    # data = convert_to_zero('zscore_sem_notas_comp_answer')

    # df = ft_importance(k=87)

    # features = df['Specs'].tolist()[:60]

    # df_train = df_train[features]

    # df_test = df_test[features]
