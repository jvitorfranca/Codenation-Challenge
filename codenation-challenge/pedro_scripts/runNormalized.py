from sklearn.ensemble import RandomForestRegressor
import sys
from utilities import convert_to_zero
from utilities import ft_importance
import DataObject as do
import pandas as pd
import numpy as np


def main(argv):
    df_train = pd.read_csv('data/train.csv', dtype={'Q001': 'category', 'Q002': 'category',
                                                    'Q006': 'category', 'Q024': 'category',
                                                    'Q025': 'category', 'Q026': 'category',
                                                    'Q047': 'category'})

    df_test = pd.read_csv('data/test.csv', dtype={'Q001': 'category', 'Q002': 'category',
                                                  'Q006': 'category', 'Q024': 'category',
                                                  'Q025': 'category', 'Q026': 'category',
                                                  'Q047': 'category'})
    df_answer = pd.DataFrame()

    features = list(df_test.columns)
    features.append('NU_NOTA_MT')

    df_answer['NU_INSCRICAO'] = df_test['NU_INSCRICAO']

    df_train = df_train[features]

    df_train['NU_NOTA_MT'].fillna(0, inplace=True)

    dataset = do.DataObject(df_train, df_test, label='NU_NOTA_MT')

    dataset.dropColumns('NU_INSCRICAO', 'SG_UF_RESIDENCIA', 'TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC',
                        'TP_PRESENCA_CH', 'Q027', 'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT')

    dataset.replaceValue('TP_SEXO', {'M': 0, 'F': 1})

    dataset.train['TP_STATUS_REDACAO'].fillna(8, inplace=True)
    dataset.test['TP_STATUS_REDACAO'].fillna(8, inplace=True)

    dataset.fillEmpty(0, ignore=['TP_STATUS_REDACAO'])
    dataset.fillEmpty(8, only_on=['TP_STATUS_REDACAO'])

    dataset.oneHotEncode()

    notas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO']

    dataset.calculateStatistics(notas, 'NOTAS')

    df = ft_importance(k=70)

    features = df['Specs'].tolist()[:70]

    # print(list(dataset.train.columns))

    dataset.train = dataset.train[features]

    dataset.test = dataset.test[features]

    dataset.normalizeData()

    model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
                                  max_features='auto', max_leaf_nodes=None,
                                  min_impurity_decrease=0.0, min_impurity_split=None,
                                  min_samples_leaf=4, min_samples_split=5,
                                  min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
                                  oob_score=False, random_state=None, verbose=0, warm_start=False)

    model.fit(dataset.train, dataset.train_y)

    predictions = model.predict(dataset.test)

    df_answer['NU_NOTA_MT'] = np.around(predictions, 2)

    df_answer.to_csv('automl_answer.csv', index=False, header=True)

    data = convert_to_zero('automl_answer')


if __name__ == '__main__':
    main(sys.argv)
