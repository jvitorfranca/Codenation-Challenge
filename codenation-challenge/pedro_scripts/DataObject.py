import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_categorical_dtype

''' Objecto genérico para fazer operações básicas em pandas mais rápido'''


class DataObject:
    def __init__(self, train_data, test_data=None, label=None, normalize=False):
        self.train = train_data
        self.test = test_data
        self.train_y = None
        self.test_y = None
        self._train_only = False
        self._label_column = label
        # self._image = image

        if label is not None:
            self.generateLabels()

        if normalize == True:
            self.normalizeData()

    def normalizeData(self):
        norm = Normalizer().fit(self.train.values)

        self.train[:] = norm.transform(self.train.values)

        if self.test is not None and not self._train_only:
            self.test[:] = norm.transform(
                self.test.values)

    def dropColumns(self, *columns):
        self.train.drop(list(columns), axis=1, inplace=True)

        if self.test is not None and not self._train_only:
            self.test.drop(list(columns), axis=1, inplace=True)

    def switchOperationMode(self):
        self._train_only = not self._train_only

    def generateLabels(self):
        self.train_y = self.train.pop(self._label_column)

        if self.test is not None and not self._train_only:
            if self._label_column in self.test.columns:
                self.test_y = self.test.pop(self._label_column)

    # def split(self, size_test):
    #     self.train.values, self.test.values, self.train_y.values, self.test_y.values = train_test_split(
    #         self.train, self.test, size_test)

    def fillEmpty(self, value, ignore=[], only_on=None):
        if only_on is not None:
            for column in only_on:
                if is_numeric_dtype(self.train[column]):
                    self.train[column].fillna(value, inplace=True)

                    if self.test is not None and not self._train_only:
                        self.test[column].fillna(value, inplace=True)

        for column in self.train.columns:
            if column in ignore:
                continue

            if is_numeric_dtype(self.train[column]):
                self.train[column].fillna(value, inplace=True)

                if self.test is not None and not self._train_only:
                    self.test[column].fillna(value, inplace=True)

    def replaceValue(self, column, value_map):
        for key, val in value_map.items():
            self.train[column].replace(key, val, inplace=True)

            if self.test is not None and not self._train_only:
                self.test[column].replace(key, val, inplace=True)

    def oneHotEncode(self):
        col_list = []

        for column in self.train.columns:
            if is_categorical_dtype(self.train[column]):
                col_list.append(column)

        self.train = pd.get_dummies(
            self.train, prefix=col_list, columns=col_list)

        if self.test is not None and not self._train_only:
            self.test = pd.get_dummies(
                self.test, prefix=col_list, columns=col_list)

    def calculateStatistics(self, columns, group_name):
        self.train['KURTOSIS_' +
                   group_name] = self.train[columns].kurtosis(axis=1)
        self.train['MEAN_' + group_name] = self.train[columns].mean(axis=1)
        self.train['MEDIAN_' + group_name] = self.train[columns].median(axis=1)
        self.train['MAD_' + group_name] = self.train[columns].mad(axis=1)
        self.train['QUANTILE_' +
                   group_name] = self.train[columns].quantile(axis=1)
        self.train['SEM_' + group_name] = self.train[columns].sem(axis=1)
        self.train['SKEW_' + group_name] = self.train[columns].skew(axis=1)
        self.train['STD_' + group_name] = self.train[columns].std(axis=1)
        self.train['VAR_' + group_name] = self.train[columns].var(axis=1)

        if self.test is not None and not self._train_only:
            self.test['KURTOSIS_' +
                      group_name] = self.test[columns].kurtosis(axis=1)
            self.test['MEAN_' + group_name] = self.test[columns].mean(axis=1)
            self.test['MEDIAN_' +
                      group_name] = self.test[columns].median(axis=1)
            self.test['MAD_' + group_name] = self.test[columns].mad(axis=1)
            self.test['QUANTILE_' +
                      group_name] = self.test[columns].quantile(axis=1)
            self.test['SEM_' + group_name] = self.test[columns].sem(axis=1)
            self.test['SKEW_' + group_name] = self.test[columns].skew(axis=1)
            self.test['STD_' + group_name] = self.test[columns].std(axis=1)
            self.test['VAR_' + group_name] = self.test[columns].var(axis=1)
