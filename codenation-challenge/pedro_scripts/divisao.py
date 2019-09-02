import pandas as pd
import sys

CHUNKSIZE = 1e7

'''
Argumento 1 -> PATH DA BASE
Argumento 2 -> Porcentagem do treino
'''


def main(argv):
    df = pd.read_csv(argv[1], sep=',')

    pArg = float(argv[2])

    if pArg >= 1:
        pArg = float(pArg/100)

    df = df.sample(
        frac=1, random_state=42).reset_index(drop=True)

    percentage = int(df.shape[0] * 0.7)

    X_train = df.iloc[:percentage, :]
    X_test = df.iloc[percentage:, :]

    X_train.to_csv('new_TRAIN.csv', index=False)
    X_test.to_csv('new_TEST.csv', index=False)


if __name__ == '__main__':
    main(sys.argv)
