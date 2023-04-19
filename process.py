import pandas as pd

"""
    Return the average score for the question groups
"""
def group_averages(df):
    for i in range(1, 61):
        c = str(i)
        if i < 11 or (i > 20 and i < 31) or (i > 40 and i < 51):
            df = df.replace({c : {'fake': 1, 'real': 0}})
        else:
            df = df.replace({c : {'real': 1, 'fake': 0}})

    r = range(1,21)
    cols = map(str, r)
    df['g1'] = df[cols].sum(axis=1)
    print('Group 1 average: {}'.format(df['g1'].mean()))

    r = range(21,41)
    cols = map(str, r)
    df['g2'] = df[cols].sum(axis=1)
    print('Group 2 average: {}'.format(df['g2'].mean()))

    r = range(41,61)
    cols = map(str, r)
    df['g3'] = df[cols].sum(axis=1)
    print('Group 3 average: {}'.format(df['g3'].mean()))


def avg_score(df):
    print('Mean score: {}'.format(df['score'].mean()))

"""
    Cleans the original csv file from unnecessary info
"""
def clean_results(df):
    data = pd.read_csv('results.csv', delimiter='\s+')
    data = data.drop(columns=['date', 'dateTime', 'IP', 'address', 'preamble'])
    data.to_csv('processed.csv')



if __name__ == '__main__':
    df = pd.read_csv('processed.csv') # cleaned csv
    group_averages(df)