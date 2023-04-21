import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


"""
    Clean original csv file
"""
def clean_results():
    data = pd.read_csv('results.csv', delimiter='\s+')
    data = data.drop(columns=['date', 'dateTime', 'IP', 'address', 'preamble'])
    data.to_csv('processed.csv')


"""
    Return the average participant score
"""
def avg_score(df):
    print('Mean score: {}'.format(df['score'].mean()))


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

    error_counts(df)


"""
    Return the error count for each question
"""
def error_counts(df):
    inds = range(60)
    counts = []
    for i in inds:
        counts.append((i, 0))

    for i in range(1,61):
        col = str(i)
        err_count = 70 - sum(df[col])
        ind = i-1
        counts[ind] = (ind, err_count)

    # counts = sorted(counts, key=lambda count: count[1], reverse=True) # sorted error counts for each q.

    trad = [0,1,2,20,21,22,23,40,41,42]
    concat = [3,4,5,27,28,29,46,47,48,49]
    neural = [6,7,8,9,24,25,26,43,44,45]

    trad_counts = []
    concat_counts = []
    neural_counts = []
    bonafide_counts = []

    for i,v in counts:
        if i in trad:
            trad_counts.append(v)

        if i in concat:
            concat_counts.append(v)

        if i in neural:
            neural_counts.append(v)

        else:
            bonafide_counts.append(v)

    print('Average error for traditional_vocoder: {}'.format(sum(trad_counts) / len(trad_counts)))
    print('Average error for waveform_concatenation: {}'.format(sum(concat_counts) / len(concat_counts)))
    print('Average error for neural_vocoder: {}'.format(sum(neural_counts) / len(neural_counts)))
    print('Average error for bonafide: {}'.format(sum(bonafide_counts) / len(bonafide_counts)))

    model_error(counts)


"""
    Calculates average participant error rate 
    when model is incorrect
"""
def model_error(counts):
    model_fails = [12, 33, 35, 36, 37, 41, 57]
    human = []
    for f in model_fails:
        i,c = counts[f-1]
        p = (c / 70) * 100
        human.append((f, p))

    human = sorted(human, key=lambda human: human[1], reverse=True)
    print(human)


"""
    Plots the performance increase from full assistance
    relative to both minimal and no assistance
"""
def plot_perf_boost():
    fst = ['G3 vs G1']
    r1 = [7.3]
    snd = ['G3 vs G2']
    r2 = [2.7]

    plt.bar(fst, r1, color='olivedrab', label='G3 vs G1')
    plt.bar(snd, r2, color='indianred', label='G3 vs G2')
    plt.xlabel('Groups')
    plt.ylabel('Performance Increase (%)')
    plt.ylim([0, 30])
    plt.title('Relative performance increase')
    plt.legend()
    plt.show()


"""
    Plots the difference in performance between
    the model and participants for each question group
"""
def plot_group_perf():
    x = ['G1', 'G2', 'G3']
    w = 0.4
    groups = [15.5, 16.2, 16.63]
    model = [19, 16, 18]
    fst = [0,1,2]
    snd = [i+w for i in fst]

    plt.bar(fst, groups, w, label='Human')
    plt.bar(snd, model, w, label='Model')
    plt.xlabel('Groups')
    plt.ylabel('Correct Answers')
    plt.ylim([0,20])
    plt.xticks([0.2, 1.2, 2.2], x)
    plt.title('Human vs Model performance in each group')
    plt.legend()
    plt.show()


"""
    Pie chart displaying vocoder distribution of participant errors
"""
def error_pie():
    scores = np.array([20,18,45,17])
    labels = ['Traditional', 'Waveform Concatenation', 'Neural', 'Bona fide']
    clrs = ['lightcoral','paleturquoise','orange','lightgreen']
    plt.pie(scores, labels=labels, colors=clrs, autopct='%0.0f%%')
    plt.title('Distribution of errors amongst vocoders')
    plt.show()


"""
    Plots participant performance in instances when
    model is incorrect
"""
def model_error_plot():
    g1 = ['12']
    g2 = ['35','36','33','37']
    g3 = ['41','57']
    r1 = [10.0]
    r2 = [48.57, 30.0, 24.29, 22.86]
    r3 = [57.14, 40.0]

    plt.barh(g1, r1, color='lightslategrey', label='Group 1')
    plt.barh(g2, r2, color='indianred', label='Group 2')
    plt.barh(g3, r3, color='olivedrab', label='Group 3')
    plt.xlabel('Incorrect particpants (%)')
    plt.ylabel('Question No.')
    plt.title('Human performance when model incorrect')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # df = pd.read_csv('processed.csv') # cleaned csv
    # group_averages(df)
    plot_perf_boost()
