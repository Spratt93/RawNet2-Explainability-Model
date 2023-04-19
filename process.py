import pandas as pd

# USED TO CLEAN THE ORIGINAL CSV TO JUST ID AND SCORES
# data = pd.read_csv('results.csv', delimiter='\s+')
# data = data.drop(columns=['date', 'dateTime', 'IP', 'address', 'preamble'])
# data.to_csv('processed.csv')

df = pd.read_csv('processed.csv')

for i in range(1, 61):
    c = str(i)
    if i < 11 or (i > 20 and i < 31) or (i > 40 and i < 51):
        df = df.replace({c : {'fake': 1, 'real': 0}})
    else:
        df = df.replace({c : {'real': 1, 'fake': 0}})

print(df.iloc[1,1:61].sum())
# print(df.head())

# AVG OVERALL SCORE
# print('Mean score: {}'.format(df['score'].mean()))

# lst = []

# for n in range(70):
#     g1 = df.iloc[n,1:21].tolist()
#     for i, ans in enumerate(g1):
#         if i < 10:
#             if ans == 'fake':
#                 g1[i] = 1
#             else:
#                 g1[i] = 0
#         else:
#             if ans == 'real':
#                 g1[i] = 1
#             else:
#                 g1[i] = 0

#         lst.append(g1)
