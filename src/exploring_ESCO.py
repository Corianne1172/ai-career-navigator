import pandas as pd

df = pd.read_csv("../data/ESCO/skills_en.csv")
columns = df.columns
print(columns)
for i in range(10):
    print(df.loc[i, 'preferredLabel'])
    print(df.loc[i, 'altLabels'])
    print(df.loc[i, 'skillType'])
    print('---')