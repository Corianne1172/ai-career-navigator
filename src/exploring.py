import pandas as pd

pd.set_option('display.max_colwidth', None)

df = pd.read_csv("../data/ESCO/skills_en.csv")
columns = df.columns
print(columns)
for i in range(10):
    print(df.loc[i, 'preferredLabel'])
    print(df.loc[i, 'altLabels'])
    print(df.loc[i, 'skillType'])
    print('---')
    
    
df1 = pd.read_csv("../data/Online_Courses.csv")
    
print("Total rows:", len(df1))
print("Rows with non-null Skills:", df1['Skills'].notna().sum())
print("Rows with null Skills:", df1['Skills'].isna().sum())

print("\nSample Skills values:")
for i in range(5):
    print(df1.loc[i, 'Skills'])
    print('---')
    
    