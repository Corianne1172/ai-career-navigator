import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
print(path)

for file in os.listdir(path):
    print(file)
    
for file in os.listdir(path):
    full = path + "/" + file
    print(full)
    if os.path.isdir(full):
        print(os.listdir(full)[:5])
        
df = pd.read_csv(path + "/Resume/Resume.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head(2))

print(df['Resume_str'][0][:500])
print(df['Category'].value_counts())

path_jd = kagglehub.dataset_download("adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles")
print(path_jd)

for file in os.listdir(path_jd):
    print(file)
    
df_jd = pd.read_csv(path_jd + "/job_dataset.csv")
print(df_jd.shape)
print(df_jd.columns.tolist())
print(df_jd.head(2))

print(df_jd['Title'][0])
print(df_jd['Skills'][0])
print(df_jd['Responsibilities'][0][:300])