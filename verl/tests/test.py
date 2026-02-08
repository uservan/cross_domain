import pandas as pd

# 读取本地 parquet 文件
df = pd.read_parquet("/home/wxy320/ondemand/program/verl/data/train/MATH.parquet")

# 查看前几行
print(df.head())