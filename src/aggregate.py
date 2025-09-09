from glob import glob
import pandas as pd
import os

files = glob("*")
agg = []
for f in files:
    if os.path.isdir(f):
        df = pd.read_csv(f"{f}/results.csv", index_col=0)
        agg.append(df)
results = pd.concat(agg)
results.to_csv("results.csv")