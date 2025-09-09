import pandas as pd
from glob import glob

csv = glob("outputs/*/results.csv")
files = []
for file in csv:
    df = pd.read_csv(file, index_col=0)
    df.index = [file.split("/")[1]]
    files.append(df)
df = pd.concat(files)
import pdb; pdb.set_trace()
df.to_csv("outputs/aggregate.csv")