import pandas as pd
pd.options.display.max_rows = None
df = pd.read_csv('../data/val2017_degraded.csv')
print((df.groupby('quality').filesize.sum() / 2**20).rename("size [MB]"))
total = df.filesize.sum()
print(f"Total: {total:,} = {total/2**30:.1f} GB")
