import pandas as pd, numpy as np
from scipy.stats import wilcoxon

b = pd.read_csv('per_image_theta0.csv')[['file_name','f2']]
m = pd.read_csv('per_image_dualbranch.csv')[['file_name','f2']]
d = b.merge(m, on='file_name', suffixes=('_b','_m'))
diff = (d.f2_m - d.f2_b).values
nz = diff[diff != 0]
ranks = pd.Series(np.abs(nz)).rank().values
r = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
stat, p = wilcoxon(d.f2_m, d.f2_b, alternative='two-sided')
print(f'r = {r:+.3f}   p = {p:.4g}   n = {len(d)}')