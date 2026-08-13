import pandas as pd, numpy as np, glob, os
from scipy.stats import wilcoxon

seg = pd.read_csv('results/segformer_b2_per_image.csv')[['image','recall']]

def rb(a, b):                       # a = SegFormer, b = competitor
    d = (a - b).values
    nz = d[d != 0]
    if len(nz) == 0: return 0.0, np.nan
    ranks = pd.Series(np.abs(nz)).rank().values
    r = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    _, p = wilcoxon(a, b, alternative='two-sided')
    return r, p

for f in sorted(glob.glob('results/*_per_image.csv')):
    if 'segformer_b2_per_image' in f: continue
    o = pd.read_csv(f)[['image','recall']]
    m = seg.merge(o, on='image', suffixes=('_s','_o'))
    if len(m) != len(seg):
        print(f'  ! {os.path.basename(f)}: merged {len(m)} of {len(seg)}')
    r, p = rb(m.recall_s, m.recall_o)
    print(f'{os.path.basename(f):48s} r = {r:+.3f}   p = {p:.3g}   n = {len(m)}')