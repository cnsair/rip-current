import pandas as pd, numpy as np
from scipy.stats import wilcoxon
rng = np.random.default_rng(42)

look = pd.read_csv('frame_to_video.csv')          # file_name, video_id, is_NR
mod  = pd.read_csv('video_modality.csv')          # video_id, modality
base = pd.read_csv('scores_theta0.csv')[['file_name','recall']]
sawi = pd.read_csv('scores_sawi_a070.csv')[['file_name','recall']]

df = (base.merge(sawi, on='file_name', suffixes=('_b','_s'))
          .merge(look, on='file_name').merge(mod, on='video_id'))
df = df[df.is_NR == 0]                            # rip-bearing videos only
df['delta'] = df.recall_s - df.recall_b

pv = df.groupby(['modality','video_id'])['delta'].mean().reset_index()

for m, g in pv.groupby('modality'):
    d = g.delta.values
    boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(10000)]
    lo, hi = np.quantile(boot, [.025, .975])
    try:    _, p = wilcoxon(d, alternative='two-sided')
    except ValueError: p = np.nan
    print(f'{m:8s} n={len(d):2d} videos  mean Δ={d.mean()*100:+.2f} pp  '
          f'95% CI [{lo*100:+.2f}, {hi*100:+.2f}]  p={p:.3g}  '
          f'won {100*np.mean(d>0):.0f}% of videos')