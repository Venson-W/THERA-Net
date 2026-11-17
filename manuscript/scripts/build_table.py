"""Regenerate the reported paired-error table from the preserved numerical record."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
metrics = json.loads((ROOT / 'data' / 'rupture_metrics.json').read_text())
labels = {'heterogeneous_D16': '$D=1.6$', 'uniform': 'Uniform', 'unseen_D13': '$D=1.3$'}
rows = []
for case in labels:
    m = next(x for x in metrics['learning_comparisons'] if x['case'] == case and x['nx'] == 360)
    vals = [100*m[k] for k in ['velocity_relative_L2_VW', 'final_slip_relative_L2_VW', 'moment_rate_relative_L2_whole_fault']]
    rows.append(f"{labels[case]} & {vals[0]:.4f} & {vals[1]:.5f} & {vals[2]:.5f} & {1000*m['arrival_0p5']['RMSE_s']:.4f}" + r" \\")
text = r'''\begin{table}[htbp]
\centering
\caption{Paired THERA-Net/reference-model errors on the 200 m grid. Field and final-slip errors use the geometric weakening rectangle; moment rate integrates the whole fault. Percentages are $100E_q$. Arrival differences use interpolated threshold crossings, with no crossing-mask disagreements.}
\label{tab:rupture}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Case & $E_V$ (\%) & $E_\delta$ (\%) & $E_{\dot M}$ (\%) & Arrival RMSE (ms)\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
(ROOT / 'rupture_table.tex').write_text(text, encoding='utf-8')
print('Wrote rupture_table.tex from three recorded 200 m comparisons.')
