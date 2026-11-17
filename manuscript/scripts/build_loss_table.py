"""Format the independently recomputed loss decomposition of the saved checkpoint."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'data/training_loss_components.json').read_text())
def scientific(value):
    base, power = f'{value:.4e}'.split('e')
    return rf'${base}\times10^{{{int(power)}}}$'
rows=[]
for d in data['datasets']:
    rows.append(f"{d['dataset'].capitalize()} & {d['samples']} & " +
                ' & '.join(scientific(d[k]) for k in ['traction_loss','state_loss','joint_loss']) + r' \\')
table = r'''\begin{table}[hbp]
\centering
\caption{Dimensionless loss components recomputed at the selected seed-12 checkpoint. All three sets use the same fixed scaling and unit weights in Eq.~\eqref{meth:loss}. The test set is evaluated only after model selection.}
\label{tab:loss_components}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Dataset & Samples & $\mathcal L_\tau$ & $\mathcal L_\theta$ & $\mathcal L$\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
(ROOT / 'training_loss_table.tex').write_text(table,encoding='utf-8')
print('Wrote training_loss_table.tex')
