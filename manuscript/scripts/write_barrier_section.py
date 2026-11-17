"""Write quantitative manuscript additions from audited, completed simulations."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
d=json.loads((ROOT/'data/barrier_metrics.json').read_text())
assert d['all_required_complete'] and len(d['main_pairs'])==12 and len(d['fine_pairs'])==3
pairs=d['main_pairs'];runs=d['runs'];pilots=d['pilot_runs']
assert all(p['outcome_agrees'] for p in pairs)

def get(w,n,dt,mode='reference',aa=.012):
    return next(r for r in runs+pilots if r['width_km']==w and r['a_barrier']==aa and r['nx']==n and r['dt_s']==dt and r['mode']==mode)

def sci(x):
    m,e=f'{x:.3e}'.split('e');return rf'{m}\times10^{{{int(e)}}}'

for w,aa,status in [(1,.012,'activated'),(2,.014,'arrested')]:
    for n,dt in [(180,.01),(360,.01),(360,.005),(720,.005)]:assert get(w,n,dt,aa=aa)['status']==status

rows=[]
for p in sorted(pairs,key=lambda p:(p['a_barrier'],p['width_km'])):
    r=next(r for r in runs if r['name']==p['reference']);s=next(r for r in runs if r['name']==p['thera'])
    t='--' if r['receiver_first_arrival_s'] is None else f"{r['receiver_first_arrival_s']:.3f}"
    rows.append(f"{p['width_km']:g} & {p['a_barrier']:.3f} & {r['receiver_fraction']:.3f}/{s['receiver_fraction']:.3f} & {t} & {100*p['field_relative_L2']:.4f} & {100*p['work_relative_error']:.5f} "+r'\\')
table=r'''\begin{table}[htbp]
\centering
\caption{Complete segmented-fault sweep on the 200 m grid with a 10 ms step.
The receiver area fraction $F_{\rm R}$ is listed as reference/THERA-Net;
$t_{\rm R}$ is the first reference threshold crossing within the receiving
region. A dash denotes no crossing by 15 s. Relative errors compare independently
evolved trajectories on matched grids; $E_W$ uses whole-fault frictional work.}
\label{tab:barrier_sweep}
\begin{tabular}{rrrrrr}
\toprule
$w$ (km) & $a_{\rm B}$ & $F_{\rm R}$ & $t_{\rm R}$ (s) & $E_V$ (\%) & $E_W$ (\%) \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}
'''
(ROOT/'barrier_sweep_table.tex').write_text(table)
rows=[]
for w,aa in [(1,.012),(2,.012),(2,.014)]:
    for n,dt in [(180,.01),(360,.01),(360,.005),(720,.005)]:
        r=get(w,n,dt,aa=aa)
        try:s=get(w,n,dt,'thera',aa=aa);f=f"{r['receiver_fraction']:.3f}/{s['receiver_fraction']:.3f}"
        except StopIteration:f=f"{r['receiver_fraction']:.3f}/--"
        t='--' if r['receiver_first_arrival_s'] is None else f"{r['receiver_first_arrival_s']:.3f}"
        rows.append(f'{w:g} & {aa:.3f} & {72000/n:g} & {dt*1000:g} & {f} & {t} '+r'\\')
(ROOT/'barrier_refinement_table.tex').write_text(r'''\begin{table}[htbp]
\centering
\caption{Resolution checks for rate-neutral and velocity-strengthening bands.
Area fractions are reference/THERA-Net. A missing neural entry denotes a
reference-only pilot or time-step control. All outcomes refer to the same
15 s observation interval and physically fixed spatial profiles.}
\label{tab:barrier_refinement}
\begin{tabular}{rrrrrr}
\toprule
$w$ (km) & $a_{\rm B}$ & $\Delta x$ (m) & $\Delta t$ (ms) & $F_{\rm R}$ & $t_{\rm R,ref}$ (s) \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}
''')

emin=min(p['field_relative_L2'] for p in pairs)*100;emax=max(p['field_relative_L2'] for p in pairs)*100
wmax=max(p['work_relative_error'] for p in pairs)*100
amax=max(p['arrival_rmse_s'] for p in pairs)*1000
mask=max(p['rupture_mask_disagreement_nodes'] for p in pairs)
refcases=[next(r for r in runs if r['name']==p['reference']) for p in pairs]
nact=sum(r['status']=='activated' for r in refcases);narr=sum(r['status']=='arrested' for r in refcases)
minp=min(r['min_power_W_m2'] for r in runs);mint=min(r['min_theta_s'] for r in runs)
fineerr=max(p['field_relative_L2'] for p in d['fine_pairs'])*100
checks=[r for r in d['refinement_comparisons'] if r['coarse_dt_s']==.01]
assert len(checks)==3
griderr={(r['width_km'],r['a_barrier']):r['field_relative_L2']*100 for r in checks}
fineagree=sum(p['outcome_agrees'] for p in d['fine_pairs'])
near_ref=get(2,360,.01);near_pred=get(2,360,.01,'thera')
near_fine=get(2,720,.005);near_fine_pred=get(2,720,.005,'thera')
onsets=[get(1,n,dt)['receiver_first_arrival_s'] for n,dt in [(180,.01),(360,.01),(360,.005),(720,.005)]]
text=rf'''\subsection{{Barrier-controlled receiver activation and rupture arrest}}
\label{{sec:barrier_results}}

The segmented-fault experiment changes the spatial organization of the rupture
problem while retaining the calibrated local material family. Twelve paired
configurations combine $w=1,2,4$ km with $a_{{\rm B}}=0.010,0.011,0.012,0.014$.
All use the 200 m grid, a 10 ms step, and the fixed checkpoint employed above.
Figure~\ref{{fig:barrier_regimes}} shows the new arrangement and the complete
outcome map. Table~\ref{{tab:barrier_sweep}} reports every main configuration:
{nact} activate the receiving patch and {narr} arrest before doing so within
15 s. THERA-Net gives the same segment-level outcome in all twelve pairs.

\input{{barrier_sweep_table}}

The transition depends on both band width and constitutive rate sensitivity.
For $a_{{\rm B}}=0.011$, the 2 km band permits receiver activation while the
4 km band arrests rupture, even though the band remains velocity weakening.
For the rate-neutral reference band, $a_{{\rm B}}=b=0.012$, changing width from
2 to 4 km likewise changes the outcome on this grid. All sampled 1--4 km strengthening
bands with $a_{{\rm B}}=0.014$ arrest rupture. These results show why the sign
of steady rate dependence alone does not classify the complete spatial event.
The pilot sweep additionally includes a no-band control and narrower or
stronger bands; all pilot records are retained with the computational data.

Figures~\ref{{fig:barrier_fronts}} and \ref{{fig:barrier_spacetime}} resolve the
two representative outcomes. The 1 km rate-neutral case develops rupture on
both sides of the band, whereas the 2 km strengthening case ($a_{{\rm B}}=0.014$)
confines seismic slip to the incident side.
The comparison includes the unreached regions rather than assigning them a
spurious arrival time. The receiver station at $(8,0)$ km records the strong
contrast in slip rate between the two outcomes. Space--time arrival traces provide
a separate view of passage across the band at the prescribed threshold;
the receiving-area measure is not itself a causal classification of remote
triggering.

Across the twelve matched comparisons, slip-rate field differences are
${emin:.4f}$--${emax:.4f}\%$, the largest jointly reached arrival-time RMSE is
${amax:.4f}$ ms, and the largest whole-fault frictional-work difference is
${wmax:.5f}\%$. The maximum disagreement in the active-region threshold masks
is {mask} nodes. These are errors from replacing the reference constitutive parameters
within the same discretized evolution. No band geometry, activation label,
or rupture history was supplied to the network during calibration.

\subsection{{Regional dissipation and resolution of the segmentation outcome}}

Figure~\ref{{fig:barrier_energy}} partitions the accumulated interface work
between the incident patch, the band including its transitions, and the
receiving patch. Activation of the second segment adds substantial dissipation
there; an arrested event leaves its accumulated work small. Reference and
neural values are computed by independently evolving each trajectory and
integrating its actual traction--slip-rate product. The minimum state across
the main and refinement records is ${sci(mint)}$ s, and the minimum sampled
local power is ${sci(minp)}$ W/m$^2$. The near-rest numerical qualification
discussed above applies to the latter quantity. Regional sums reproduce the
nodal work accumulation, and the whole-fault sum reproduces the time integral
of the stored frictional-power history to relative tolerance $10^{{-10}}$.

Activation for the 1 km rate-neutral band and arrest for the 2 km strengthening
band persist on 400, 200 and 100 m grids and in the additional 200 m, 5 ms
reference controls (Table~\ref{{tab:barrier_refinement}}). Across the three
100 m neural/reference pairs, {fineagree} have the same receiver outcome,
with paired field differences no greater than ${fineerr:.4f}\%$.

The intervening 2 km rate-neutral band is resolution sensitive: it arrests
on the 400 m grid but activates the receiver on the 200 m grid. This outcome
change is retained explicitly in Table~\ref{{tab:barrier_refinement}}. The
reference receiving-area fraction by 15 s increases from
${near_ref['receiver_fraction']:.4f}$ at 200 m to
${near_fine['receiver_fraction']:.4f}$ at 100 m; the corresponding neural values
are ${near_pred['receiver_fraction']:.4f}$ and
${near_fine_pred['receiver_fraction']:.4f}$. Thus activation does not imply
that the entire receiving patch has ruptured within the observation window. The
sampled map establishes stable examples of both propagation and arrest, but
its coarse-grid transition cannot be interpreted as a converged physical
critical width. The distinction matters because close neural/reference
agreement on a given grid does not remove discretization error in that grid's
activation decision.

\input{{barrier_refinement_table}}

Receiver onset for the 1 km band ranges from ${min(onsets):.3f}$ to
${max(onsets):.3f}$ s across these discretizations. Joint refinement from
200 m/10 ms to 100 m/5 ms changes the reference slip-rate field by
${griderr[1.,.012]:.2f}\%$ for the 1 km neutral band,
${griderr[2.,.012]:.2f}\%$ for the 2 km neutral band, and
${griderr[2.,.014]:.2f}\%$ for the 2 km strengthening band, at common nodes and
snapshot times. Robustness of the selected segment-level outcomes is therefore
reported separately from pointwise waveform resolution. These results belong
to the specified periodic computational domain,
initial state convention and 15 s window; domain-size and longer-time tests
would be needed to identify an isolated-fault, asymptotic arrest threshold.

\begin{{figure}}[p]
\centering\includegraphics{{barrier_setup_regimes.pdf}}
\caption{{Segmented-fault experiment: (a) prescribed $a_s-b$ for the 2 km
rate-neutral band, with the nucleation star, dashed active rectangle, and
dotted boundaries enclosing the band transitions; (b) the complete 200 m outcome map, showing reference/
THERA-Net receiving-area fractions by 15 s, rounded to two decimal places.
Each cell is a computed case;
no continuous phase boundary is interpolated between samples.}}
\label{{fig:barrier_regimes}}
\end{{figure}}

\begin{{figure}}[p]
\centering\includegraphics{{barrier_fronts_stations.pdf}}
\caption{{Representative transmission and arrest: (a) a 1 km rate-neutral band
and (b) a 2 km strengthening band. Front contours are spaced by 1 s, with
reference in black and THERA-Net in red dashed lines; gray marks
the band and its transitions. Unreached nodes are masked. (c) Full-step speed
at the receiving station $(8,0)$ km. Both calculations use 200 m spacing and
a 10 ms step. The logarithmic scale resolves the arrested receiver response.}}
\label{{fig:barrier_fronts}}
\end{{figure}}

\begin{{figure}}[p]
\centering\includegraphics{{barrier_spacetime.pdf}}
\caption{{Centerline space--time slip-rate fields for the transmitting neutral
band and the arresting strengthening band in Fig.~\ref{{fig:barrier_fronts}}.
Color shows the reference speed, with a plotting floor of $10^{{-8}}$
m/s. White solid and red dashed lines show first $0.5$ m/s crossings of the
reference and THERA-Net fields, tracked at every integration step. Colors
use the 0.1 s snapshots. Dotted lines enclose
the band and its transitions.}}
\label{{fig:barrier_spacetime}}
\end{{figure}}

\begin{{figure}}[p]
\centering\includegraphics{{barrier_energy_refinement.pdf}}
\caption{{(a) Accumulated frictional work in the left patch, band and receiving
patch by 15 s for the two cases in Fig.~\ref{{fig:barrier_fronts}} on the 200 m grid.
Hatched bars identify THERA-Net; the regions
partition the active rectangle, excluding the exterior. (b) First receiving-
region threshold crossing for the 1 and 2 km rate-neutral bands. Solid circles
denote reference results and dashed crosses THERA-Net. Standard steps are 10 ms at
400 and 200 m and 5 ms at 100 m; squares denote the additional 200 m, 5 ms
reference controls. The 2 km neutral band has no receiver crossing at 400 m.}}
\label{{fig:barrier_energy}}
\end{{figure}}
'''
(ROOT/'barrier_results.tex').write_text(text)
# Abstract, Introduction and Conclusions are edited narrative sections.
# Numerical regeneration must not reintroduce performance figures into them.
print(json.dumps(dict(main_pairs=len(pairs),activated=nact,arrested=narr,
    field_error_percent=[emin,emax]),indent=2))
