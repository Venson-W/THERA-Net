REGENERATING THE MANUSCRIPT FIGURES

From any directory, run:
  python scripts/build_figures.py
The script resolves data paths relative to itself. Install NumPy, Matplotlib,
pdfplumber, pypdf, pypdfium2 and a working pdflatex with PGF, Latin Modern and
amsmath. MiKTeX or TeX Live is suitable. pdflatex must be on PATH; the local
Windows D:/latex/install/miktex/bin/x64 installation is detected if present.

Default execution reads only data/figure_data.npz. It does not train, simulate,
or modify original solver outputs. --prepare-data is an optional one-time
developer operation requiring the original Fault mechanics project.

Each output PDF has a natural width of exactly 6.5 in. All figure labels,
annotations, ticks and legends are explicitly 14 TeX pt, rendered with genuine
LaTeX through Matplotlib's PGF backend and scalable Latin Modern. Mathematical
subscripts/superscripts use normal LaTeX script sizing. A 14-pt TeX font appears
as 13.9477 PDF big points because TeX uses 72.27 pt/in and PDF uses72 bp/in.
Do not resize figures below their natural width in a 6.5-in text block.

PDF curves, contours, axes and text are vector elements. The input stress maps
embed their original cellwise sampled raster fields with vector annotations;
no interpolation or smoothing is applied to the stress values. QA PNGs are
rendered from the completed PDFs at 144dpi and are not manuscript figure inputs.
font_and_render_audit.json records exact PDF dimensions, verified text sizes,
text-boundary checks, and SHA256 hashes. Every page was also visually inspected.

Figure content/caption facts:
- source_family_fit.pdf: (a) b-a from reference and saved learned parameters;
  (b) signed relative parameter errors in parts per million, across prescribed
  a in[.006,.018]. This is a parameter audit, not event-prediction training.
- thermodynamic_structure.pdf: representative source descriptor a=.008;
  prescribed reference parameters(.008,.012,.02m,.6), saved neural outputs,
  p=120MPa. (a) traction at frozen theta=1s using a symmetric-log V axis;
  (b) convex potential at theta=1s; (c) steady friction with theta=Dc/|V|.
  This figure illustrates a single velocity-weakening material, not two regimes.
- rupture_inputs.pdf: actual200m initial stress samples; dashed30x15km VW
  rectangle and nucleation stars; common50--100MPa color scale. D16/D13 are
  the new seeded realizations1101/1301; uniform is75MPa.
- rupture_heterogeneous_D16.pdf, rupture_uniform.pdf, rupture_unseen_D13.pdf:
  each uses actual200m, dt=.01s paired runs. (a) first arrival at|V|=.5m/s,
  contour interval.5s, arrival tracked every integration step with linear
  crossing interpolation. (b)--(g) raw every-step signed slip-rate histories
  at A--F. Reference black; THERA-Net red dashed. No smoothing, waveform decimation
  or dynamic alignment. All station coordinates are retained in local data.
- rupture_diagnostics.pdf: case rows D16/uniform/D13; left whole-fault moment
  rate, right whole-fault integrated friction power, every integration step.
  It does not show minimum theta or minimum local friction power. Positive
  integrated power is not a global discrete-energy certificate and does not
  remove the small negative near-rest local-power floor discussed in the text.

Figure 2 uses a 1-by-3 layout (thermodynamic_structure, height 3.2 in).
Figure 3 uses a 1-by-2 layout (source_family_fit, height 3.65 in).
Both use all four axis borders, no background grid, and unchanged 14 pt text.
Other numerical figure heights remain 7.6 in. Include natural-width
PDFs without bbox trimming. Seven figure PDFs plus local compact data support
standalone figure regeneration after sharing this manuscript folder.

ADDITIONAL TIKZ FRAMEWORK FIGURE
Figure1 is figures/thera_framework.tex, an original two-panel schematic.
Compile with python scripts/build_framework.py; --audit-only checks an existing
compiled figure. This is separate from build_figures.py, which regenerates the
seven numerical/analytical plots. The new figure uses14TeXpt text throughout,
with standard math-script sizing, and has natural dimensions6.5x7.559in.
framework_font_audit.json stores source/PDF hashes, dimensions and font checks.

BARRIER FIGURES
Run python scripts/build_barrier_figures.py using data/barrier_figure_data.npz
and data/barrier_metrics.json. All labels are real LaTeX at 14 pt, at natural
6.5-inch width. Colors show unsmoothed cellwise fields; curves, contours and
labels are vector elements. Four added figures show the complete outcome map,
representative fronts/stations, centerline space-time histories, and regional
work/refinement. Every main case is tabulated, including the resolution-sensitive
2 km neutral band. The separate barrier_figure_audit.json records font sizes,
dimensions, text-boundary checks and hashes.

EDITORIAL SECTIONS
write_barrier_section.py regenerates quantitative results and tables only.
It does not overwrite the edited abstract, Introduction or Conclusions.
