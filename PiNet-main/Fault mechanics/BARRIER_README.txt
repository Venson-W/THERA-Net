TITAN-FAULT: SEGMENTED-FAULT RUPTURE STUDY

This is a new controlled single-event application, motivated by Kaneko,
Avouac and Lapusta (2010), DOI 10.1038/ngeo843. It is not an exact reproduction
of their earthquake-cycle model or of the supplied FNO preprint.

PHYSICAL SETUP
The existing 72 x 36 km periodic spectral domain contains a 30 x 15 km active
rectangle. Two weakening patches (a=.008, b=.012) are separated by an internal
frictional band. The band's flat width is w, with a fixed 500 m smooth edge
transition on each side. Exterior a=.016. Reference Dc=.02 m, f*=.6, p=120 MPa,
and uniform initial shear traction=75 MPa. A 3 km radius, 25 MPa nucleation
perturbation centered at (-9,0) km rises over 1 s and then persists.
The same reference-inverted initial state and prestress are given to each pair.
The neural branch uses all four outputs of the original frozen material map.
No retraining or rupture-label fitting is performed.

SWEEP AND CONTROLS
pilot.json: 6 reference runs, including a no-band control.
pilot_strength.json: 15 reference runs used to locate the transition.
main.json: 12 configurations, each with reference and TITAN-Fault (24 runs).
  w=1,2,4 km; a_B=.010,.011,.012,.014; dx=200 m; dt=10 ms.
refinement.json: 1 and 2 km neutral bands, paired at dx=100 m, dt=5 ms (4 runs).
temporal.json: the same two cases, reference at dx=200 m, dt=5 ms (2 runs).
refinement_strong.json: paired 2 km a_B=.014 band at 100 m, 5 ms (2 runs).
temporal_strong.json: reference 2 km a_B=.014 band at 200 m, 5 ms (1 run).
All integrate to 15 s: 54 trajectories in total, plus a separate original-driver
regression calculation. All pilots and main configurations remain in the records.

Receiver activation is defined by at least 50% of the receiving region reaching
|V|>=0.5 m/s by 15 s. At most 1% is labeled arrest within the observation window;
intermediate fractions are partial activation. Fractions, onset and peak speed
are preserved, so the classification never replaces the underlying measurements.
No claim of permanent arrest or an exact continuum critical width is made.
The 2 km neutral band changes from arrest at 400 m to activation at 200 m:
this numerical sensitivity is included in the manuscript and the complete tables.

FILES AND REPRODUCTION
barrier_cases.py generates analytical and neural coefficient fields.
rupture_solver/barrier_driver.cc adds regional work and receiving-patch diagnostics
to the retained UGUCA adapter; the constitutive law and elastic solver are unchanged.
rupture_solver/run_barrier_suite.py stages computation on WSL ext4 and publishes
completed outputs. The runner stores commands, input/binary hashes, runtime and
exit status. Existing completed results are reused only when hashes agree.
barrier_analysis.py validates shapes, finite states, output times, threshold masks
and agreement of nodal, regional and whole-interface frictional work.
collect_barrier_results.py writes every main result to barrier_metrics.json and
exports compact plotting data for the two illustrated representative cases.
test_barrier_cases.py checks grid-independent physical fields and localized loading.
research_notes/verify_barrier_driver.py compares original/instrumented trajectories.

Use the existing UGUCA version ea1c4c252f8cb119f48602e0b1dba885851c1764.
See research_notes/uguca_build_notes.txt for dependencies and build provenance.
Build UGUCA at research_notes/uguca-build, then configure the adapter project at
research_notes/pinet-rupture-build with uguca_DIR pointing to that UGUCA build.
CMake builds both rupture_driver and barrier_driver. Use WSL Python for the runner;
NumPy is required. Input generation and analysis can run on Windows or Linux.

From the workspace root, examples:
  python research_notes/prepare_barrier_batch.py main --nx 360 --dt .01 --widths 1 2 4 --a .010 .011 .012 .014 --modes reference pinet
  python "PiNet-main/Fault mechanics/rupture_solver/run_barrier_suite.py" "PiNet-main/Fault mechanics/results_barrier/main.json"
  python "PiNet-main/Fault mechanics/collect_barrier_results.py"
The last command requires all listed batches to be complete. --partial produces
an explicitly incomplete progress audit; it must not be used to finalize a paper.

Extract the computational and LaTeX archives into the same parent directory.
The LaTeX folder has scripts/build_barrier_figures.py and
scripts/write_barrier_section.py, which regenerate the figures, numerical tables
and quantitative prose from the audited data. Plot labels use real LaTeX at 14 pt;
field images are cellwise samples with vector contours and labels, without smoothing.

ARCHIVE CONTENTS
The computational archive includes the code, frozen weights and calibration data,
batch specifications, per-input configuration/hash records, complete station and
diagnostic histories, final fields and nodal frictional work. Large full-field
snapshot histories remain in results_barrier/runs in the working project and
can be regenerated from the inputs; they are not included in the compact archive.
The manuscript source archive includes the full results JSON and the compact data
needed for every plotted quantity. No unmatched comparison with published FNO
accuracy or speedup is inferred from these paired constitutive-replacement tests.
