"""Compile and audit the original 14-pt TikZ framework schematic."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--audit-only', action='store_true')
args = parser.parse_args()
fig = ROOT / 'figures'
if not args.audit_only:
    executable = shutil.which('pdflatex')
    local = Path('D:/latex/install/miktex/bin/x64/pdflatex.exe')
    if executable is None and local.exists():
        executable = str(local)
    if executable is None:
        raise RuntimeError('Install a TeX distribution with TikZ, standalone and Latin Modern.')
    result = subprocess.run([executable, '-interaction=nonstopmode', '-halt-on-error',
                             'thera_framework.tex'], cwd=fig, capture_output=True)
    (fig / 'framework_compile.txt').write_bytes(result.stdout + result.stderr)
    result.check_returncode()
pdf = fig / 'thera_framework.pdf'
with pdfplumber.open(pdf) as doc:
    assert len(doc.pages) == 1
    page = doc.pages[0]
    assert abs(page.width / 72 - 6.5) < 1e-6
    assert all(c['x0'] >= 0 and c['x1'] <= page.width and c['top'] >= 0
               and c['bottom'] <= page.height for c in page.chars)
    sizes = sorted(set(round(c['size'], 3) for c in page.chars))
    base_bp = 14 * 72 / 72.27
    assert any(abs(size - base_bp) < .002 for size in sizes)
    assert all(abs(size - base_bp) < .002 or abs(size - 10*72/72.27) < .002
               or abs(size - 7*72/72.27) < .002 for size in sizes)
    audit = dict(file=pdf.name, source='thera_framework.tex',
                 width_inches=page.width/72, height_inches=page.height/72,
                 backend='LaTeX/TikZ; scalable Latin Modern', requested_font_points=14,
                 pdf_font_sizes_points=sizes, all_extracted_text_inside_media_box=True,
                 base_text_font_verified=True,
                 note='14 TeX pt = 13.9477 PDF bp. Mathematical scripts use standard smaller sizing. Included in the paper at natural size.',
                 sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                 source_sha256=hashlib.sha256((fig/'thera_framework.tex').read_bytes()).hexdigest())
(fig / 'framework_font_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
print('Verified: one-page vector TikZ figure, 6.5 in width, 14 pt base font, no clipped text.')
