"""Seven LaTeX-rendered manuscript figures from archived, compact plotting data.

Default: regenerate PDFs and QA previews using only ../data/figure_data.npz.
One-time data preparation: --prepare-data [--source ROOT_OF_FAULT_MECHANICS].
This script never trains a network or advances a rupture simulation.
Requires Python, NumPy, Matplotlib, pdfplumber, pypdfium2 and pdflatex/PGF.
All text objects are 14 pt. PDF media boxes retain natural width 6.5 in;
do not rescale these PDFs when including them in a 6.5-in manuscript text block.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
LOCAL_DEPS = WORK / 'research_notes/python_deps'
if LOCAL_DEPS.exists(): sys.path.insert(0, str(LOCAL_DEPS))
MIKTEX = Path('D:/latex/install/miktex/bin/x64')
if MIKTEX.exists(): os.environ['PATH'] = str(MIKTEX) + os.pathsep + os.environ.get('PATH', '')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'scripts/.mplconfig'))
import numpy as np

CASES = ['heterogeneous_D16', 'uniform', 'unseen_D13']
LABELS = [r'$D=1.6$', 'Uniform prestress', r'$D=1.3$']
BLACK = '#191919'
RED = '#bf2b2b'
COLORS = [BLACK, '#1f77a0', RED, '#8c6500']
FONTSIZE = 14.
WIDTH = 6.5
AUDITS = []


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def prepare_data(source):
    sys.path.insert(0, str(source))
    from postprocess_rupture import load_compact
    from source_family_thera import load_source_model, source_family_parameters
    arrays = {}; hashes = {}
    for case in CASES:
        for mode in ['reference', 'thera']:
            path = source / f'results_rupture/compact/{case}_N360_{mode}.npz'
            run = load_compact(path)
            prefix = f'{case}_{mode}_'
            arrays.update({prefix+'arrival': run.final[...,1], prefix+'slip': run.final[...,4],
                           prefix+'stations': run.stations, prefix+'diagnostics': run.diagnostics,
                           prefix+'diagnostic_names': np.asarray(run.diagnostic_names),
                           prefix+'metadata_json': np.asarray(json.dumps(run.meta))})
            if mode == 'reference':
                arrays[case+'_x'] = run.x
                arrays[case+'_y'] = run.y
                arrays[case+'_station_xy'] = np.asarray(run.meta['stations_actual_xy_m'])
            hashes[str(path.relative_to(WORK))] = digest(path)
            del run
        path = source / f'results_rupture/inputs/{case}_N360/fields.npz'
        with np.load(path, allow_pickle=False) as d:
            arrays[case+'_prestress'] = d['initial_stress_Pa']
            arrays[case+'_nucleation'] = d['nucleation_spatial_shape']
        hashes[str(path.relative_to(WORK))] = digest(path)
    path = source / 'results_rupture/source_family/material_family_dense.csv'
    arrays['source_family_dense'] = np.loadtxt(path, delimiter=',', skiprows=1)
    hashes[str(path.relative_to(WORK))] = digest(path)
    model = load_source_model(source/'results_rupture/source_family/source_model.npz')
    parameters = np.stack([np.array([.008,.012,.02,.6]), source_family_parameters(model, np.array(.008))])
    arrays['thermo_parameters'] = parameters
    arrays['thermo_velocity_signed'] = np.r_[-np.geomspace(10.,1e-8,800),0.,np.geomspace(1e-8,10.,800)]
    arrays['thermo_velocity_linear'] = np.linspace(-3.,3.,1001)
    arrays['thermo_velocity_steady'] = np.geomspace(1e-8,10.,1000)
    (ROOT/'data').mkdir(exist_ok=True)
    np.savez_compressed(ROOT/'data/figure_data.npz', **arrays)
    (ROOT/'data/provenance.json').write_text(json.dumps(dict(
        source='Previously completed Nx360 reference/THERA-Net simulations; no new simulations',
        snapshot_policy='Only arrival/final slip, full-step stations/diagnostics, input maps and learned coefficients copied',
        source_sha256=hashes, data_sha256=digest(ROOT/'data/figure_data.npz')),indent=2),encoding='utf-8')


def configure():
    import matplotlib
    matplotlib.use('pgf')
    matplotlib.rcParams.update({
        'pgf.texsystem':'pdflatex', 'pgf.rcfonts':False,
        'pgf.preamble':r'\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}\pagestyle{empty}',
        'text.usetex':True, 'font.family':'serif', 'font.size':FONTSIZE,
        'axes.labelsize':FONTSIZE, 'axes.titlesize':FONTSIZE,
        'xtick.labelsize':FONTSIZE, 'ytick.labelsize':FONTSIZE,
        'legend.fontsize':FONTSIZE, 'figure.titlesize':FONTSIZE,
        'axes.linewidth':.7, 'lines.linewidth':1.35,
        'xtick.major.width':.7, 'ytick.major.width':.7,
        'savefig.bbox':None, 'figure.dpi':120,
        'axes.formatter.useoffset':False,
    })
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    return plt, MaxNLocator


def style(ax, grid=False, box=False):
    ax.spines[['top','right']].set_visible(box)
    ax.tick_params(direction='out', length=3, pad=3)
    ax.grid(False, which='both')
    if grid: ax.grid(axis='y',color='.85',lw=.5)


def pair(ax, time, reference, thera, **kw):
    ax.plot(time,reference,color=BLACK,label='Reference model',lw=1.4,**kw)
    ax.plot(time,thera,color=RED,ls=(0,(5,3)),label='THERA-Net',lw=1.25,**kw)


def global_legend(fig, axes, y=.985):
    handles,labels=axes.get_legend_handles_labels()
    fig.legend(handles[:2],labels[:2],loc='upper center',bbox_to_anchor=(.53,y),
               ncol=2,frameon=False,columnspacing=1.2,handlelength=2.)


def save(fig, name):
    import matplotlib.text as mtext
    # Explicitly enforce every label/tick/legend/panel annotation at 14 pt.
    for t in fig.findobj(mtext.Text): t.set_fontsize(FONTSIZE)
    path=ROOT/'figures'/f'{name}.pdf'
    fig.savefig(path)  # PGF invokes genuine pdflatex; never bbox_inches='tight'.
    texts=[t for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text()]
    assert all(abs(t.get_fontsize()-FONTSIZE)<1e-10 for t in texts)
    import pypdfium2 as pdfium
    import pdfplumber
    from pypdf import PdfReader
    from pypdf.generic import ContentStream
    render_doc=pdfium.PdfDocument(str(path))
    render_doc[0].render(scale=2).to_pil().save(ROOT/'figures/qa'/f'{name}.png')
    render_doc.close()
    doc=pdfplumber.open(path);page=doc.pages[0]
    assert len(doc.pages)==1 and abs(page.width-72*WIDTH)<.02
    spans=[s for s in page.chars if s['text'].strip()]
    outside=[s['text'] for s in spans if s['x0']<-.5 or s['top']<-.5
             or s['x1']>page.width+.5 or s['bottom']>page.height+.5]
    assert not outside, outside
    reader=PdfReader(path)
    font_sizes=sorted(set(float(args[1]) for args,op in ContentStream(reader.pages[0].get_contents(),reader).operations if op==b'Tf'))
    assert any(abs(s-14*72/72.27)<.001 for s in font_sizes)
    AUDITS.append(dict(file=path.name,width_inches=page.width/72,
        height_inches=page.height/72,matplotlib_text_count=len(texts),
        requested_font_points=14,all_text_objects_exactly_14=True,
        backend='Matplotlib PGF + pdflatex; Latin Modern',
        pdf_text_operator_font_sizes_bp=font_sizes,
        note='14 TeX pt = 13.9477 PDF bp (72/72.27 conversion). Mathematical scripts are naturally smaller; no label/legend/tick fontsize is reduced.',
        all_extracted_text_inside_media_box=True,sha256=digest(path)))
    doc.close(); plt.close(fig)
    print('Saved',path.name,flush=True)


def rupture_case(d, case):
    fig=plt.figure(figsize=(WIDTH,7.6))
    # Equal spatial aspect; front is naturally 30 km wide by 15 km tall.
    ax=fig.add_axes([.23,.685,.70,.299])
    x,y=d[case+'_x']/1000,d[case+'_y']/1000
    ref=d[case+'_reference_arrival']; pred=d[case+'_thera_arrival']
    for arr,col,ls in [(ref,BLACK,'solid'),(pred,RED,'dashed')]:
        cs=ax.contour(x,y,np.ma.masked_where(arr<0,arr).T,
                      levels=np.arange(.5,15.01,.5),colors=col,linestyles=ls,linewidths=.65)
        if col==BLACK:
            labs=ax.clabel(cs,levels=[2.,4.,6.,8.,10.,12.,14.],fmt=lambda v:rf'${v:g}$',fontsize=14)
            for lab in labs:
                lx,ly=lab.get_position()
                if abs(lx)>13 or abs(ly)>5.7: lab.remove()
    for label,(sx,sy) in zip('ABCDEF',d[case+'_station_xy']/1000):
        ax.scatter(sx,sy,s=16,color=BLACK,zorder=6)
        offset=(-5,4) if sx>12 else (4,4)
        ax.annotate(label,(sx,sy),xytext=offset,textcoords='offset points',zorder=7,
                    ha='right' if sx>12 else 'left',
                    bbox=dict(facecolor='white',edgecolor='none',pad=.1,alpha=.75))
    ax.set(xlim=(-15,15),ylim=(-7.5,7.5),xticks=[-15,0,15],yticks=[-7.5,0,7.5],
           xlabel=r'$x$ (km)',ylabel=r'$y$ (km)')
    ax.set_aspect('equal',adjustable='box')
    ax.text(.025,.98,'(a)',transform=ax.transAxes,va='top',bbox=dict(facecolor='white',edgecolor='none',pad=.1))
    # No line smoothing or decimation: every saved integration-step station value.
    rs=d[case+'_reference_stations']; ps=d[case+'_thera_stations']
    axes=[]
    for j in range(6):
        row,col=divmod(j,2)
        aa=fig.add_axes([.14+.465*col,.072+.183*(2-row),.355,.135])
        pair(aa,rs[:,0],rs[:,j+1],ps[:,j+1])
        aa.set_xlim(0,15); aa.set_xticks([0,5,10,15])
        aa.set_ylim(min(0,float(rs[:,j+1].min()),float(ps[:,j+1].min())),
                    max(float(rs[:,j+1].max()),float(ps[:,j+1].max()))*1.15)
        aa.yaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=2))
        aa.text(.97,.97,rf'({chr(98+j)}) {"ABCDEF"[j]}',transform=aa.transAxes,va='top',ha='right')
        if row<2: aa.tick_params(labelbottom=False)
        else: aa.set_xlabel(r'$t$ (s)')
        style(aa); axes.append(aa)
    fig.text(.015,.335,r'$V$ (m s$^{-1}$)',rotation=90,va='center')
    global_legend(fig,axes[0],y=.622)
    save(fig,'rupture_'+case)


def inputs(d):
    fig=plt.figure(figsize=(WIDTH,7.6))
    fields=[d[c+'_prestress']/1e6 for c in CASES]
    low=5*np.floor(min(a.min() for a in fields)/5)
    high=5*np.ceil(max(a.max() for a in fields)/5)
    for j,(case,label,field) in enumerate(zip(CASES,LABELS,fields)):
        ax=fig.add_axes([.13,.705-.295*j,.66,.282])
        x,y=d[case+'_x']/1000,d[case+'_y']/1000
        im=ax.imshow(field.T,extent=[x[0]-.1,x[-1]+.1,y[0]-.1,y[-1]+.1],
                     origin='lower',cmap='viridis',vmin=low,vmax=high,interpolation='none',aspect='equal',rasterized=True)
        ax.plot([-15,15,15,-15,-15],[-7.5,-7.5,7.5,7.5,-7.5],color='white',ls='--',lw=.9)
        ix,iy=np.unravel_index(np.argmax(d[case+'_nucleation']),d[case+'_nucleation'].shape)
        ax.plot(x[ix],y[iy],marker='*',ms=10,color='white',mec=BLACK,mew=.7)
        ax.set(xticks=[-30,0,30],yticks=[-15,0,15],ylabel=r'$y$ (km)')
        ax.text(.03,.97,rf'({chr(97+j)}) '+label,transform=ax.transAxes,va='top',
                bbox=dict(facecolor='white',edgecolor='none',pad=.5,alpha=.85))
        if j<2: ax.tick_params(labelbottom=False)
        else: ax.set_xlabel(r'$x$ (km)')
    cax=fig.add_axes([.82,.15,.035,.77])
    cb=fig.colorbar(im,cax=cax,ticks=np.linspace(low,high,5))
    cb.set_label(r'Initial shear stress (MPa)',labelpad=8)
    save(fig,'rupture_inputs')


def source_fit(d):
    a=d['source_family_dense'][:,0]; pred=d['source_family_dense'][:,1:]
    truth=np.stack([a,np.full_like(a,.012),np.full_like(a,.02),np.full_like(a,.6)],axis=1)
    fig,axs=plt.subplots(1,2,figsize=(WIDTH,3.65),sharex=True)
    fig.subplots_adjust(left=.13,right=.97,bottom=.23,top=.76,wspace=.58)
    pair(axs[0],a*1e3,(truth[:,1]-truth[:,0])*1e3,(pred[:,1]-pred[:,0])*1e3)
    axs[0].axhline(0,color='.5',lw=.6)
    axs[0].set_ylabel(r'$(b-a)\,10^{3}$')
    axs[0].text(.97,.95,'(a)',transform=axs[0].transAxes,va='top',ha='right')
    axs[0].legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.02),
                  handlelength=1.6,borderaxespad=0)
    rel=(pred-truth)/truth*1e6
    for k,label in enumerate([r'$a$',r'$b$',r'$D_c$',r'$f_* $']):
        axs[1].plot(a*1e3,rel[:,k],color=COLORS[k],ls=['-','--','-.',':'][k],label=label)
    axs[1].set_ylabel('Parameter error (ppm)')
    axs[1].text(.03,.95,'(b)',transform=axs[1].transAxes,va='top')
    axs[1].legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.02),
                  columnspacing=.8,handlelength=1.5,borderaxespad=0)
    for ax in axs:
        ax.set(xlabel=r'Prescribed $a$ ($10^{-3}$)',xlim=(6,18),xticks=[6,12,18])
        style(ax,box=True)
    save(fig,'source_family_fit')


def thermo(d):
    fig,axs=plt.subplots(1,3,figsize=(WIDTH,3.2))
    fig.subplots_adjust(left=.13,right=.975,bottom=.25,top=.79,wspace=.70)
    params=d['thermo_parameters']
    vs=d['thermo_velocity_signed']; vl=d['thermo_velocity_linear']; vss=d['thermo_velocity_steady']
    curves=[]
    for a,b,L,f in params:
        c=2e-6*np.exp(-(f+b*np.log(1e-6/L))/a) # frozen theta=1 s
        traction=a*120e6*np.arcsinh(vs/c)
        phi=a*120e6*(vl*np.arcsinh(vl/c)-np.hypot(vl,c)+c)
        th=L/vss
        friction=a*np.arcsinh(vss/(2e-6)*np.exp((f+b*np.log(1e-6*th/L))/a))
        curves.append((traction/1e6,phi/1e6,friction))
    for j,(ax,v) in enumerate(zip(axs,[vs,vl,vss])):
        pair(ax,v,curves[0][j],curves[1][j]); style(ax,box=True)
        xpos=[.025,.5,.97][j]
        ax.text(xpos,.95,rf'({chr(97+j)})',transform=ax.transAxes,
                va='top',ha=['left','center','right'][j])
        ax.set_xlabel(r'$V$ (m s$^{-1}$)')
    axs[0].set_xscale('symlog',linthresh=1e-6)
    axs[0].set(xticks=[-10,0,10],ylabel=r'$\tau$ (MPa)',ylim=(-90,90))
    axs[0].set_yticks([-80,0,80])
    axs[1].set(xlim=(-3,3),xticks=[-3,0,3],ylabel=r'$\Phi$ (MW m$^{-2}$)')
    axs[1].yaxis.set_major_locator(MaxNLocator(nbins=3))
    axs[2].set_xscale('log');axs[2].set(xlim=(1e-8,10),xticks=[1e-8,1e-4,1],ylabel=r'$\tau_{\rm ss}/p$')
    axs[2].yaxis.set_major_locator(MaxNLocator(nbins=3))
    global_legend(fig,axs[0],y=.99)
    save(fig,'thermodynamic_structure')


def diagnostic(d,case,mode,name):
    prefix=f'{case}_{mode}_'
    names=list(d[prefix+'diagnostic_names'])
    return d[prefix+'diagnostics'][:,names.index(name)]


def diagnostics(d):
    fig,axs=plt.subplots(3,2,figsize=(WIDTH,7.6),sharex=True)
    fig.subplots_adjust(left=.16,right=.97,bottom=.09,top=.89,hspace=.36,wspace=.75)
    for j,(case,label) in enumerate(zip(CASES,LABELS)):
        t=diagnostic(d,case,'reference','time')
        for k,(key,scale) in enumerate([('moment_rate_Nm_s',1e18),('integrated_friction_power_W',1e17)]):
            pair(axs[j,k],t,diagnostic(d,case,'reference',key)/scale,diagnostic(d,case,'thera',key)/scale)
            style(axs[j,k]);axs[j,k].set_xlim(0,15);axs[j,k].set_xticks([0,5,10,15])
            axs[j,k].yaxis.set_major_locator(MaxNLocator(nbins=3))
            axs[j,k].text(.035,.98,rf'({chr(97+2*j+k)})',transform=axs[j,k].transAxes,va='top')
        axs[j,0].set_title(label,pad=6,loc='left')
        axs[j,0].set_ylabel(r'$\dot M_0$'+'\n'+r'($10^{18}$ N m s$^{-1}$)')
        axs[j,1].set_ylabel(r'$\int_\Gamma\tau V\,dA$'+'\n'+r'($10^{17}$ W)')
    for ax in axs[-1]: ax.set_xlabel(r'$t$ (s)')
    global_legend(fig,axs[0,0],y=.985)
    save(fig,'rupture_diagnostics')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-data',action='store_true')
    parser.add_argument('--source',type=Path,default=WORK/'src/fault_mechanics')
    parser.add_argument('--only',default='',help='Optional figure stem for layout iteration')
    args=parser.parse_args()
    (ROOT/'figures/qa').mkdir(parents=True,exist_ok=True)
    if args.prepare_data: prepare_data(args.source)
    plt,MaxNLocator=configure()
    data_path=ROOT/'data/figure_data.npz'
    with np.load(data_path,allow_pickle=False) as d:
        for stem,func in [('source_family_fit',source_fit),('thermodynamic_structure',thermo),
                          ('rupture_inputs',inputs),('rupture_diagnostics',diagnostics)]:
            if not args.only or args.only==stem: func(d)
        for case in CASES:
            if not args.only or args.only=='rupture_'+case: rupture_case(d,case)
    audit_path=ROOT/'figures/font_and_render_audit.json'
    previous=json.loads(audit_path.read_text()) if args.only and audit_path.exists() else []
    audit_path.write_text(json.dumps([a for a in previous if a['file'] not in {b['file'] for b in AUDITS}]+AUDITS,indent=2),encoding='utf-8')
