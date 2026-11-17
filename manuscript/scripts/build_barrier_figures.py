"""Four 14-pt LaTeX/PGF figures from archived barrier data; no solver needed."""
from pathlib import Path
import json,sys
import numpy as np
import build_figures as b

ROOT=Path(__file__).resolve().parents[1]
b.plt,b.MaxNLocator=b.configure()
plt=b.plt
D=np.load(ROOT/'data/barrier_figure_data.npz',allow_pickle=False)
M=json.loads((ROOT/'data/barrier_metrics.json').read_text())
preview='--preview' in sys.argv
assert M['all_required_complete'] or (preview and len(M['main_pairs'])==12)
BLACK=b.BLACK; RED=b.RED

def key(w,a,mode='reference'):
    return f'w{w:g}_a{a:.3f}'.replace('.','p')+'_'+mode

def setup_regimes():
    from matplotlib.patches import Rectangle
    fig=plt.figure(figsize=(6.5,7.5))
    ax=fig.add_axes([.13,.64,.62,.27])
    x,y=D['setup_x']/1000,D['setup_y']/1000;dx=x[1]-x[0]
    im=ax.imshow(D['setup_a_minus_b'].T,origin='lower',extent=(x[0]-dx/2,x[-1]+dx/2,y[0]-dx/2,y[-1]+dx/2),
        cmap='coolwarm',vmin=-.004,vmax=.004,interpolation='nearest')
    ax.plot(-9,0,marker='*',ms=12,color='black')
    ax.axvline(-1.5,ls=':',lw=1,color='black');ax.axvline(1.5,ls=':',lw=1,color='black')
    ax.add_patch(Rectangle((-15,-7.5),30,15,fill=False,ls='--',lw=.7,color='black'))
    ax.set(xlim=(-18,18),ylim=(-10.5,10.5),xlabel='$x$ (km)',ylabel='$y$ (km)',xticks=[-15,0,15],yticks=[-10,0,10],aspect='equal')
    ax.set_title(r'(a) Segmented fault: $w=2$ km',loc='left',pad=9)
    cax=fig.add_axes([.89,.65,.025,.25]);cb=fig.colorbar(im,cax=cax,ticks=[-.004,0,.004])
    cb.ax.set_yticklabels(['$-0.004$','$0$','$0.004$']);cb.ax.yaxis.set_ticks_position('left')
    cb.ax.set_title('$a_s-b$',pad=9)
    ax=fig.add_axes([.17,.14,.72,.34])
    widths=[1,2,4]; aa=[.010,.011,.012,.014]
    values=np.zeros((3,4));labels={}
    for p in M['main_pairs']:
        i=widths.index(p['width_km']);j=aa.index(p['a_barrier'])
        r=next(r for r in M['runs'] if r['name']==p['reference'])
        s=next(r for r in M['runs'] if r['name']==p['thera'])
        values[i,j]=r['receiver_fraction']
        labels[i,j]=f"{r['receiver_fraction']:.2f}/{s['receiver_fraction']:.2f}"
    ax.imshow(values,origin='lower',cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='nearest')
    for (i,j),label in labels.items(): ax.text(j,i,label,ha='center',va='center',color='white' if values[i,j]>.5 else BLACK)
    ax.set(xticks=range(4),xticklabels=['0.010','0.011','0.012','0.014'],yticks=range(3),yticklabels=['1','2','4'],
        xlabel=r'Band coefficient $a_{\rm B}$',ylabel='$w$ (km)')
    ax.set_title(r'(b) Receiver activated area fraction',loc='left',pad=10)
    fig.text(.53,.025,'Reference / THERA-Net',ha='center')
    b.save(fig,'barrier_setup_regimes')

def fronts_stations():
    fig=plt.figure(figsize=(6.5,7.8))
    for index,(w,aa,ypos) in enumerate([(1,.012,.64),(2,.014,.33)]):
        ax=fig.add_axes([.20,ypos,.64,.25])
        for mode,color,style in [('reference',BLACK,'solid'),('thera',RED,'dashed')]:
            k=key(w,aa,mode);arrival=D[k+'_final'][...,1]
            ax.contour(D[k+'_x']/1000,D[k+'_y']/1000,np.ma.masked_less(arrival,0).T,
                levels=np.arange(1,15.01,1),colors=color,linestyles=style,linewidths=.8)
        ax.axvspan(-w/2-.5,w/2+.5,color='.8',alpha=.35,zorder=-1)
        ax.plot([-9,8],[0,0],ls='none',marker='o',ms=4,color='black')
        ax.set(xlim=(-15,15),ylim=(-7.5,7.5),ylabel='$y$ (km)',xticks=[-15,0,15],yticks=[-5,0,5],aspect='equal')
        ax.set_title(f'({chr(97+index)}) '+rf'$w={w}$ km, $a_{{\rm B}}={aa:.3f}$',loc='left',pad=6)
        if index==1:ax.set_xlabel('$x$ (km)')
        else:ax.tick_params(labelbottom=False)
    ax=fig.add_axes([.14,.065,.77,.15])
    for w,aa,col in [(1,.012,'#176b91'),(2,.014,'#ac4e10')]:
        for mode,ls in [('reference','-'),('thera','--')]:
            dat=D[key(w,aa,mode)+'_stations']
            ax.semilogy(dat['time'],np.maximum(abs(dat['E']),1e-14),color=col,ls=ls,
                label=f'{w} km: '+('reference' if mode=='reference' else 'THERA-Net'))
    ax.set(xlim=(0,15),ylim=(1e-12,30),xticks=[0,5,10,15],yticks=[1e-12,1e-6,1],
        xlabel='Time (s)',ylabel=r'$|V|$ (m/s)')
    ax.set_title('(c) Receiving station: $(8,0)$ km',loc='left',pad=5)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,1.005),ncol=2,frameon=False,
        handlelength=1.8,columnspacing=1.)
    b.style(ax)
    b.save(fig,'barrier_fronts_stations')

def spacetime():
    from matplotlib.lines import Line2D
    fig=plt.figure(figsize=(6.5,7.6))
    for i,(w,aa,ypos) in enumerate([(1,.012,.57),(2,.014,.20)]):
        ax=fig.add_axes([.14,ypos,.79,.29]);k=key(w,aa)
        x=D[k+'_x']/1000;t=D[k+'_time'];v=D[k+'_centerline_velocity']
        dx=x[1]-x[0];dt=t[1]-t[0]
        im=ax.imshow(np.log10(np.maximum(abs(v),1e-8)),origin='lower',
            extent=(x[0]-dx/2,x[-1]+dx/2,t[0]-dt/2,t[-1]+dt/2),cmap='viridis',vmin=-8,vmax=1,
            aspect='auto',interpolation='nearest')
        for mode,col,ls in [('reference','white','solid'),('thera','#e44c43','dashed')]:
            q=key(w,aa,mode)
            arrivals=D[q+'_final'][:,D[q+'_final'].shape[1]//2,1]
            ax.plot(x,np.ma.masked_less(arrivals,0),color=col,ls='-' if mode=='reference' else '--',lw=1.1)
        ax.axvline(-w/2-.5,ls=':',color='.6');ax.axvline(w/2+.5,ls=':',color='.6')
        ax.set(xlim=(-15,15),ylim=(0,15),xticks=[-15,0,15],yticks=[0,5,10,15],ylabel='Time (s)')
        ax.set_title(f'({chr(97+i)}) '+rf'$w={w}$ km, $a_{{\rm B}}={aa:.3f}$, $y=0$',loc='left',pad=7)
        if i==1:ax.set_xlabel('$x$ (km)')
    cax=fig.add_axes([.20,.09,.65,.025]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-8,-5,-2,1])
    cb.set_label(r'$\log_{10}(|V_{\rm ref}|/[\mathrm{m/s}])$',labelpad=3)
    fig.legend([Line2D([],[],color='.4'),Line2D([],[],color='#e44c43',ls='--')],
        ['Reference threshold','THERA-Net threshold'],loc='upper center',bbox_to_anchor=(.54,.99),ncol=1,frameon=False)
    b.save(fig,'barrier_spacetime')

def energy_refinement():
    fig=plt.figure(figsize=(6.5,7.7))
    ax=fig.add_axes([.15,.59,.77,.30]);colors=['#176b91','#c77a20','#805398']
    for i,(w,aa) in enumerate([(1,.012),(2,.014)]):
        for j,mode in enumerate(['reference','thera']):
            r=D[key(w,aa,mode)+'_regional'];dt=np.diff(r['time']).mean()
            vals=[r[n].sum()*dt/1e16 for n in ['left_power_W','barrier_power_W','right_power_W']]
            bottom=0;x=i*3+j
            for val,col,label in zip(vals,colors,['Left patch','Band','Receiver']):
                ax.bar(x,val,bottom=bottom,width=.7,color=col,edgecolor='white',hatch='//' if j else None,
                    label=label if i==j==0 else None);bottom+=val
    ax.set(xticks=[0,1,3,4],xticklabels=['Ref.','THERA-Net','Ref.','THERA-Net'],ylabel=r'$W_j$ ($10^{16}$ J)')
    ax.text(.5,-.23,'1 km, neutral',transform=ax.get_xaxis_transform(),ha='center')
    ax.text(3.5,-.23,'2 km, strengthening',transform=ax.get_xaxis_transform(),ha='center')
    ax.set_title('(a) Dissipated work in the active region',loc='left',pad=8)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.30),ncol=3,frameon=False,columnspacing=.6,handlelength=1.2)
    b.style(ax)
    ax=fig.add_axes([.15,.13,.77,.27])
    records=[r for r in M['pilot_runs']+M['runs'] if r['a_barrier']==.012]
    for w,col in [(1,'#176b91'),(2,'#ac4e10')]:
        for mode,marker in [('reference','o'),('thera','x')]:
            seq=sorted([r for r in records if r['width_km']==w and r['mode']==mode and r['receiver_first_arrival_s'] is not None and (r['nx']!=360 or r['dt_s']==.01)],key=lambda r:r['dx_m'])
            ax.plot([r['dx_m'] for r in seq],[r['receiver_first_arrival_s'] for r in seq],color=col,marker=marker,
                ls='-' if mode=='reference' else '--',label=f'$w={w}$ km' if mode=='reference' else None)
        controls=[r for r in records if r['width_km']==w and r['nx']==360 and r['dt_s']==.005]
        if controls and controls[0]['receiver_first_arrival_s'] is not None:
            ax.scatter([200],[controls[0]['receiver_first_arrival_s']],marker='s',facecolors='none',edgecolors=col,s=65)
    ax.plot([],[],color='.4',ls='--',marker='x',label='THERA-Net')
    ax.set(xlim=(75,425),xticks=[100,200,400],xlabel='Grid spacing (m)',ylabel='Receiver onset (s)')
    ax.set_title('(b) Rate-neutral bands: receiver onset',loc='left',pad=8)
    ax.legend(loc='upper right',frameon=False)
    b.style(ax)
    fig.text(.53,.025,'2 km neutral band: no receiver onset on the 400 m grid.',ha='center')
    b.save(fig,'barrier_energy_refinement')

setup_regimes();fronts_stations();spacetime()
if not preview: energy_refinement()
(ROOT/'figures/barrier_figure_audit.json').write_text(json.dumps(b.AUDITS,indent=2)+'\n')
