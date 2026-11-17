"""Render the selected source-family checkpoint audit, without retraining."""
from pathlib import Path
import os
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research_notes"/"python_deps"))
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).parent/".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from source_family_pinet import DEFAULT_OUTPUT,load_source_model,source_family_parameters,source_truth

model=load_source_model()
a=np.linspace(.006,.018,4001)
actual=source_truth(a)
pred=source_family_parameters(model,a)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,
                     "axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,2,figsize=(7.3,3.25),constrained_layout=True)
colors=["#176b9a","#c55025","#5f8935","#824a9b"]
for j,(label,color) in enumerate(zip(["$a$","$b$","$D_c$","$f_*$"],colors)):
    axes[0].plot(a,100*(pred[:,j]-actual[:,j])/actual[:,j],color=color,label=label,lw=1.2)
axes[0].axhline(0,color="black",lw=.6)
axes[0].set(xlabel="Prescribed source coefficient $a_s$",ylabel="Signed coefficient error (%)")
axes[0].legend(ncol=2,fontsize=8)
axes[1].plot(a,1e3*(actual[:,1]-actual[:,0]),color="black",lw=1.7,label="Prescribed family")
axes[1].plot(a,1e3*(pred[:,1]-pred[:,0]),color=colors[0],ls="--",lw=1.4,label="Neural family")
endpoints=np.array([.0065,.007,.008,.016,.017,.0175])
endpred=source_family_parameters(model,endpoints)
axes[1].scatter(endpoints,1e3*(endpred[:,1]-endpred[:,0]),s=23,marker="o",facecolors="none",
                edgecolors=colors[1],label="Held-out source endpoints",zorder=4)
axes[1].axhline(0,color="black",lw=.6)
axes[1].set(xlabel="Prescribed source coefficient $a_s$",ylabel="$10^3 (b-a)$")
axes[1].legend(fontsize=7)
for ax in axes:
    ax.grid(alpha=.15)
    ax.ticklabel_format(axis="x",style="plain",useOffset=False)
fig.savefig(ROOT/"manuscript"/"figures"/"source_family_fit.pdf",bbox_inches="tight")
fig.savefig(DEFAULT_OUTPUT/"source_family_fit.png",dpi=200,bbox_inches="tight")
plt.close(fig)
