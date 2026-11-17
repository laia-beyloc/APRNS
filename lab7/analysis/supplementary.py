"""Anàlisi complementària de l'estudi d'ablació (no pre-registrada).

Complementa analysis.py amb: tests de permutació aparellats exactes (2^10 assignacions de
signe) amb correcció de Holm, resums robustos (mediana, IQM), taxa d'aterratges reeixits
(retorn ≥ 200) i d'episodis truncats (planeig), corbes per llavor, biaix DDPG vs TD3
(exploratori, llavor 0), relació biaix-retorn i comptatge de caigudes de rendiment.

Ús (des de lab7/):  python -m analysis.supplementary --results results --out analysis/supplementary
"""
import argparse, json, csv, itertools
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--results", type=Path, default=Path("results"))
parser.add_argument("--out", type=Path, default=Path("analysis/supplementary"))
args = parser.parse_args()
R, OUT = args.results, args.out; OUT.mkdir(parents=True, exist_ok=True)
V = ["full","no_minimum","no_smoothing","no_delay"]
def rd(p): return list(csv.DictReader(open(p, encoding="utf-8")))
runs = {v:{int(d.name[5:]): rd(d/"evaluations.csv") for d in (R/v).glob("seed_*")} for v in V+["ddpg"]}
T=300_000
def metrics(rows):
    x=np.array([float(r["transitions"]) for r in rows]); y=np.array([float(r["eval_mean_return"]) for r in rows])
    rets=np.array([x_ for r in rows[-10:] for x_ in json.loads(r["eval_returns"])])
    lens=np.array([float(r["eval_mean_length"]) for r in rows[-10:]])
    return dict(last=y[-10:].mean(), auc=np.trapezoid(y,x)/T, succ=(rets>=200).mean(), hover=(lens>=999).mean())
M={v:{s:metrics(r) for s,r in runs[v].items()} for v in V}
def perm_p(d):
    d=np.asarray(d); obs=abs(d.mean())
    signs=np.array(list(itertools.product([1,-1],repeat=len(d))))
    return ((np.abs((signs*d).mean(1))>=obs-1e-12).mean())
def iqm(x):
    x=np.sort(x); n=len(x); k=int(np.floor(n*0.25)); return x[k:n-k].mean()
lines=["metric,variant,mean_diff,median_diff,wins_of_10,perm_p,holm_p"]
for met in ["last","auc","succ","hover"]:
    res=[]
    for v in V[1:]:
        d=np.array([M[v][s][met]-M["full"][s][met] for s in range(10)])
        res.append([v,d.mean(),np.median(d),int((d>0).sum()),perm_p(d)])
    order=np.argsort([r[4] for r in res]); m=len(res); holm=[0]*m; run=0
    for rank,i in enumerate(order):
        run=max(run,min(1,(m-rank)*res[i][4])); holm[i]=run
    for r,h in zip(res,holm): lines.append(f"{met},{r[0]},{r[1]:.3f},{r[2]:.3f},{r[3]},{r[4]:.4f},{h:.4f}")
open(OUT/"paired_permutation_tests.csv","w").write("\n".join(lines)+"\n")
print("\n".join(lines))
lines=["variant,mean_last,median_last,iqm_last,mean_auc,iqm_auc,mean_success,mean_hover,seeds_majority_landing"]
for v in V:
    a=lambda k: np.array([M[v][s][k] for s in range(10)])
    lines.append(f"{v},{a('last').mean():.1f},{np.median(a('last')):.1f},{iqm(a('last')):.1f},{a('auc').mean():.1f},{iqm(a('auc')):.1f},{a('succ').mean():.3f},{a('hover').mean():.3f},{int((a('succ')>=0.5).sum())}")
open(OUT/"robust_summary.csv","w").write("\n".join(lines)+"\n"); print("\n".join(lines))
# corbes de comportament: fracció d'episodis d'èxit i truncats al llarg de l'entrenament
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for v in V:
    S=[];H=[]
    for s in range(10):
        rows=runs[v][s]
        S.append([np.mean(np.array(json.loads(r["eval_returns"]))>=200) for r in rows])
        H.append([float(r["eval_mean_length"]) for r in rows])
    x=np.array([float(r["transitions"]) for r in runs[v][0]])
    k=10; sm=lambda a: np.convolve(np.mean(a,0),np.ones(k)/k,mode="valid")
    axes[0].plot(x[k-1:],sm(S),label=v); axes[1].plot(x[k-1:],sm(H),label=v)
axes[0].set(xlabel="Transicions",ylabel="Fracció episodis amb retorn ≥ 200",title="Aterratges reeixits (mitjana 10 llavors, mòbil 10)")
axes[1].set(xlabel="Transicions",ylabel="Durada mitjana episodi d'avaluació",title="Durada (1000 = truncat: planeja)")
axes[0].legend(); fig.tight_layout(); fig.savefig(OUT/"success_and_length.png",dpi=150); plt.close(fig)
# corbes per llavor
fig,axes=plt.subplots(1,4,figsize=(14,3.5),sharey=True)
for ax,v in zip(axes,V):
    for s in range(10):
        rows=runs[v][s]; x=[float(r["transitions"]) for r in rows]; y=np.convolve([float(r["eval_mean_return"]) for r in rows],np.ones(5)/5,mode="same")
        ax.plot(x,y,lw=0.8)
    ax.axhline(200,color="k",ls="--",lw=0.7); ax.set(title=v,xlabel="Transicions")
axes[0].set_ylim(-400,300); axes[0].set_ylabel("Retorn avaluació (mòbil 5)")
fig.tight_layout(); fig.savefig(OUT/"per_seed_curves.png",dpi=150); plt.close(fig)
# DDPG vs full llavor 0: biaix
fig,ax=plt.subplots(figsize=(7,3.5))
for v in ["ddpg","full"]:
    rows=runs[v][0]; ax.plot([float(r["transitions"]) for r in rows],[float(r["bias_all"]) for r in rows],label=f"{v} (llavor 0)")
ax.axhline(0,color="k",lw=0.7); ax.set(xlabel="Transicions",ylabel="Q(s,μ(s)) − G",ylim=(-100,200),title="Biaix d'estimació: DDPG vs TD3 (exploratori)")
ax.legend(); fig.tight_layout(); fig.savefig(OUT/"ddpg_vs_td3_bias.png",dpi=150); plt.close(fig)
# relació biaix final - retorn final entre execucions TD3
bias={v:[np.nanmean([float(r["bias_all"]) for r in runs[v][s][-10:]]) for s in range(10)] for v in V}
fig,ax=plt.subplots(figsize=(5,4))
allb=[];allr=[]
for v in V:
    r=[M[v][s]["last"] for s in range(10)]; ax.scatter(bias[v],r,label=v,s=18); allb+=bias[v]; allr+=r
ax.set(xlabel="Biaix final (Q − G)",ylabel="Retorn final (últimes 10)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT/"bias_vs_return.png",dpi=150); plt.close(fig)
rk=lambda a: np.argsort(np.argsort(a)); print("Spearman biaix-retorn (40 exec.):",np.corrcoef(rk(allb),rk(allr))[0,1])
for v in V: print(v,"Spearman intra-variant:",round(np.corrcoef(rk(bias[v]),rk([M[v][s]["last"] for s in range(10)]))[0,1],2))
# inestabilitat: caigudes > 200 entre avaluacions consecutives després de 50k
for v in V:
    c=[int(np.sum(np.diff([float(r["eval_mean_return"]) for r in runs[v][s][50:]])<-200)) for s in range(10)]
    w=[float(json.load(open(R/v/f"seed_{s}"/"config.json"))["wall_time_s"]) for s in range(10)]
    print(v,"caigudes>200:",sum(c),c," temps mitjà h:",round(np.mean(w)/3600,2))
