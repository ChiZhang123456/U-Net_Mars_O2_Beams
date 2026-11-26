"""Append native 10:50 O2+ spectrum and simultaneous 2-count response.

Retain the cached three input panels and window markers. Right-hand evidence
compares actual native energy bins against the response for two net counts
in one central O2 mass bin, using the same iv4 calibration as the left figure.
Python quantitative layout, Arial, PDF/PNG, no spectral extrapolation.
"""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs'
os.environ.setdefault('SPACEPY',str(OUT/'runtime'))
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,ListedColormap,BoundaryNorm
import sys
sys.path.insert(0, str(ROOT))
from unet_mars_o2_beams import build_input_channels

def main():
    OUT.mkdir(exist_ok=True)
    with np.load(ROOT/'examples/data/static_20180805_count2.npz') as z:d={k:z[k] for k in z.files}
    t=d['time'];energy=d['energy_ev'];f=d['o2_def'];n=len(t)
    valid=build_input_channels(f,energy,two_count_def=d['two_count_def'])[1].astype(bool)
    ii=int(np.argmin(abs(t-np.datetime64('2018-08-05T10:50'))))
    df=pd.read_csv(ROOT/'examples/data/native_1050_spectrum_count2.csv')
    report={'model_version':'v12_count2','window':512,'inference_stride':51}
    mpl.rcParams.update({'font.family':'Arial','font.size':12,'axes.labelsize':14,'mathtext.fontset':'dejavusans','pdf.fonttype':42})
    fig=plt.figure(figsize=(14.5,9))
    gs=fig.add_gridspec(3,1,left=.085,right=.585,top=.80,bottom=.16,hspace=.14)
    axes=[fig.add_subplot(gs[i,0]) for i in range(3)]
    for ax in axes[1:]:ax.sharex(axes[0])
    channels=[f,valid.astype(float),np.broadcast_to(energy,f.shape)]
    labels=[r'O$_2^+$ DEF',r'O$_2^+$ validity','Energy']
    for i,(ax,values) in enumerate(zip(axes,channels)):
        if i==1:
            cmap=ListedColormap(['#e0e0e0','#2166ac']);norm=BoundaryNorm([-.5,.5,1.5],2);z=values.T
        else:
            cmap=plt.colormaps['turbo'].copy();cmap.set_bad('black');cmap.set_under('black')
            norm=LogNorm(1e4,1e8) if i==0 else LogNorm(.2,3e4)
            z=np.ma.masked_invalid(np.where(values.T>0,values.T,np.nan))
        mesh=ax.pcolormesh(np.arange(1,n+1),energy,z,cmap=cmap,norm=norm,shading='nearest',rasterized=True)
        ax.set(yscale='log',ylim=(energy.min(),energy.max()),xlim=(.5,n+.5))
        ax.set_yticks([1,10,100,1000,10000]);ax.set_ylabel(labels[i]+'\n'+r'$E/q$ (eV/q)',fontsize=14)
        ax.tick_params(which='both',top=True,right=True,direction='in');ax.tick_params(labelbottom=i==2)
        ax.text(.008,.94,chr(97+i),transform=ax.transAxes,va='top',fontweight='bold',fontsize=15,bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1))
        pos=ax.get_position();cb=fig.colorbar(mesh,cax=fig.add_axes([.596,pos.y0,.012,pos.height]));cb.set_label(['DEF','Validity','Energy (eV)'][i])
        if i==1:cb.set_ticks([0,1])
    axes[1].text(.985,.95,'0: Invalid; 1: DEF > 2-count response',transform=axes[1].transAxes,ha='right',va='top',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
    top=fig.add_axes([.085,.815,.5,.16],sharex=axes[0]);top.set_ylim(0,3.7);top.set_axis_off()
    for i,start in enumerate([0,51,102]):
        l,r,y=start+1,start+512,2.7-i
        top.plot([l,r],[y,y],color='.3',lw=1.8)
        for v in [l,r]:top.plot([v,v],[y-.12,y+.12],color='.3',lw=1.8)
        top.text((l+r)/2,y+.18,f'Input window {i+1}: {l}-{r}  (512 × 32 × 3)',ha='center',va='bottom',fontsize=14,color='.3')
    tick_times=np.arange(np.datetime64('2018-08-05T10:30'),np.datetime64('2018-08-05T11:12'),np.timedelta64(10,'m'))
    positions=np.interp(tick_times.astype('datetime64[ns]').astype('int64'),t.astype('int64'),np.arange(1,n+1))
    axes[-1].set_xticks(positions,[str(v)[11:16] for v in tick_times]);axes[-1].set_xlabel('UTC on 2018-08-05',fontsize=14)
    secondary=axes[-1].secondary_xaxis(-.38);secondary.set_xticks(np.arange(100,n+1,100));secondary.set_xlabel('Sample index')
    ax=fig.add_axes([.72,.26,.19125,.49])
    en=df.energy_ev.to_numpy();actual=df.observed_def.to_numpy().copy();actual[actual<=0]=np.nan
    threshold=df.two_count_def.to_numpy()
    native_valid=np.isfinite(actual)&np.isfinite(threshold)&(threshold>0)&(actual>threshold)
    native_invalid=np.isfinite(actual)&~native_valid
    df['valid']=np.isfinite(df.observed_def)&np.isfinite(threshold)&(threshold>0)&(df.observed_def.to_numpy()>threshold)
    ax.plot(en,threshold,'-',color='#d17a24',lw=1.8,label='2 counts')
    ax.scatter(en[native_valid],actual[native_valid],marker='o',color='#2166ac',s=32,label='Valid',zorder=3)
    ax.scatter(en[native_invalid],actual[native_invalid],marker='x',color='#888888',s=36,label='Invalid',zorder=3)
    ax.set(xscale='log',yscale='log',xlim=(.3,600),ylim=(1e3,1e6),xlabel=r'O$_2^+$ energy (eV)',ylabel=r'DEF (eV cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$)')
    ax.tick_params(which='both',direction='in',top=True,right=True);ax.grid(which='major',alpha=.16);ax.legend(loc='upper right',fontsize=14,labelspacing=.25,borderpad=.3,handlelength=1.5,handletextpad=.5)
    ax.text(.02,.98,'d',transform=ax.transAxes,va='top',fontweight='bold',fontsize=15)
    ax.set_title('2018-08-05 10:49:58.66 UTC',fontsize=12,pad=10)
    stem=OUT/'v12_count2_inputs'
    for ext in ['png','pdf']:fig.savefig(stem.with_suffix('.'+ext),dpi=220,facecolor='white')
    plt.close(fig)
    # Native calibration data are supplied in examples/data.
    report.update(selected_cache_index_zero_based=ii,selected_sample_index_one_based=ii+1,
        native_validity_rule='Valid if finite observed DEF > simultaneous two-count DEF response; same rule as panel b, evaluated on native energies in panel d',
        native_valid_points=int(native_valid.sum()),native_invalid_positive_points=int(native_invalid.sum()),
        log_plot_note='Zero and negative observed DEF omitted from logarithmic scatter; CSV retains zero values and validity flags')
    (OUT/'comparison_method.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print(stem.with_suffix('.png'))

if __name__=='__main__':main()
