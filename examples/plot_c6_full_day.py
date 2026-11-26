"""Run the current model on the supplied corrected C6 day and plot H+, O2+, classes."""
from pathlib import Path
import sys, csv, json, hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm,ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from unet_mars_o2_beams import build_input_channels,load_pretrained_model,predict_spectrogram

def main():
    path=R/'examples/data/static_c6_20180120_corrected.npz'
    with np.load(path) as d:
        time=d['time'];energy=d['energy_ev'];h=d['h_def'];o2=d['o2_def'];threshold=d['two_count_def']
    inputs=build_input_channels(o2,energy,two_count_def=threshold)
    checkpoint=R/'weights/unet_v12_count2_best_validation.pt'
    model,metadata,device=load_pretrained_model(checkpoint,'cpu');torch.set_num_threads(4)
    probabilities,raw=predict_spectrogram(model,inputs,device)
    # Display the count-mask convention explicitly; preserve unconstrained predictions separately.
    labels=np.where(inputs[1].astype(bool),raw,0).astype(np.uint8)
    out=R/'docs';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'static_c6_20180120_predictions.npz',time=time,energy_ev=energy,
        probabilities=probabilities,raw_labels=raw,labels=labels,validity=inputs[1].astype(bool))
    method=dict(model=metadata['version'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        inference_stride=51,window=512,classes=['Noise','Beam','Low-energy ions'],
        source='STATIC C6 L2 and same-day iv4, 2018-01-20',
        display_rule='Invalid count-mask pixels displayed as Noise; raw_labels preserved separately',
        counts=np.bincount(labels.ravel(),minlength=3).tolist())
    (out/'static_c6_20180120_method.json').write_text(json.dumps(method,indent=2))
    plt.rcParams.update({'font.family':'Arial','mathtext.fontset':'dejavusans','pdf.fonttype':42,'font.size':11})
    fig,axes=plt.subplots(3,1,figsize=(11.5,7.4),sharex=True)
    fig.subplots_adjust(left=.13,right=.82,top=.91,bottom=.115,hspace=.045)
    cm=plt.colormaps['turbo'].copy();cm.set_bad('black');cm.set_under('black')
    for ax,flux,label in zip(axes[:2],[h,o2],[r'STATIC H$^+$',r'STATIC O$_2^+$']):
        mesh=ax.pcolormesh(time,energy,np.ma.masked_invalid(flux.T),shading='nearest',cmap=cm,norm=LogNorm(1e4,1e8),rasterized=True)
        ax.set_ylabel(label+'\n'+r'$E_i/q$ (eV/q)',fontsize=14)
    colors=['#d9d9d9','#7b2cbf','#1b9e77'];names=['Noise','Beam','Low-energy ions']
    axes[2].pcolormesh(time,energy,labels.T,shading='nearest',cmap=ListedColormap(colors),norm=BoundaryNorm([-.5,.5,1.5,2.5],3),rasterized=True)
    axes[2].set_ylabel(r'STATIC O$_2^+$'+'\n'+r'$E_i/q$ (eV/q)',fontsize=14)
    for letter,ax in zip('abc',axes):
        ax.set(yscale='log',ylim=(.3,3e4),xlim=(np.datetime64('2018-01-20'),np.datetime64('2018-01-21')))
        ax.set_yticks([1,10,100,1000,10000]);ax.tick_params(which='both',top=True,right=True,direction='in')
        ax.text(-.16,1,letter,transform=ax.transAxes,fontweight='bold',fontsize=16,va='top')
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2));axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].set_xlabel('UTC on 20 January 2018',fontsize=14)
    pos=axes[0].get_position();bottom=axes[1].get_position().y0
    cb=fig.colorbar(mesh,cax=fig.add_axes([pos.x1+.006,bottom,.015,pos.y1-bottom]));cb.set_label(r'DEF (keV cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$)')
    regions=list(csv.DictReader((R/'examples/data/regions_20180120.csv').open(encoding='utf-8-sig')))
    region_ax=fig.add_axes([pos.x0,pos.y1+.006,pos.width,.045],sharex=axes[0]);region_ax.set_ylim(0,1.4)
    region_colors={'SW':'#d8645a','MSH':'#4c78a8','MSP':'#5f6368'}
    for row in regions:region_ax.axvspan(np.datetime64(row['start_utc']),np.datetime64(row['end_utc']),ymin=0,ymax=.43,color=region_colors[row['region']],linewidth=0)
    for key,title in [('SW','Solar Wind'),('MSH','Magnetosheath'),('MSP','Induced Magnetosphere')]:
        chosen=max((row for row in regions if row['region']==key),key=lambda row:np.datetime64(row['end_utc'])-np.datetime64(row['start_utc']))
        left=np.datetime64(chosen['start_utc']);right=np.datetime64(chosen['end_utc'])
        region_ax.text(left+(right-left)/2,1,title,ha='center',va='center',color=region_colors[key],fontsize=10,fontweight='bold')
    region_ax.axis('off')
    fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(colors,names)],loc='center left',bbox_to_anchor=(.835,.24),frameon=False,fontsize=12)
    for ext in ['png','pdf']:fig.savefig(out/f'static_c6_20180120_example.{ext}',dpi=300)
    plt.close(fig);print(method,flush=True)
if __name__=='__main__':main()
