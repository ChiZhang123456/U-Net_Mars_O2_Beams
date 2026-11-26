from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Polygon,Rectangle
from matplotlib.colors import LogNorm,ListedColormap
R=Path(__file__).resolve().parents[1]
O=R/'docs';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'font.size':11})
f=plt.figure(figsize=(20,12.5),facecolor='white');a=f.add_axes([0,0,1,1]);a.set(xlim=(0,16),ylim=(0,10));a.axis('off')
def text(x,y,s,size=11,**kw):a.text(x,y,s,ha='center',va='center',fontsize=size,**kw)
def box(x,y,w,h,title,sub,color):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.12',facecolor=color,edgecolor='#80909c',lw=1))
 text(x+w/2,y+h*.65,title,12,fontweight='bold');text(x+w/2,y+h*.25,sub,11)
def arrow(p,q,color='#536778',dash=False):a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=15,lw=1.7,color=color,linestyle='--' if dash else '-'))
text(8,9.68,'U-Net v12 count2 for O$_2^+$ ion classification',22,fontweight='bold',color='#243747')
text(8,9.36,'272,463 parameters | 512 × 32 windows | inference stride 51',11,color='#536778')
import sys, torch, json, hashlib
sys.path.insert(0,str(R))
from unet_mars_o2_beams import build_input_channels,load_pretrained_model,predict_spectrogram
from unet_mars_o2_beams.inference import _window_starts
with np.load(R/'examples/data/static_20180805_count2.npz') as d:
 en=d['energy_ev'];flux=d['o2_def'];threshold=d['two_count_def']
x=build_input_channels(flux,en,two_count_def=threshold);valid=x[1]
checkpoint=R/'weights/unet_v12_count2_best_validation.pt'
model,metadata,device=load_pretrained_model(checkpoint,'cpu');torch.set_num_threads(2)
with torch.no_grad():prob=torch.softmax(model(torch.from_numpy(x[:,:512])[None])[0],dim=0).numpy()
averaged,labels=predict_spectrogram(model,x,device)
np.testing.assert_allclose(prob.sum(0),1,atol=1e-6)
np.savez_compressed(O/'v12_count2_figure_outputs.npz',single_window_probabilities=prob,probabilities=averaged,labels=labels)
(O/'v12_count2_figure_method.json').write_text(json.dumps(dict(model_version=metadata['version'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),window=512,stride=51,window_starts_one_based=[v+1 for v in _window_starts(len(flux),512,51)],architecture_probability_panel='Direct softmax of first 512-sample window',full_interval_output='Mean overlapping window probabilities before argmax',input_display='Physical DEF and energy; normalized internally',validity='Finite corrected DEF strictly greater than finite positive two-count response'),indent=2))
# Display one actual 512-sample model window rather than the complete 600-sample interval.
for side in range(2):
 x0=.65 if side==0 else 9.2
 text(x0+2.9,9.12,('Input' if side==0 else 'Class probabilities')+'   512 × 32 × 3',15,fontweight='bold')
 for j in range(3):
  ax=f.add_axes([x0/16,(8.02-j*.58)/10,5.7/16,.53/10])
  z=([flux[:512],valid[:512],np.broadcast_to(en,flux[:512].shape)][j] if side==0 else prob[j,:512])
  kw=({'cmap':ListedColormap(['#e0e0e0','#2166ac']),'vmin':0,'vmax':1} if side==0 and j==1 else {'cmap':'turbo','norm':LogNorm(1e4,1e8) if j==0 else LogNorm(.2,3e4)} if side==0 else {'cmap':'turbo','vmin':0,'vmax':1})
  ax.set_facecolor('black') if side==0 and j==0 else None
  if side==0 and j==0:
   cm=plt.colormaps['turbo'].copy();cm.set_bad('black');cm.set_under('black');kw['cmap']=cm
  im=ax.pcolormesh(np.arange(512),en,z.T,shading='nearest',rasterized=True,**kw);ax.set_yscale('log');ax.set_ylim(en.min(),en.max());ax.axis('off')
  ax.text(.012,.88,(['O$_2^+$ DEF','O$_2^+$ validity (DEF > 2-count response)','Energy'][j] if side==0 else ['Noise','Beam','Low-energy ions'][j]),transform=ax.transAxes,va='top',fontsize=10,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.92,pad=1.5))
  if side==1 and j==2:last=im
cb=f.colorbar(last,cax=f.add_axes([.946,.686,.009,.169]));cb.set_label('Class probability',fontsize=10);cb.set_ticks([0,.5,1]);cb.ax.tick_params(labelsize=9)

def cube(x,y,shape,green=False,label=True):
 c=['#add29f','#d4e9c9','#7fa875'] if green else ['#c4b7a0','#e0d6c3','#9e907b']
 a.add_patch(Rectangle((x,y),.25,.42,facecolor=c[0],edgecolor='#354752',lw=.7,zorder=4))
 a.add_patch(Polygon([(x,y+.42),(x+.18,y+.57),(x+.43,y+.57),(x+.25,y+.42)],facecolor=c[1],edgecolor='#354752',lw=.7,zorder=4))
 a.add_patch(Polygon([(x+.25,y),(x+.43,y+.15),(x+.43,y+.57),(x+.25,y+.42)],facecolor=c[2],edgecolor='#354752',lw=.7,zorder=4))
 if label:text(x+.22,y+.75,shape,8)
def op(x,z,y,c,label):
 arrow((x+.48,y+.22),(z-.05,y+.22),c);text((x+z+.4)/2,y+(.43 if label=='Concatenate' else -.18),label,8,color=c)
ys=[5.58,4.13,2.68];conv='#a04396';pool='#327bb0';up='#398654';cat='#c75a3b'
for i,y in enumerate(ys):
 t=512//2**i;e=32//2**i;c=12*2**i;x=1+i*1.1;z=x+1.1
 cube(x,y,f'{t} × {e} × {3 if i==0 else c//2}');cube(z,y,f'{t} × {e} × {c}');op(x,z,y,conv,'Conv block')
 text(x+.75,y-.5,f'Encoder {i+1}',12,fontweight='bold')
 if i<2:
  arrow((z+.22,y-.03),(z+.22,ys[i+1]+.6),pool);text(z+.72,y-.57,'Max pool',8,color=pool)
 cube(10,y,f'{t} × {e} × {c}',True)
 cube(11.6,y,'',False,False);cube(11.84,y,'',True,False);text(11.95,y+.75,f'{t} × {e} × {2*c}',8)
 cube(13.35,y,f'{t} × {e} × {c}',True)
 op(10,11.6,y,cat,'Concatenate');op(11.84,13.35,y,conv,'Conv block')
 text(14.65,y+.22,f'Decoder {i+1}',12,fontweight='bold')
 sy=y+1.02
 a.plot([z+.22,z+.22,11.8],[y+.6,sy,sy],color='#92999d',ls=(0,(5,4)),lw=1.3)
 arrow((11.8,sy),(11.8,y+.6),'#92999d',True);text(7.1,sy+.13,'Skip connection',8,color='#737e85')
 if i<2:
  arrow((13.57,ys[i+1]+.62),(10.22,y-.03),up)
cube(4.3,1.15,'64 × 4 × 48');cube(5.5,1.15,'64 × 4 × 96');op(4.3,5.5,1.15,conv,'Conv block')
arrow((4.52,2.63),(4.52,1.77),pool);text(5.05,2.05,'Max pool',8,color=pool)
text(5,.66,'Bottleneck',12,fontweight='bold');arrow((5.98,1.37),(10.22,2.63),up);text(8.15,1.7,'Transpose conv',8,color=up)
arrow((1.22,6.8),(1.22,6.21));arrow((13.57,6.21),(12.2,6.8));text(14.1,6.55,'1 × 1 Conv → Softmax',9)
text(4.75,.22,'Conv block: [Conv 3 × 3 → GroupNorm (4 groups) → ReLU] × 2',9)
text(12.4,1.0,'Max pool: 2 × 2, stride 2',9,color=pool)
text(12.4,.61,'Transpose conv: 2 × 2, stride 2',9,color=up)
text(12.4,.22,'Dimensions: time × energy × channels',9,color='#536778')
for ext in ['png','pdf']:f.savefig(O/f'unet_v12_count2_architecture.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
plt.close(f)
