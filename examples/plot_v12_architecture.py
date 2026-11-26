"""Current model with the supplied compact U-shaped architecture layout."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
from matplotlib.colors import LogNorm, ListedColormap
R=Path(__file__).resolve().parents[1]
O=R/'docs';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'font.size':11})
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

f=plt.figure(figsize=(15,10),facecolor='white')
a=f.add_axes([0,0,1,1]);a.set(xlim=(0,975),ylim=(650,0));a.axis('off')
def text(x,y,s,size=12,**kw):
 a.text(x,y,s,ha='center',va='center',fontsize=size,**kw)
def arrow(p,q,c='black',dash=False):
 a.add_patch(FancyArrowPatch(p,q,arrowstyle='->',mutation_scale=14,lw=1.5,color=c,linestyle=(0,(5,4)) if dash else '-'))
def frame(x,y,w,h):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=40',fill=False,edgecolor='gray',lw=1.5,linestyle=(0,(4,3))))
frame(4,6,414,230);frame(530,8,440,235)
text(190,19,'Input',20,fontweight='bold');text(277,20,'(512×32×3)',12)
text(724,21,'Output',20,fontweight='bold');text(809,22,'(512×32×3)',12)
for side in range(2):
 for j in range(3):
  left=25 if side==0 else 560;top=36+j*64 if side==0 else 39+j*66
  width=376 if side==0 else 352;height=59
  ax=f.add_axes([left/975,(650-top-height)/650,width/975,height/650])
  z=([flux[:512],valid[:512],np.broadcast_to(en,flux[:512].shape)][j] if side==0 else prob[j])
  if side==0 and j==1:kw=dict(cmap=ListedColormap(['#e0e0e0','#2166ac']),vmin=0,vmax=1)
  elif side==0:
   cm=plt.colormaps['turbo'].copy();cm.set_bad('black');cm.set_under('black')
   kw=dict(cmap=cm,norm=LogNorm(1e4,1e8) if j==0 else LogNorm(.2,3e4))
  else:kw=dict(cmap='turbo',vmin=0,vmax=1)
  im=ax.pcolormesh(np.arange(512),en,z.T,shading='nearest',rasterized=True,**kw)
  ax.set(yscale='log',ylim=(en.min(),en.max()));ax.axis('off')
  label=([r'O$_2^+$ DEF',r'O$_2^+$ validity','Energy'][j] if side==0 else ['Noise probability','Beam probability','Low-energy ions probability'][j])
  ax.text(.02,.95,label,transform=ax.transAxes,va='top',fontsize=10,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1))
  if side==1:last=im
cb=f.colorbar(last,cax=f.add_axes([922/975,(650-217)/650,8/975,163/650]));cb.set_label('Class probability',fontsize=9);cb.ax.tick_params(labelsize=8,length=2);cb.set_ticks(np.linspace(0,1,6))
conv='#b719a5';pool='#165aa5';up='#18712a';cat='red'
def cube(x,y,green=False):
 colors=['#a8ce96','#cde5bc','#7aa46a'] if green else ['#c3b59b','#e1d7c4','#998b74']
 a.add_patch(Rectangle((x,y+12),14,28,facecolor=colors[0],edgecolor='#465661',lw=.6,zorder=4))
 a.add_patch(Polygon([(x,y+12),(x+12,y),(x+26,y),(x+14,y+12)],facecolor=colors[1],edgecolor='#465661',lw=.6,zorder=4))
 a.add_patch(Polygon([(x+14,y+12),(x+26,y),(x+26,y+28),(x+14,y+40)],facecolor=colors[2],edgecolor='#465661',lw=.6,zorder=4))
def dims(x,y,s):text(x+13,y-9,s,10)
# Encoder and bottleneck placements follow the reference image.
cube(130,283);dims(130,283,'512×32×12')
cube(130,383);dims(130,383,'256×16×12');cube(187,383);dims(187,383,'256×16×24')
cube(187,479);dims(187,479,'128×8×24');cube(247,479);dims(247,479,'128×8×48')
cube(250,577);dims(250,577,'64×4×48');cube(314,577);dims(314,577,'64×4×96')
arrow((143,236),(143,279),conv);arrow((143,327),(143,378),pool)
arrow((158,402),(182,402),conv);arrow((200,426),(200,474),pool)
arrow((216,498),(242,498),conv);arrow((260,522),(260,572),pool)
arrow((279,596),(309,596),conv)
text(60,340,'Encoder 1',18);text(77,443,'Encoder 2',18);text(104,541,'Encoder 3',18);text(327,638,'Bottleneck',18)
# Three decoder stages, numbered in computation order as in the reference.
for x,y,t,e,c,num in [(372,479,128,8,48,1),(491,383,256,16,24,2),(594,283,512,32,12,3)]:
 cube(x,y);cube(x+13,y,True);dims(x+6,y,f'{t}×{e}×{2*c}')
 cube(x+76,y,True);dims(x+76,y,f'{t}×{e}×{c}')
 if num<3:
  cube(x+132,y,True);dims(x+132,y,f'{t*2}×{e*2}×{c//2}')
 arrow((x+42,y+20),(x+71,y+20),conv)
 if num<3:arrow((x+105,y+20),(x+127,y+20),up)
 text(x+129,y+66,f'Decoder {num}',18)
arrow((343,596),(375,596),up);cube(384,577,True);dims(384,577,'128×8×48');arrow((401,575),(401,522),cat)
arrow((518,478),(518,426),cat);arrow((624,380),(624,326),cat)
for start,end in [((182,303),(574,303)),((238,401),(472,401)),((286,497),(366,497))]:arrow(start,end,'gray',True)
text(355,284,'Skip Connection',14,color='gray')
arrow((690,283),(690,241),'black')
# Original operation legend and convolution inset, expanded to include normalization.
frame(781,291,188,139)
for j,(label,color) in enumerate([('Conv Block',conv),('Max Pool',pool),('Transpose Conv',up),('Concatenate',cat),('1×1 Conv','black')]):
 yy=312+j*25;arrow((803,yy),(833,yy),color);text(895,yy,label,13,color=color)
frame(781,458,188,155);text(875,476,'Conv Block',14,color=conv)
for col in range(2):
 xx=789+col*88
 for row,label in enumerate(['Conv 3×3','GroupNorm','ReLU']):
  yy=495+row*36;a.add_patch(Rectangle((xx,yy),77,25,facecolor='#ddd',edgecolor='none'));text(xx+38,yy+12,label,10)
  if row<2:arrow((xx+38,yy+25),(xx+38,yy+35),'gray')
arrow((866,579),(879,509),'gray')
text(875,604,'4 groups per GroupNorm',8,color='gray')
for ext in ['png','pdf']:f.savefig(O/f'unet_v12_count2_architecture.{ext}',dpi=220,facecolor='white')
plt.close(f)
