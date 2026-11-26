"""Export corrected STATIC C6 spectra and counts from L2 and iv4 CDFs.

Requires spacepy only for this optional raw-data export step.
"""
import argparse
from pathlib import Path
import numpy as np

def export(l2, iv4, output):
    from spacepy import pycdf
    with pycdf.CDF(str(l2)) as c, pycdf.CDF(str(iv4)) as b:
        time=np.asarray(c['epoch'][...],dtype='datetime64[ns]')
        order=np.argsort(time,kind='stable');time=time[order]
        raw=np.asarray(c['data'][...],dtype=np.float32)[order]
        dead=np.asarray(c['dead'][...],dtype=np.float32)[order]
        sweep=np.asarray(c['swp_ind'][...],int)[order]
        energy_table=np.asarray(c['energy'][...],dtype=np.float32)
        mass=np.mean(np.nanmean(c['mass_arr'][...],axis=1),axis=1)
        center=int(np.argmin(abs(mass-32)))
        bt=np.asarray(b['epoch'][...],dtype='datetime64[ns]');bo=np.argsort(bt,kind='stable');bt=bt[bo]
        right=np.searchsorted(bt,time).clip(1,len(bt)-1);left=right-1
        weight=np.asarray((time-bt[left])/(bt[right]-bt[left]),float)
        ai=np.asarray(b['att_ind'][...],int)[bo];si=np.asarray(b['swp_ind'][...],int)[bo];ei=np.asarray(b['eff_ind'][...],int)[bo]
        gf=np.asarray(b['gf'][...]);eff=np.asarray(b['eff'][...]);geom=float(b['geom_factor'][...])
        dt=np.asarray(b['time_integ'][...])[bo];integ=dt[left]*(1-weight)+dt[right]*weight
        background=np.asarray(b['bkg'][...],dtype=np.float32)[bo]
        net=np.empty_like(raw);corrected=np.empty_like(raw);two_native=None
        for mi in range(raw.shape[1]):
            effective=geom*gf[ai,:,si]*eff[mi][:,ei].T
            effective=effective[left]*(1-weight[:,None])+effective[right]*weight[:,None]
            bg=background[left,mi,:]*(1-weight[:,None])+background[right,mi,:]*weight[:,None]
            net[:,mi,:]=raw[:,mi,:]-bg
            flux=net[:,mi,:]*dead[:,mi,:]/(integ[:,None]*effective)
            corrected[:,mi,:]=np.where(np.isfinite(flux)&(flux>0),flux,0)
            if mi==center:two_native=2*dead[:,mi,:]/(integ[:,None]*effective)
    native_energy=energy_table[center][:,sweep].T
    energy=np.geomspace(.2,3e4,32).astype(np.float32)
    species={}
    for name,lo,hi in [('h',0,1.55),('o2',24,40)]:
        use=(mass>=lo)&(mass<=hi)
        native_flux=corrected[:,use,:].sum(1)
        grid=np.full((len(time),32),np.nan,np.float32)
        for row in range(len(time)):
            ix=np.argsort(native_energy[row]);grid[row]=np.interp(np.log10(energy),np.log10(native_energy[row,ix]),native_flux[row,ix],left=np.nan,right=np.nan)
        species[name+'_def']=grid
    threshold=np.full_like(species['o2_def'],np.nan)
    for row in range(len(time)):
        ix=np.argsort(native_energy[row]);threshold[row]=np.interp(np.log10(energy),np.log10(native_energy[row,ix]),two_native[row,ix],left=np.nan,right=np.nan)
    np.savez_compressed(output,time=time,energy_ev=energy,two_count_def=threshold,**species,
        c6_raw_counts=raw,c6_net_counts=net,c6_def_corrected=corrected,
        c6_mass_amu=mass.astype(np.float32),c6_energy_table_ev=energy_table,c6_sweep_index=sweep.astype(np.int16),
        c6_deadtime_factor=dead,central_o2_mass_index=np.int16(center))
    print(output,Path(output).stat().st_size,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--l2',type=Path,required=True);parser.add_argument('--iv4',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'data/static_c6_20180120_corrected.npz')
    args=parser.parse_args();export(args.l2,args.iv4,args.output)
