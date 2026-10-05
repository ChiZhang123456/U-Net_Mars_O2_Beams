from pathlib import Path
import unittest
import numpy as np
import torch
from unet_mars_o2_beams import build_input_channels,load_pretrained_model,predict_spectrogram
ROOT=Path(__file__).resolve().parents[1]
class V11Tests(unittest.TestCase):
 def test_preprocessing_and_regression(self):
  torch.set_num_threads(2)
  e=np.geomspace(.2,3e4,32);f=np.full((19,32),1e5,dtype=np.float32);f[0,0]=np.nan
  x=build_input_channels(o2_def=f,energy_ev=e)
  self.assertEqual(x.shape,(3,19,32));self.assertEqual(x[1,0,0],0)
  np.testing.assert_allclose(x[0,1],.25);np.testing.assert_allclose(x[2,0,[0,-1]],[0,1],atol=1e-7)
  with self.assertRaises(ValueError):build_input_channels(f,e,o2_validity=np.full_like(f,np.nan))
  model,meta,device=load_pretrained_model(ROOT/'weights/unet_v11_best_validation.pt','cpu')
  self.assertEqual(sum(p.numel() for p in model.parameters()),272463)
  with np.load(ROOT/'examples/data/static_20180120_v11.npz') as d:
   x=build_input_channels(o2_def=d['o2_def'],energy_ev=d['energy_ev'])
   p,y=predict_spectrogram(model,x,device)
   np.testing.assert_array_equal(y,d['labels']);np.testing.assert_allclose(p.sum(0),1,atol=1e-6)
  with self.assertRaises(ValueError):predict_spectrogram(model,x[:2],device)
 def test_interval_table(self):
  x=np.loadtxt(ROOT/'catalogs/beam_points_v11_intervals.txt',dtype=str)
  self.assertEqual(x.shape,(14484,3));np.testing.assert_array_equal(x[:,0].astype(int),np.arange(1,14485))
  a=x[:,1].astype('datetime64[ns]');b=x[:,2].astype('datetime64[ns]')
  self.assertTrue(np.all(b>=a));self.assertTrue(np.all(a[1:]-b[:-1]>np.timedelta64(10,'m')))
if __name__=='__main__':unittest.main()
