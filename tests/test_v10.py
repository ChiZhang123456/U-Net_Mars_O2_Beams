"""Regression checks for the public v10 inference contract."""
from pathlib import Path
import unittest
import numpy as np
import torch
from unet_mars_o2_beams import build_input_channels, load_pretrained_model, predict_spectrogram

ROOT = Path(__file__).resolve().parents[1]

class V10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model, cls.metadata, cls.device = load_pretrained_model(ROOT / "weights/unet_v10_best_validation.pt", "cpu")

    def spectra(self, n=19):
        return dict(o_def=np.full((n,32),1e6,np.float32), o2_def=np.full((n,32),1e5,np.float32),energy_ev=np.geomspace(.2,3e4,32))

    def test_channel_order_and_scaling(self):
        x=build_input_channels(**self.spectra())
        self.assertEqual(x.shape,(6,19,32))
        np.testing.assert_allclose(x[:3,0,0],[.25,.5,2/3],atol=1e-7)
        np.testing.assert_array_equal(x[3:5],1)
        np.testing.assert_allclose(x[5,0,[0,-1]],[0,1],atol=1e-7)

    def test_invalid_pixels_and_mask(self):
        d=self.spectra();d['o_def'][0,0]=np.nan;d['o2_def'][0,1]=-1
        x=build_input_channels(**d)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x[4,0,0],0);self.assertEqual(x[3,0,1],0)
        np.testing.assert_array_equal(x[2,0,:2],0)
        with self.assertRaises(ValueError):build_input_channels(**d,o_validity=np.full((19,32),np.nan))

    def test_checkpoint_shape_and_short_input(self):
        p,y=predict_spectrogram(self.model,build_input_channels(**self.spectra()),self.device)
        self.assertEqual(p.shape,(3,19,32));self.assertEqual(y.shape,(19,32))
        np.testing.assert_allclose(p.sum(0),1,atol=1e-6)
        self.assertEqual(sum(v.numel() for v in self.model.parameters()),272787)
        self.assertEqual(self.metadata['class_names'],['Noise','Beam','Low-energy ions'])

    def test_overlap_and_full_day_regression(self):
        with np.load(ROOT/'examples/data/static_20180120_v10.npz') as s:
            x=build_input_channels(o_def=s['o_def'],o2_def=s['o2_def'],energy_ev=s['energy_ev'])
            p,y=predict_spectrogram(self.model,x,self.device)
            np.testing.assert_array_equal(y,s['labels'])
            np.testing.assert_allclose(p.sum(0),1,atol=1e-6)

    def test_invalid_inference_geometry(self):
        x=build_input_channels(**self.spectra())
        for kw in [dict(stride=513),dict(window_size=511)]:
            with self.assertRaises(ValueError):predict_spectrogram(self.model,x,self.device,**kw)
        with self.assertRaises(ValueError):predict_spectrogram(self.model,x[:5],self.device)
        x[0,0,0]=np.nan
        with self.assertRaises(ValueError):predict_spectrogram(self.model,x,self.device)

    def test_v9_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            load_pretrained_model(ROOT/'weights/unet_v9_best_selection.pt','cpu')

if __name__=='__main__':unittest.main()
