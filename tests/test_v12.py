from pathlib import Path
import unittest
import numpy as np
import torch
from unet_mars_o2_beams import build_input_channels, load_pretrained_model, predict_spectrogram

ROOT = Path(__file__).resolve().parents[1]

class CountMaskTests(unittest.TestCase):
    def test_count_boundaries(self):
        energy = np.geomspace(.2, 3e4, 32)
        flux = np.full((8, 32), 1e5, dtype=np.float32)
        threshold = np.full_like(flux, 2e4)
        flux[0, :6] = [5000, 20000, 20001, np.nan, 1e5, 1e5]
        threshold[0, :6] = [4000, 20000, 20000, 20000, np.nan, 0]
        x = build_input_channels(flux, energy, two_count_def=threshold)
        np.testing.assert_array_equal(x[1, 0, :6], [1, 0, 1, 0, 0, 0])
        self.assertEqual(x[0, 0, 0], 0)
        np.testing.assert_array_equal(x, build_input_channels(flux, energy, o2_validity=x[1]))
        with self.assertRaises(ValueError):
            build_input_channels(flux, energy)
        with self.assertRaises(ValueError):
            build_input_channels(flux, energy, two_count_def=threshold[:1])
        with self.assertRaises(ValueError):
            build_input_channels(flux, energy, o2_validity=np.full_like(flux, .5))
        model, metadata, device = load_pretrained_model(ROOT / 'weights/unet_v12_count2_best_validation.pt', 'cpu')
        self.assertEqual(metadata['version'], 'v12_count2')
        torch.set_num_threads(2)
        probabilities, labels = predict_spectrogram(model, x, device)
        self.assertEqual(labels.shape, flux.shape)
        np.testing.assert_allclose(probabilities.sum(0), 1, atol=1e-6)

    def test_inference_stride(self):
        import inspect
        from unet_mars_o2_beams.inference import _window_starts
        self.assertEqual(inspect.signature(predict_spectrogram).parameters['stride'].default, 51)
        self.assertEqual(_window_starts(700, 512, 51), [0, 51, 102, 153, 188])
        metadata = __import__('json').loads((ROOT / 'weights/model_metadata.json').read_text())
        self.assertEqual(metadata['inference_stride'], 51)
        self.assertEqual(metadata['training_patch_stride'], 256)
