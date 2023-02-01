from typing import Dict, Optional

import torch as tr
import torchaudio
from torch import Tensor as T

from torchsynth.config import SynthConfig
from torchsynth.module import VCO
from torchsynth.signal_ import Signal


class Wavetable(VCO):
    def __init__(self,
                 synthconfig: SynthConfig,
                 device: Optional[tr.device] = None,
                 wt_frames: Optional[T] = None,
                 **kwargs: Dict[str, T]) -> None:
        super().__init__(synthconfig, device, **kwargs)
        self.set_parameter("tuning", tr.zeros((self.batch_size,)))
        self.set_parameter("mod_depth", tr.full((self.batch_size,), 12.0))

        if wt_frames is None:
            # Create sine to square to noise wavetable
            n_wt_samples = 2048
            frame_0 = tr.sin(tr.linspace(0.0, 2 * tr.pi, n_wt_samples))
            frame_1 = tr.ones((n_wt_samples,), dtype=tr.float)
            frame_1[1024:] = -1.0
            frame_2 = tr.rand_like(frame_1)
            wt_frames = tr.stack([frame_0, frame_1, frame_2], dim=0)

        assert wt_frames is not None
        assert wt_frames.ndim == 2
        assert wt_frames.size(0) <= 256, "More than 256 frames is not supported"

        self.n_frames = wt_frames.size(0)
        self.n_wt_samples = wt_frames.size(1)
        self.wt = wt_frames

    def output(self, midi_f0: T, pitch_mod: Optional[Signal] = None, pos_mod: Optional[Signal] = None) -> Signal:
        audio_frames = super().output(midi_f0, pitch_mod)  # TODO(cm): override super method
        if pos_mod is None or self.n_frames == 1:
            return audio_frames[:, 0, :].as_subclass(Signal)

        # TODO(cm): make this more understandable / less memory intensive
        pos_mod = tr.clip(pos_mod, self.eps, 1.0 - self.eps)
        pos_mod *= (self.n_frames - 1)
        pos_mod = pos_mod.unsqueeze(1).repeat(1, self.n_frames - 1, 1)
        frame_indices_lo = tr.arange(1, self.n_frames).view(1, -1, 1).expand(pos_mod.size(0), -1, pos_mod.size(2))
        frame_indices_hi = tr.arange(0, self.n_frames - 1).view(1, -1, 1).expand(pos_mod.size(0), -1, pos_mod.size(2))
        amount_lo = frame_indices_lo - pos_mod
        amount_lo[amount_lo >= 1.0] = 0.0
        amount_lo[amount_lo < 0.0] = 0.0
        amount_lo = tr.cat([amount_lo, tr.zeros((amount_lo.size(0), 1, amount_lo.size(2)))], dim=1)
        amount_hi = pos_mod - frame_indices_hi
        amount_hi[amount_hi > 1.0] = 0.0
        amount_hi[amount_hi < 0.0] = 0.0
        amount_hi = tr.cat([tr.zeros((amount_hi.size(0), 1, amount_hi.size(2))), amount_hi], dim=1)

        amount = amount_hi + amount_lo
        assert tr.allclose(tr.sum(amount, dim=1), tr.tensor(1.0))
        audio = tr.sum(amount * audio_frames, dim=1)
        return audio.as_subclass(Signal)

    def oscillator(self, argument: Signal, midi_f0: T) -> T:
        argument %= 2 * tr.pi
        indices = (argument / (2 * tr.pi)) * self.n_wt_samples
        indices = tr.round(indices)
        indices %= self.n_wt_samples
        indices = indices.long()
        indices = indices.unsqueeze(1).expand(-1, self.n_frames, -1)
        wt = self.wt.unsqueeze(0).expand(indices.size(0), -1, -1)
        return tr.gather(wt, -1, indices)


if __name__ == "__main__":
    bs = 3
    sr = 44100
    buffer_size_seconds = 3

    sc = SynthConfig(batch_size=3,
                     sample_rate=sr,
                     buffer_size_seconds=buffer_size_seconds,
                     reproducible=False)
    wt = Wavetable(sc)
    pitch = tr.full((sc.batch_size,), 48)
    mod_sig = tr.linspace(0.0, 1.0, buffer_size_seconds * sr).view(1, -1).expand(3, -1)
    pos_sig = tr.sin(tr.linspace(0.0, 2 * tr.pi, buffer_size_seconds * sr)).view(1, -1).expand(3, -1)
    pos_sig = (pos_sig + 1.0) / 2.0
    audio = wt.output(pitch, mod_sig, pos_sig)

    curr = audio[0, :].unsqueeze(0)
    torchaudio.save("../out/tmp.wav", curr, sr)
