"""
Synth modules.

    TODO    :   - ADSR needs fixing. sustain duration should actually decay.
                - Convert operations to tensors, obvs.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import resample

from ddspdrum.defaults import CONTROL_RATE, SAMPLE_RATE
from ddspdrum.parameter import Parameter
from ddspdrum.util import crossfade, fix_length, midi_to_hz


class SynthModule:
    """
    Base class for synthesis modules.

    WARNING: For now, SynthModules should be atomic and not contain other SynthModules.
    """

    def __init__(
        self, sample_rate: int = SAMPLE_RATE, control_rate: int = CONTROL_RATE
    ):
        """
        NOTE:
        __init__ should only set parameters.
        We shouldn't be doing computations in __init__ because
        the computations will change when the parameters change.
        """
        self.sample_rate = sample_rate
        self.control_rate = control_rate
        self.parameters: Dict[Parameter] = {}

    def add_parameters(self, parameters: List[Parameter]):
        """
        Add parameters to this SynthModule's parameters dictionary.
        (Since there is inheritance, this might happen several times.)
        """
        for parameter in parameters:
            assert parameter.name not in self.parameters
            self.parameters[parameter.name] = parameter

    def control_to_sample_rate(self, control: np.array) -> np.array:
        """
        Resample a control signal to the sample rate
        TODO: One thing I worry about with all these casts to
        ints back and forth between sample and control rate is
        off-by-one errors.
        I'm beginning to believe we should standardize as much
        as possible on nsamples in most places instead of
        float duration in seconds, and just convert to ncontrol
        when needed..
        """
        # Right now it appears that all signals are 1d, but later
        # we'll probably convert things to 2d: instance x signal

        if self.control_rate == self.sample_rate:
            return control
        else:
            assert control.ndim == 1
            num_samples = int(
                round(len(control) * self.sample_rate / self.control_rate)
            )
            return resample(control, num_samples)

    def connect_parameter(
        self, parameter_id: str, module: SynthModule, module_parameter_id: str
    ):
        """
        Create a named parameter for this synthesizer that is connected to a parameter
        in an underlying synth module

        Parameters
        ----------
        parameter_id (str)          : name of the new parameter
        module (SynthModule)        : the SynthModule to connect to this parameter
        module_parameter_id (str)   : parameter_id in SynthModule to target
        """
        if parameter_id in self.parameters:
            raise ValueError("parameter_id: {} already used".format(parameter_id))

        if module_parameter_id not in module.parameters:
            raise KeyError(
                "parameter_id: {} not a parameter in {}".format(
                    module_parameter_id, module
                )
            )

        self.parameters[parameter_id] = module.get_parameter(module_parameter_id)

    def get_parameter(self, parameter_id: str) -> Parameter:
        """
        Get a single parameter for this module

        Parameters
        ----------
        parameter_id (str)  :   Id of the parameter to return
        """
        return self.parameters[parameter_id]

    def get_parameter_0to1(self, parameter_id: str) -> float:
        """
        Get the value of a single parameter in the range of [0,1]

        Parameters
        ----------
        parameter_id (str)  :   Id of the parameter to return the value for
        """
        return self.parameters[parameter_id].get_value_0to1()

    def set_parameter(self, parameter_id: str, value: float):
        """
        Update a specific parameter value, ensuring that it is within a specified range

        Parameters
        ----------
        parameter_id (str)  : Id of the parameter to update
        value (float)       : Value to update parameter with
        """
        self.parameters[parameter_id].set_value(value)

    def set_parameter_0to1(self, parameter_id: str, value: float):
        """
        Update a specific parameter with a value in the range [0,1]

        Parameters
        ----------
        parameter_id (str)  : Id of the parameter to update
        value (float)       : Value to update parameter with
        """
        self.parameters[parameter_id].set_value_0to1(value)

    def list_parameters(self):
        """
        Return a dictionary of the parameters as strings
        """
        return {key: str(self.parameters[key]) for key in self.parameters}


class ADSR(SynthModule):
    """
    Envelope class for building a control rate ADSR signal
    """

    def __init__(
        self,
        a: float = 0.25,
        d: float = 0.25,
        s: float = 0.5,
        r: float = 0.5,
        alpha: float = 3.0,
        sample_rate: int = SAMPLE_RATE,
        control_rate: int = CONTROL_RATE,
    ):
        """
        Parameters
        ----------
        a                   :   attack time (sec), >= 0
        d                   :   decay time (sec), >= 0
        s                   :   sustain amplitude between 0-1. The only part of
                                ADSR that (confusingly, by convention) is not
                                a time value.
        r                   :   release time (sec), >= 0
        alpha               :   envelope curve, >= 0. 1 is linear, >1 is
                                exponential.
        """
        super().__init__(sample_rate=sample_rate, control_rate=control_rate)
        self.add_parameters(
            [
                Parameter("attack", a, 0, 20, curve="log"),
                Parameter("decay", d, 0, 20, curve="log"),
                Parameter("sustain", s, 0, 1),
                Parameter("release", r, 0, 20, curve="log"),
                Parameter("alpha", alpha, 0, 10),
            ]
        )

    def __call__(self, sustain_duration: float = 0):
        """Generate an envelope that sustains for a given duration in seconds.

        Generates a control-rate envelope signal with given attack, decay and
        release times, sustained for `sustain_duration` in seconds. E.g., an
        envelope with no attack or decay, a sustain duration of 1 and a 0.5
        release will last for 1.5 seconds.

        """

        assert sustain_duration >= 0
        return np.concatenate(
            (
                self.note_on,
                self.sustain(sustain_duration),
                self.note_off,
            )
        )

    def _ramp(self, duration: float):
        """Makes a ramp of a given duration in seconds at control rate.

        This function is used for the piece-wise construction of the envelope
        signal. Its output monotonically increases from 0 to 1. As a result,
        each component of the envelope is a scaled and possibly reversed
        version of this ramp:

        attack      -->     returns an `a`-length ramp, as is.
        decay       -->     `d`-length reverse ramp, descends from 1 to `s`.
        release     -->     `r`-length reverse ramp, descends from `s` to 0.

        Its curve is determined by alpha:

        alpha = 1 --> linear,
        alpha > 1 --> exponential,
        alpha < 1 --> logarithmic.

        """

        t = np.linspace(
            0, duration, int(round(duration * self.control_rate)), endpoint=False
        )
        return (t / duration) ** self.parameters["alpha"].value

    @property
    def attack(self):
        return self._ramp(self.parameters["attack"].value)

    @property
    def decay(self):
        # `d`-length reverse ramp, scaled and shifted to descend from 1 to `s`.
        decay = self.parameters["decay"].value
        sustain = self.parameters["sustain"].value
        return self._ramp(decay)[::-1] * (1 - sustain) + sustain

    def sustain(self, duration):
        return np.full(
            round(int(duration * CONTROL_RATE)),
            fill_value=self.parameters["sustain"].value,
        )

    @property
    def release(self):
        # `r`-length reverse ramp, scaled and shifted to descend from `s` to 0.
        sustain = self.parameters["sustain"].value
        release = self.parameters["release"].value
        return self._ramp(release)[::-1] * sustain

    @property
    def note_on(self):
        return np.append(self.attack, self.decay)

    @property
    def note_off(self):
        return self.release

    def __str__(self):
        return f"""ADSR(a={self.parameters['attack']}, d={self.parameters['decay']},
                s={self.parameters['sustain']}, r={self.parameters['release']},
                alpha={self.get_parameter('alpha')})"""


class VCO(SynthModule):
    """
    Voltage controlled oscillator.

    Think of this as a VCO on a modular synthesizer. It has a base pitch
    (specified here as a midi value), and a pitch modulation depth. Its call
    accepts a control-rate modulation signal between 0 - 1. An array of 0's
    returns a stationary audio signal at its base pitch.


    Parameters
    ----------

    midi_f0 (flt)       :       pitch value in 'midi' (69 = 440Hz).
    mod_depth (flt)     :       depth of the pitch modulation in semitones.

    Examples
    --------

    >>> myVCO = VCO(midi_f0=69, mod_depth=24)
    >>> two_8ve_chirp = myVCO(np.linspace(0, 1, 1000, endpoint=False))
    """

    def __init__(
        self,
        midi_f0: float = 10,
        mod_depth: float = 50,
        phase: float = 0,
        sample_rate: int = SAMPLE_RATE,
        control_rate: int = CONTROL_RATE,
    ):
        super().__init__(sample_rate=sample_rate, control_rate=control_rate)
        self.add_parameters(
            [
                Parameter("pitch", midi_f0, 0, 127),
                Parameter("mod_depth", mod_depth, 0, 127),
            ]
        )
        # TODO: Make this a parameter too?
        self.phase = phase

    def __call__(self, mod_signal: np.array, phase: float = 0) -> np.array:
        """
        Generates audio signal from control-rate mod.

        There are three representations of the 'pitch' at play here: (1) midi,
        (2) instantaneous frequency, and (3) phase, a.k.a. 'argument'.

        (1) midi    This is an abuse of the standard midi convention, where
                    semitone pitches are mapped from 0 - 127. Here it's a
                    convenient way to represent pitch linearly. An A above
                    middle C is midi 69.

        (2) freq    Pitch scales logarithmically in frequency. A is 440Hz.

        (3) phase   This is the argument of the cosine function that generates
                    sound. Frequency is the first derivative of phase; phase is
                    integrated frequency (~ish).

        First we generate the 'pitch contour' of the signal in midi values (mod
        contour + base pitch). Then we convert to a phase argument (via
        frequency), then output sound.

        """

        assert (mod_signal >= 0).all() and (mod_signal <= 1).all()

        modulation = self.parameters["mod_depth"].value * mod_signal
        control_as_midi = self.parameters["pitch"].value + modulation
        control_as_frequency = midi_to_hz(control_as_midi)
        cosine_argument = self.make_argument(control_as_frequency) + phase

        self.phase = cosine_argument[-1]
        return self.oscillator(cosine_argument)

    def make_argument(self, control_as_frequency: np.array):
        """
        Generates the phase argument to feed a cosine function to make audio.
        """

        up_sampled = self.control_to_sample_rate(control_as_frequency)
        return np.cumsum(2 * np.pi * up_sampled / SAMPLE_RATE)

    @abstractmethod
    def oscillator(self, argument):
        """
        Dummy method. Overridden by child class VCO's.
        """
        pass


class SineVCO(VCO):
    """
    Simple VCO that generates a pitched sinudoid.

    Built off the VCO base class, it simply implements a cosine function as oscillator.
    """

    def __init__(self, midi_f0: float = 10, mod_depth: float = 50, phase: float = 0):
        super().__init__(midi_f0=midi_f0, mod_depth=mod_depth, phase=phase)

    def oscillator(self, argument):
        return np.cos(argument)


class SquareSawVCO(VCO):
    """
    VCO that can be either a square or a sawtooth waveshape.
    Tweak with the shape parameter. (0 is square.)

    With apologies to:

    Lazzarini, Victor, and Joseph Timoney. "New perspectives on distortion synthesis for
        virtual analog oscillators." Computer Music Journal 34, no. 1 (2010): 28-40.
    """

    def __init__(
        self,
        shape: float = 0,
        midi_f0: float = 10,
        mod_depth: float = 50,
        phase: float = 0,
    ):
        super().__init__(midi_f0=midi_f0, mod_depth=mod_depth, phase=phase)
        self.add_parameters(
            [
                Parameter("shape", shape, 0, 1),
            ]
        )

    def oscillator(self, argument):
        square = np.tanh(np.pi * self.partials_constant * np.sin(argument) / 2)
        shape = self.parameters['shape'].value
        return (1 - shape / 2) * square * (1 + shape * np.cos(argument))

    @property
    def partials_constant(self):
        """
        Constant value that controls the number of partials in the resulting square /
        saw wave in order to keep aliasing at an acceptable level. Higher frequencies
        require fewer partials whereas lower frequency sounds can safely have more
        partials without causing audible aliasing.
        """
        max_pitch = self.parameters['pitch'].value + self.parameters['mod_depth'].value
        max_f0 = midi_to_hz(max_pitch)
        return 12000 / (max_f0 * np.log10(max_f0))


class VCA(SynthModule):
    """
    Voltage controlled amplifier.
    """

    def __init__(
        self, sample_rate: int = SAMPLE_RATE, control_rate: int = CONTROL_RATE
    ):
        super().__init__(sample_rate=sample_rate, control_rate=control_rate)

    def __call__(self, control_in: np.array, audio_in: np.array):
        control_in = np.clip(control_in, 0, 1)
        audio_in = np.clip(audio_in, -1, 1)
        amp = self.control_to_sample_rate(control_in)
        audio_in = fix_length(audio_in, len(amp))
        return amp * audio_in


class NoiseModule(SynthModule):
    """
    Adds noise.
    """

    def __init__(
        self,
        ratio: float = 0.25,
        sample_rate: int = SAMPLE_RATE,
        control_rate: int = CONTROL_RATE,
    ):
        super().__init__(sample_rate=sample_rate, control_rate=control_rate)
        self.add_parameters(
            [
                Parameter("ratio", ratio, 0, 1),
            ]
        )

    def __call__(self, audio_in: np.ndarray):
        noise = self.noise_of_length(audio_in)
        return crossfade(audio_in, noise, self.parameters["ratio"].value)

    @staticmethod
    def noise_of_length(audio_in: np.ndarray):
        return np.random.rand(len(audio_in))


class DummyModule(SynthModule):
    """
    A dummy module just encapsulates some parameters and don't create sound.
    This is a temporary workaround that allows a Synth to contain parameters
    without nesting SynthModules and without forcing Synth to be a SynthModule.
    """

    def __init__(
        self,
        parameters: List[Parameter],
        sample_rate: int = SAMPLE_RATE,
        control_rate: int = CONTROL_RATE,
    ):
        """
        Parameters
        ----------
        dict                :   List of parameter dictionaries, for
                                `Parameter`.,
                                TODO: Take this as a list of Parameters instead?
        """
        super().__init__(sample_rate=sample_rate, control_rate=control_rate)
        self.add_parameters(parameters)

    def __call__(self):
        assert False


# TODO: Remove SynthModule
class Synth(SynthModule):
    """
    A base class for a modular synth, ensuring that all modules
    have the same sample and control rate.
    """

    def __init__(self, modules: Tuple[SynthModule]):
        """
        NOTE: __init__ should only set parameters.
        We shouldn't be doing computations in __init__ because
        the computations will change when the parameters change.
        Instead, consider @property getters that use the instance's parameters.
        """
        sample_rate = modules[0].sample_rate
        control_rate = modules[0].control_rate
        super().__init__(sample_rate=sample_rate, control_rate=control_rate)

        # Parameter list
        # TODO: We can remove this later
        self.parameters = {}

        # Check that we are not mixing different control rates or sample rates
        for m in modules[:1]:
            assert m.sample_rate == sample_rate
            assert m.control_rate == control_rate


class Drum(Synth):
    """
    A package of modules that makes one drum hit.
    """

    def __init__(
        self,
        sustain_duration: float,
        drum_params: DummyModule = DummyModule(
            parameters=[
                Parameter(
                    name="vco_ratio",
                    value=0.5,
                    minimum=0.0,
                    maximum=1.0,
                )
            ]
        ),
        pitch_adsr: ADSR = ADSR(),
        amp_adsr: ADSR = ADSR(),
        vco_1: VCO = SineVCO(),
        vco_2: VCO = SquareSawVCO(),
        noise_module: NoiseModule = NoiseModule(),
        vca: VCA = VCA(),
    ):
        super().__init__(
            modules=[pitch_adsr, amp_adsr, vco_1, vco_2, noise_module, vca]
        )
        assert sustain_duration >= 0

        # We assume that sustain duration is a hyper-parameter,
        # with the mindset that if you are trying to learn to
        # synthesize a sound, you won't be adjusting the sustain_duration.
        # However, this is easily changed if desired.
        self.sustain_duration = sustain_duration

        self.drum_params = drum_params
        self.pitch_adsr = pitch_adsr
        self.amp_adsr = amp_adsr
        self.vco_1 = vco_1
        self.vco_2 = vco_2
        self.noise_module = noise_module
        self.vca = vca

        # Pitch Envelope
        self.connect_parameter("pitch_attack", self.pitch_adsr, "attack")
        self.connect_parameter("pitch_decay", self.pitch_adsr, "decay")
        self.connect_parameter("pitch_sustain", self.pitch_adsr, "sustain")
        self.connect_parameter("pitch_release", self.pitch_adsr, "release")
        self.connect_parameter("pitch_alpha", self.pitch_adsr, "alpha")

        # Amplitude Envelope
        self.connect_parameter("amp_attack", self.amp_adsr, "attack")
        self.connect_parameter("amp_decay", self.amp_adsr, "decay")
        self.connect_parameter("amp_sustain", self.amp_adsr, "sustain")
        self.connect_parameter("amp_release", self.amp_adsr, "release")
        self.connect_parameter("amp_alpha", self.amp_adsr, "alpha")

        # VCO 1
        self.connect_parameter("vco_1_pitch", self.vco_1, "pitch")
        self.connect_parameter("vco_1_mod_depth", self.vco_1, "mod_depth")

        # VCO 2
        self.connect_parameter("vco_2_pitch", self.vco_2, "pitch")
        self.connect_parameter("vco_2_mod_depth", self.vco_2, "mod_depth")
        self.connect_parameter("vco_2_shape", self.vco_2, "shape")

        # Mix between the two VCOs
        self.connect_parameter("vco_ratio", self.drum_params, "vco_ratio")

        # Noise
        self.connect_parameter("noise_ratio", self.noise_module, "ratio")

    def __call__(self):
        # The convention for triggering a note event is that it has
        # the same sustain_duration for both ADSRs.
        sustain_duration = self.sustain_duration
        pitch_envelope = self.pitch_adsr(sustain_duration)
        amp_envelope = self.amp_adsr(sustain_duration)
        pitch_envelope = fix_length(pitch_envelope, len(amp_envelope))

        vco_1_out = self.vco_1(pitch_envelope)
        vco_2_out = self.vco_2(pitch_envelope)

        audio_out = crossfade(
            vco_1_out, vco_2_out, self.parameters["vco_ratio"].value
        )

        audio_out = self.noise_module(audio_out)

        return self.vca(amp_envelope, audio_out)


class SVF(SynthModule):
    """
    A State Variable Filter that can do low-pass, high-pass, band-pass, and
    band-reject filtering. Allows modulation of the cutoff frequency and an
    adjustable resonance parameter. Can self-oscillate to make a sinusoid oscillator.

    Parameters
    ----------

    mode (str)              :   filter type, one of LPF, HPF, BPF, or BSF
    cutoff (float)          :   cutoff frequency in Hz must be between 5 and half the
                                sample rate. Defaults to 1000Hz
    resonance (float)       :   filter resonance, or "Quality Factor". Higher values
                                cause the filter to resonate more. Must be greater than
                                0.5. Defaults to 0.707.
    self_oscillate (bool)   :   Set the filter into self-oscillation mode, which turns
                                this into a sine wave oscillator with the filter cutoff
                                as the frequency. Defaults to False.
    sample_rate (float)     :   Processing sample rate.
    """

    def __init__(
        self,
        mode: str,
        cutoff: float = 1000,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
        control_rate: int = CONTROL_RATE,
    ):
        super().__init__(sample_rate=sample_rate, control_rate=control_rate)
        self.mode = mode
        self.self_oscillate = self_oscillate
        self.add_parameters(
            [
                Parameter("cutoff", cutoff, 5, self.sample_rate / 2.0, curve="log"),
                Parameter("resonance", resonance, 0.5, 1000, curve="log"),
            ]
        )

    def __call__(
        self,
        audio: np.ndarray,
        cutoff_mod: np.ndarray = None,
        cutoff_mod_amount: float = 0.0,
    ) -> np.ndarray:
        """
        Process audio samples and return filtered results.

        Parameters
        ----------

        audio (np.ndarray)          :   Audio samples to filter
        cutoff_mod (np.ndarray)     :   Control signal used to modulate the filter
                                        cutoff. Values must be in range [0,1]
        cutoff_mod_amount (float)   :   How much to apply the control signal to the
                                        filter cutoff in Hz. Can be positive or
                                        negative. Defaults to 0.
        """

        h = np.zeros(2)
        y = np.zeros_like(audio)

        # Calculate initial coefficients
        cutoff = self.parameters["cutoff"].value
        coeff0, coeff1, rho = self.calculate_coefficients(cutoff)

        # Check if there is a filter cutoff envelope to apply
        if cutoff_mod_amount != 0.0:
            # Cutoff modulation must be same length as audio input
            assert len(cutoff_mod) == len(audio)

        # Processing loop
        for i in range(len(audio)):

            # If there is a cutoff modulation envelope, update coefficients
            if cutoff_mod_amount != 0.0:
                cutoff_val = cutoff + cutoff_mod[i] * cutoff_mod_amount
                coeff0, coeff1, rho = self.calculate_coefficients(cutoff_val)

            # Calculate each of the filter components
            hpf = coeff0 * (audio[i] - rho * h[0] - h[1])
            bpf = coeff1 * hpf + h[0]
            lpf = coeff1 * bpf + h[1]

            # Feedback samples
            h[0] = coeff1 * hpf + bpf
            h[1] = coeff1 * bpf + lpf

            if self.mode == "LPF":
                y[i] = lpf
            elif self.mode == "BPF":
                y[i] = bpf
            elif self.mode == "BSF":
                y[i] = hpf + lpf
            else:
                y[i] = hpf

        return y

    def calculate_coefficients(self, cutoff: float) -> (float, float, float):
        """
        Calculates the filter coefficients for SVF.

        Parameters
        ----------
        cutoff (float)  :   Filter cutoff frequency in Hz.
        """

        g = np.tan(np.pi * cutoff / self.sample_rate)
        resonance = self.parameters["resonance"].value
        R = 0.0 if self.self_oscillate else 1.0 / (2.0 * resonance)
        coeff0 = 1.0 / (1.0 + 2.0 * R * g + g * g)
        coeff1 = g
        rho = 2.0 * R + g

        return coeff0, coeff1, rho


class LowPassSVF(SVF):
    """
    Low-pass filter using SVF architecture
    """

    def __init__(
        self,
        cutoff: float = 1000,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(
            mode="LPF",
            cutoff=cutoff,
            resonance=resonance,
            self_oscillate=self_oscillate,
            sample_rate=sample_rate,
        )


class HighPassSVF(SVF):
    """
    High-pass filter using SVF architecture
    """

    def __init__(
        self,
        cutoff: float = 1000,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(
            mode="HPF",
            cutoff=cutoff,
            resonance=resonance,
            self_oscillate=self_oscillate,
            sample_rate=sample_rate,
        )


class BandPassSVF(SVF):
    """
    Band-pass filter using SVF architecture
    """

    def __init__(
        self,
        cutoff: float = 1000,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(
            mode="BPF",
            cutoff=cutoff,
            resonance=resonance,
            self_oscillate=self_oscillate,
            sample_rate=sample_rate,
        )


class BandRejectSVF(SVF):
    """
    Band-reject / band-stop filter using SVF architecture
    """

    def __init__(
        self,
        cutoff: float = 1000,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(
            mode="BSF",
            cutoff=cutoff,
            resonance=resonance,
            self_oscillate=self_oscillate,
            sample_rate=sample_rate,
        )


class FIR(SynthModule):
    """
    A finite impulse response low-pass filter. Uses convolution with a symmetric
    windowed sinc function.

    Parameters
    ----------

    cutoff (float)      :   cutoff frequency of low-pass in Hz, must be between 5 and
                            half the sampling rate. Defaults to 1000Hz.
    filter_length (int) :   The length of the filter in samples. A longer filter will
                            result in a steeper filter cutoff. Should be greater than 4.
                            Defaults to 512 samples.
    sample_rate (int)   :   Sampling rate to run processing at.
    """

    def __init__(
        self,
        cutoff: float = 1000,
        filter_length: int = 512,
        sample_rate: int = SAMPLE_RATE,
        control_rate: int = CONTROL_RATE,
    ):
        super().__init__(sample_rate=sample_rate, control_rate=control_rate)
        self.add_parameters(
            [
                Parameter("cutoff", cutoff, 5, sample_rate / 2.0, curve="log"),
                Parameter("length", filter_length, 4, 4096),
            ]
        )

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Filter audio samples
        TODO: Cutoff frequency modulation, if there is an efficient way to do it

        Parameters
        ----------

        audio (np.ndarray)  :   audio samples to filter
        """

        impulse = self.windowed_sinc(
            self.parameters["cutoff"].value, self.parameters["length"].value
        )
        y = np.convolve(audio, impulse)
        return y

    def windowed_sinc(self, cutoff: float, filter_length: int) -> np.ndarray:
        """
        Calculates the impulse response for FIR low-pass filter using the
        windowed sinc function method

        Parameters
        ----------

        cutoff (float)      :   Low-pass cutoff frequency in Hz. Must be between 0 and
                                half the sampling rate.
        filter_length (int) :   Length of the filter impulse response to create. Creates
                                a symmetric filter so if this is even then the filter
                                returned will have a length of filter_length + 1.
        """

        # Normalized frequency
        omega = 2 * np.pi * cutoff / self.sample_rate

        # Create a symmetric sinc function
        half_length = int(filter_length / 2)
        t = np.arange(0 - half_length, half_length + 1)
        ir = np.sin(t * omega)
        ir[half_length] = omega
        ir = np.divide(ir, t, out=ir, where=t != 0)

        # Window using blackman-harris
        n = np.arange(len(ir))
        cos_a = np.cos(2 * np.pi * n / len(n))
        cos_b = np.cos(4 * np.pi * n / len(n))
        window = 0.42 - 0.5 * cos_a + 0.08 * cos_b
        ir *= window

        return ir


class MovingAverage(SynthModule):
    """
    A finite impulse response moving average filter.

    Parameters
    ----------

    filter_length (int) :   Length of filter and number of samples to take average over.
                            Must be greater than 0. Defaults to 32.
    sample_rate (int)   :   Sampling rate to run processing at.
    """

    def __init__(
        self,
        filter_length: int = 32,
        sample_rate: int = SAMPLE_RATE,
        control_rate: int = CONTROL_RATE,
    ):
        super().__init__(sample_rate=sample_rate, control_rate=control_rate)
        self.add_parameters(
            [
                Parameter("length", filter_length, 1, 4096),
            ]
        )

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Filter audio samples

        Parameters
        ----------

        audio (np.ndarray)  :   audio samples to filter
        """
        length = self.parameters["length"].value
        impulse = np.ones(length) / length
        y = np.convolve(audio, impulse)
        return y
