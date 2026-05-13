import numpy as np
import audiomentations as am
from pathlib import Path

class ClassroomAugmenter:
    def __init__(self, sample_rate: int = 16000, ir_dir: str = "data/augmentation/ir", backnoise_dir: str = "data/augmentation/backnoise", 
                 intensity: float = 1.0, p_reverb: float = 0.5, p_noise: float = 0.5, p_bandpass: float = 0.5, p_gain: float = 0.5):
        self.sample_rate = sample_rate
        self.max_samples = int(30.0 * sample_rate)
        
        ir_path = Path(ir_dir)
        backnoise_path = Path(backnoise_dir)
        
        self.transforms_with_p = []
        
        # 1. Reverb
        if ir_path.exists() and any(ir_path.iterdir()):
            self.transforms_with_p.append((
                am.ApplyImpulseResponse(
                    ir_path=str(ir_path),
                    p=1.0,
                    leave_length_unchanged=True
                ),
                p_reverb
            ))
            
        # 2. Background Noise
        if backnoise_path.exists() and any(backnoise_path.iterdir()):
            self.transforms_with_p.append((
                am.AddBackgroundNoise(
                    sounds_path=str(backnoise_path),
                    min_snr_db=max(0.0, 5.0 / intensity if intensity > 0 else 5.0),
                    max_snr_db=max(5.0, 15.0 / intensity if intensity > 0 else 15.0),
                    p=1.0
                ),
                p_noise
            ))
            
        # 3. Hardware Limit (BandPass)
        self.transforms_with_p.append((
            am.BandPassFilter(
                min_center_freq=300,
                max_center_freq=6000,
                p=1.0
            ),
            p_bandpass
        ))
        
        # 4. Volume Fluctuation (Gain Transition)
        self.transforms_with_p.append((
            am.GainTransition(
                min_gain_db=-1.5 * intensity,
                max_gain_db=1.5 * intensity,
                min_duration=0.5,
                max_duration=3.0,
                p=1.0
            ),
            p_gain
        ))
        
        # For direct testing
        self.bandpass = am.BandPassFilter(min_center_freq=300, max_center_freq=6000, p=1.0)
        self.gain_transition = am.GainTransition(min_gain_db=-1.5, max_gain_db=1.5, min_duration=0.5, max_duration=3.0, p=1.0)

    def apply_bandpass(self, audio: np.ndarray) -> np.ndarray:
        return self.bandpass(samples=audio, sample_rate=self.sample_rate)
        
    def apply_volume_fluctuation(self, audio: np.ndarray) -> np.ndarray:
        return self.gain_transition(samples=audio, sample_rate=self.sample_rate)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies the augmentation pipeline and strictly enforces the duration constraint.
        Max 2 modifications per audio file.
        """
        active_transforms = []
        for transform, p_val in self.transforms_with_p:
            if np.random.random() < p_val:
                active_transforms.append(transform)
                
        # Limit to 2 max
        if len(active_transforms) > 2:
            np.random.shuffle(active_transforms)
            active_transforms = active_transforms[:2]
            
        augmented = audio.copy()
        for transform in active_transforms:
            augmented = transform(samples=augmented, sample_rate=self.sample_rate)
        
        # Strict duration constraint: trim to max 30.0s
        if len(augmented) > self.max_samples:
            augmented = augmented[:self.max_samples]
            
        return augmented
