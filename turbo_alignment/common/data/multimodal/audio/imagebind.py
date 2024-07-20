from pathlib import Path

import torch
import torchaudio
from loguru import logger
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from torchvision import transforms

from turbo_alignment.common.data.multimodal.image.base import BaseImageReader
from turbo_alignment.common.data.multimodal.registry import AudioModalityReaderRegistry
from turbo_alignment.settings.modality import ModalityReader


@AudioModalityReaderRegistry.register(ModalityReader.IMAGEBIND)
class ImageBindAudioReader(BaseImageReader):
    def __init__(
        self,
        *args,
        num_mel_bins=128,
        target_length=204,
        sample_rate=16000,
        clip_duration=2,
        clips_per_video=3,
        mean=-4.268,
        std=9.138,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.clips_per_video = clips_per_video
        self.mean = mean
        self.std = std

        self.clip_sampler = ConstantClipsPerVideoSampler(clip_duration=clip_duration, clips_per_video=clips_per_video)

    def read(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        if self.sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
        all_clips_timepoints = self.get_clip_timepoints(self.clip_sampler, waveform.size(1) / self.sample_rate)
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * self.sample_rate) : int(clip_timepoints[1] * self.sample_rate),
            ]
            waveform_melspec = self.waveform2melspec(
                waveform_clip, self.sample_rate, self.num_mel_bins, self.target_length
            )
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        all_clips = [normalize(ac) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        return all_clips

    @staticmethod
    def get_clip_timepoints(clip_sampler, duration):
        all_clips_timepoints = []
        is_last_clip = False
        end = 0.0
        while not is_last_clip:
            start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
            all_clips_timepoints.append((start, end))
        return all_clips_timepoints

    @staticmethod
    def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
        waveform -= waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=num_mel_bins,
            dither=0.0,
            frame_length=25,
            frame_shift=10,
        )
        fbank = fbank.transpose(0, 1)

        n_frames = fbank.size(1)
        p = target_length - n_frames

        if abs(p) / n_frames > 0.2:
            logger.warning(
                f'Large gap between audio {n_frames=} and {target_length=}. Is the audio_target_length setting correct?'
            )

        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, p), mode='constant', value=0)
        elif p < 0:
            fbank = fbank[:, 0:target_length]

        fbank = fbank.unsqueeze(0)
        return fbank
