#!/usr/bin/env python
# coding: utf-8

# # Imports

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import random
from pytorch_lightning.loggers import WandbLogger
import subprocess
import wandb
import auraloss
import collections
from tqdm import tqdm
import librosa
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
print(torch.cuda.is_available())
from torch.optim import lr_scheduler
import pretty_midi
from typing import Optional, Tuple

# # Set Seeds

seed_value = 3407
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.set_float32_matmul_precision('high')


# # Construct Teh Datas


path = "/import/c4dm-datasets/musdb18hq_v2/"
path_adtof = "/homes/hsk01/drum_sep/data/adtof/adtof/"


train = list(os.listdir(path+'train'))
test = list(os.listdir(path+'test'))


sources = ['drum', 'bass', 'other', 'vocals']

all_scenes = {}
counter = 0
for idx, val in tqdm(enumerate(train)):
    p = path + 'train/' + val + "/"
    p2 = path_adtof + 'train/' + val + "/adtof.mid"
    
    info = torchaudio.info(f"{p}mixture.wav")
    seconds = info.num_frames // 44100
    for i in range(0, seconds - 4, 4):
        start_point = i * 44100
        if start_point + (44100 * 4) < info.num_frames:
            all_scenes[counter] = {'music_path': p, 'adtof_path': p2 , 'start_point': start_point, 'length': (44100 * 4),'frames' : info.num_frames}
            counter += 1


def turn_transcription_into_roll(transcription, frames):
    # Determine your sampling frequency (frames per second)
    fs = 44100
    
    piano_roll_length = int(frames)
    
    # Initialize the piano roll array
    piano_roll = np.zeros((64, piano_roll_length))
    
    # Fill in the piano roll array
    for note in transcription.instruments[0].notes:
        # Convert start and end times to frame indices
        start_frame = int(np.floor(note.start * fs))
        end_frame = int(np.ceil(note.end * fs))
        
        # Set the corresponding frames to 1 (or note.velocity for a velocity-sensitive representation)
        piano_roll[note.pitch, start_frame:end_frame] = 1  # Or use note.velocity
        
    roll = np.vstack([piano_roll[35:36, :], piano_roll[38:39, :], piano_roll[42:43, :], piano_roll[47:48, :], piano_roll[49:50, :]])
    return roll


class AudioDataGenerator(Dataset):
    def __init__(self, data, sample_rate=HDEMUCS_HIGH_MUSDB.sample_rate, segment_length = 4):
        self.data = data
        self.sample_rate = sample_rate
        self.segment_length = sample_rate * segment_length

    def __len__(self):
        return len(self.data)
    
    def load_audio(self, path, start_point, filename):
        file = filename
        start_seconds = start_point // 44100
        duration = self.segment_length // 44100

        segment, _ = librosa.load(f"{path}/{file}", sr=44100, mono=False, offset=start_seconds, duration=duration)
        segment_tensor = torch.tensor(segment)

        return segment_tensor

    def load_roll(self, path, start_point, frames):
        midi = path
        transcription = pretty_midi.PrettyMIDI(midi)
        roll = turn_transcription_into_roll(transcription, frames)
        # print(roll.shape)
        roll = roll[:, start_point: start_point + self.segment_length]
        return torch.from_numpy(roll).float()

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        # Load audio as a tensor
        audio_path = sample['music_path']

        start_point = sample['start_point']

        mixture_tensor = self.load_audio(audio_path, start_point,'mixture.wav')
        drum_tensor = self.load_audio(audio_path, start_point,'drums.wav')
        roll_tensor = self.load_roll(sample['adtof_path'], start_point, sample['frames'])
        return mixture_tensor, drum_tensor, roll_tensor


# ## Lightning Data Module

class AudioDataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size=32, num_workers=0, persistent_workers=False, shuffle=False):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers=persistent_workers
        self.shuffle = shuffle

    def setup(self, stage=None):
        # Split your data here if necessary, e.g., into train, validation, test
        self.dataset = AudioDataGenerator(self.data)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers = self.num_workers, persistent_workers=self.persistent_workers)

    # Implement val_dataloader() and test_dataloader() if you have validation and test data


# # making the model


class ConvBlock(torch.nn.Module):
    """1D Convolutional block.

    Args:
        io_channels (int): The number of input/output channels, <B, Sc>
        hidden_channels (int): The number of channels in the internal layers, <H>.
        kernel_size (int): The convolution kernel size of the middle layer, <P>.
        padding (int): Padding value of the convolution in the middle layer.
        dilation (int, optional): Dilation value of the convolution in the middle layer.
        no_redisual (bool, optional): Disable residual block/output.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

    def __init__(
        self,
        io_channels: int,
        hidden_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int = 1,
        no_residual: bool = False,
    ):
        super().__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=io_channels, out_channels=hidden_channels, kernel_size=1),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
            torch.nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_channels,
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
        )

        self.res_out = (
            None
            if no_residual
            else torch.nn.Conv1d(in_channels=hidden_channels, out_channels=io_channels, kernel_size=1)
        )
        self.skip_out = torch.nn.Conv1d(in_channels=hidden_channels, out_channels=io_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        feature = self.conv_layers(input)
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)
        skip_out = self.skip_out(feature)
        return residual, skip_out




# In[12]:


class MaskGenerator(torch.nn.Module):
    """TCN (Temporal Convolution Network) Separation Module

    Generates masks for separation.

    Args:
        input_dim (int): Input feature dimension, <N>.
        num_sources (int): The number of sources to separate.
        kernel_size (int): The convolution kernel size of conv blocks, <P>.
        num_featrs (int): Input/output feature dimenstion of conv blocks, <B, Sc>.
        num_hidden (int): Intermediate feature dimention of conv blocks, <H>
        num_layers (int): The number of conv blocks in one stack, <X>.
        num_stacks (int): The number of conv block stacks, <R>.
        msk_activate (str): The activation function of the mask output.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

    def __init__(
        self,
        input_dim: int,
        num_sources: int,
        kernel_size: int,
        num_feats: int,
        num_hidden: int,
        num_layers: int,
        num_stacks: int,
        msk_activate: str,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_sources = num_sources

        self.input_norm = torch.nn.GroupNorm(num_groups=1, num_channels=input_dim, eps=1e-8)
        self.input_conv = torch.nn.Conv1d(in_channels=input_dim, out_channels=num_feats, kernel_size=1)

        self.receptive_field = 0
        self.conv_layers = torch.nn.ModuleList([])
        for s in range(num_stacks):
            for l in range(num_layers):
                multi = 2**l
                self.conv_layers.append(
                    ConvBlock(
                        io_channels=num_feats,
                        hidden_channels=num_hidden,
                        kernel_size=kernel_size,
                        dilation=multi,
                        padding=multi,
                        # The last ConvBlock does not need residual
                        no_residual=(l == (num_layers - 1) and s == (num_stacks - 1)),
                    )
                )
                self.receptive_field += kernel_size if s == 0 and l == 0 else (kernel_size - 1) * multi
        self.output_prelu = torch.nn.PReLU()
        self.output_conv = torch.nn.Conv1d(
            in_channels=num_feats,
            out_channels=input_dim * num_sources,
            kernel_size=1,
        )
        if msk_activate == "sigmoid":
            self.mask_activate = torch.nn.Sigmoid()
        elif msk_activate == "relu":
            self.mask_activate = torch.nn.ReLU()
        elif msk_activate == "prelu":
            self.mask_activate = torch.nn.PReLU()
        else:
            raise ValueError(f"Unsupported activation {msk_activate}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Generate separation mask.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, features, frames]

        Returns:
            Tensor: shape [batch, num_sources, features, frames]
        """
        batch_size = input.shape[0]
        feats = self.input_norm(input)
        feats = self.input_conv(feats)
        output = 0.0
        for layer in self.conv_layers:
            residual, skip = layer(feats)
            if residual is not None:  # the last conv layer does not produce residual
                feats = feats + residual
            output = output + skip
        output = self.output_prelu(output)
        output = self.output_conv(output)
        output = self.mask_activate(output)
        return output.view(batch_size, self.num_sources, self.input_dim, -1)


# In[13]:


class ConvTasNet(torch.nn.Module):
    """Conv-TasNet architecture introduced in
    *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*
    :cite:`Luo_2019`.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.

    See Also:
        * :class:`torchaudio.pipelines.SourceSeparationBundle`: Source separation pipeline with pre-trained models.

    Args:
        num_sources (int, optional): The number of sources to split.
        enc_kernel_size (int, optional): The convolution kernel size of the encoder/decoder, <L>.
        enc_num_feats (int, optional): The feature dimensions passed to mask generator, <N>.
        msk_kernel_size (int, optional): The convolution kernel size of the mask generator, <P>.
        msk_num_feats (int, optional): The input/output feature dimension of conv block in the mask generator, <B, Sc>.
        msk_num_hidden_feats (int, optional): The internal feature dimension of conv block of the mask generator, <H>.
        msk_num_layers (int, optional): The number of layers in one conv block of the mask generator, <X>.
        msk_num_stacks (int, optional): The numbr of conv blocks of the mask generator, <R>.
        msk_activate (str, optional): The activation function of the mask output (Default: ``sigmoid``).
    """

    def __init__(
        self,
        num_sources: int = 2,
        # encoder/decoder parameters
        enc_kernel_size: int = 16,
        enc_num_feats: int = 512,
        # mask generator parameters
        msk_kernel_size: int = 3,
        msk_num_feats: int = 128,
        msk_num_hidden_feats: int = 512,
        msk_num_layers: int = 8,
        msk_num_stacks: int = 3,
        msk_activate: str = "sigmoid",
    ):
        super().__init__()

        self.num_sources = num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2

        self.encoder = torch.nn.Conv1d(
            in_channels=7,
            out_channels=enc_num_feats,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )
        self.mask_generator = MaskGenerator(
            input_dim=enc_num_feats,
            num_sources=num_sources,
            kernel_size=msk_kernel_size,
            num_feats=msk_num_feats,
            num_hidden=msk_num_hidden_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks,
            msk_activate=msk_activate,
        )
        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=enc_num_feats,
            out_channels=2,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )

    def _align_num_frames_with_strides(self, input: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad input Tensor so that the end of the input tensor corresponds with

        1. (if kernel size is odd) the center of the last convolution kernel
        or 2. (if kernel size is even) the end of the first half of the last convolution kernel

        Assumption:
            The resulting Tensor will be padded with the size of stride (== kernel_width // 2)
            on the both ends in Conv1D

        |<--- k_1 --->|
        |      |            |<-- k_n-1 -->|
        |      |                  |  |<--- k_n --->|
        |      |                  |         |      |
        |      |                  |         |      |
        |      v                  v         v      |
        |<---->|<--- input signal --->|<--->|<---->|
         stride                         PAD  stride

        Args:
            input (torch.Tensor): 3D Tensor with shape (batch_size, channels==1, frames)

        Returns:
            Tensor: Padded Tensor
            int: Number of paddings performed
        """
        batch_size, num_channels, num_frames = input.shape
        is_odd = self.enc_kernel_size % 2
        num_strides = (num_frames - is_odd) // self.enc_stride
        num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
        if num_remainings == 0:
            return input, 0

        num_paddings = self.enc_stride - num_remainings
        pad = torch.zeros(
            batch_size,
            num_channels,
            num_paddings,
            dtype=input.dtype,
            device=input.device,
        )
        return torch.cat([input, pad], 2), num_paddings
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform source separation. Generate audio source waveforms.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, channel==1, frames]

        Returns:
            Tensor: 3D Tensor with shape [batch, channel==num_sources, frames]
        """

        # B: batch size
        # L: input frame length
        # L': padded input frame length
        # F: feature dimension
        # M: feature frame length
        # S: number of sources

        padded, num_pads = self._align_num_frames_with_strides(input)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]
        feats = self.encoder(padded)  # B, F, M
        masked = self.mask_generator(feats) * feats.unsqueeze(1)  # B, S, F, M
        masked = masked.view(batch_size * self.num_sources, self.enc_num_feats, -1)  # B*S, F, M
        decoded = self.decoder(masked)  # B*S, 1, L'
        out = decoded.reshape(batch_size, 4, -1)
        # print(out.shape)
        return out


# In[14]:


class DrumConvTasnet(pl.LightningModule):
    def __init__(self):
        super(DrumConvTasnet, self).__init__()

        self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[1024, 2048, 4096],
                    hop_sizes=[256, 512, 1024],
                    win_lengths=[1024, 2048, 4096],
                    scale="mel", 
                    n_bins=150,
                    sample_rate=44100,
                    device="cuda"
                )

        self.loss_fn_2 = auraloss.time.SISDRLoss()

        self.loss_fn_3 = torch.nn.L1Loss()

        self.loss_used = 0
        
        self.conv_tasnet =  ConvTasNet(
            num_sources=2,
            enc_kernel_size=16,
            enc_num_feats=512,
            msk_kernel_size=3,
            msk_num_feats=128,
            msk_num_hidden_feats=512,
            msk_num_layers=8,
            msk_num_stacks=3,
            msk_activate="prelu",
        )

        self.out = nn.Conv1d(4, 2, kernel_size=1)

    def compute_loss(self, outputs, ref_signals):
        loss = self.loss_fn(outputs, ref_signals) + self.loss_fn_2(outputs, ref_signals) +  self.loss_fn_3(outputs, ref_signals)
        return loss

    def forward(self, audio, drumroll):
        to_mix = torch.cat([audio, drumroll], axis=1)
        out = self.conv_tasnet(to_mix)
        out = self.out(out)
        return out
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        audio, drum, drumroll = batch
        
        outputs = self.forward(audio, drumroll)
        # print(outputs.size())

        if batch_idx % 64 == 0:
            input_signal = audio[0].cpu().detach().numpy().T
            generated_signal = outputs[0].cpu().detach().numpy().T
            drum_signal = drum[0].cpu().detach().numpy().T 
            wandb.log({'audio_input': [wandb.Audio(input_signal, caption="Input", sample_rate=44100)]})
            wandb.log({'audio_reference': [wandb.Audio(drum_signal, caption="Reference", sample_rate=44100)]})
            wandb.log({'audio_output': [wandb.Audio(generated_signal, caption="Output", sample_rate=44100)]})
             
            for i in range(5):
                wandb.log({f'drum_{i + 1}': [wandb.Audio(drumroll[0].cpu().detach().numpy()[i, :], caption="Output", sample_rate=44100)]})


        loss = self.compute_loss(outputs, drum)         

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    

    def configure_optimizers(self):
        # Define your optimizer and optionally learning rate scheduler here
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        return [optimizer], [scheduler]
        
  


# ## Lightning Callbacks

class WandbCleanupCallback(pl.Callback):
    def __init__(self, cleanup_hours=1, artifact_max_size='10GB'):
        """
        Cleanup callback for wandb logger.

        Parameters:
        - cleanup_hours: int. Runs older than this many hours will be cleaned up. Ensure this is longer than your longest run.
        - artifact_max_size: str. The maximum size allowed for the cache. Artifacts above this size will be cleaned up.
        """
        super().__init__()
        self.cleanup_hours = cleanup_hours
        self.artifact_max_size = artifact_max_size

    def on_train_epoch_end(self, trainer, pl_module):
        # Cleanup old runs
        subprocess.run(["wandb", "sync", "--clean-old-hours", str(self.cleanup_hours), "--no-include-online"])
        # Cleanup artifact cache
        subprocess.run(["wandb", "artifact", "cache", "cleanup", self.artifact_max_size])


# # # Train Loop
if __name__ == "__main__":

    model = DrumConvTasnet()
    wandb_logger = WandbLogger(project='DrumConvTasnetModified', log_model='all')
    audio_data_module = AudioDataModule(all_scenes, batch_size=2, num_workers=0, persistent_workers=False)

    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu", 
        devices=1,
        logger=wandb_logger,
        callbacks=[WandbCleanupCallback(cleanup_hours=2, artifact_max_size='20GB')],
        accumulate_grad_batches=2,
        gradient_clip_val=5,
    )

    trainer.fit(model, audio_data_module, ckpt_path='/homes/hsk01/drum_sep/conv/checkpoints/epoch=7-step=11296.ckpt')