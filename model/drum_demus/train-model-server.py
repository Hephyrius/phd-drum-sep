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
    for i in range(0, seconds - 10, 10):
        start_point = i * 44100
        if start_point + 441000 < info.num_frames:
            all_scenes[counter] = {'music_path': p, 'adtof_path': p2 , 'start_point': start_point, 'length': 441000, 'frames' : info.num_frames}
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
    def __init__(self, data, sample_rate=HDEMUCS_HIGH_MUSDB.sample_rate, segment_length = 10):
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

class DrumDemucs(pl.LightningModule):
    def __init__(self):
        super(DrumDemucs, self).__init__()

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

        sources = ['drum',
                   'noise',
                   ]
        
        self.demucs_mixer =  torchaudio.models.HDemucs(
            sources=sources,
            audio_channels=7,
            depth=6,
        )

        self.out_conv = nn.Conv1d(in_channels=7, out_channels=2, kernel_size=1)
        self.out = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=1)      


    def compute_loss(self, outputs, ref_signals):
        loss = self.loss_fn(outputs, ref_signals) + self.loss_fn_2(outputs, ref_signals) +  self.loss_fn_3(outputs, ref_signals)
        return loss

    def forward(self, audio, drumroll):
        to_mix = torch.cat([audio, drumroll], axis=1)
        out = self.demucs_mixer(to_mix)
        out_2 = self.out_conv(out[:, 0, :, :])
        out_2 = self.out(out_2)
        # out_2 = torch.tanh(out_2)

        return out_2
    
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

    model = DrumDemucs()
    wandb_logger = WandbLogger(project='DrumDemucs', log_model='all')
    audio_data_module = AudioDataModule(all_scenes, batch_size=4, num_workers=0, persistent_workers=False)

    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu", 
        devices=1,
        logger=wandb_logger,
        callbacks=[WandbCleanupCallback(cleanup_hours=2, artifact_max_size='20GB')],
        # accumulate_grad_batches=2,
        gradient_clip_val=5,
    )

    trainer.fit(model, audio_data_module, ckpt_path='/homes/hsk01/drum_sep/model/epoch=111-step=62496.ckpt')