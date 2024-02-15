#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import os
import pywt
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
import wandb
import auraloss
import collections
from tqdm import tqdm
import pretty_midi
import matplotlib.pyplot as plt
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
print(torch.cuda.is_available())
import plotly.graph_objects as go
from torch.optim import lr_scheduler


# # Set Seeds

# In[2]:


seed_value = 3407
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.set_float32_matmul_precision('high')


# # Construct Teh Datas

# In[3]:


path = "D:/Github/phd-drum-sep/Data/musdb18hq/"


# In[4]:


os.listdir(path)


# In[5]:


train = list(os.listdir(path+'train'))
test = list(os.listdir(path+'test'))


# In[6]:


sources = ['drum', 'bass', 'other', 'vocals']


# In[7]:


all_scenes = {}
counter = 0
for idx, val in tqdm(enumerate(train)):
    p = path + 'train/' + val + "/"
    info = torchaudio.info(f"{p}mixture.wav")
    seconds = info.num_frames // 44100
    for i in range(0, seconds - 10, 10):
        start_point = i * 44100
        if start_point + 441000 < info.num_frames:
            all_scenes[counter] = {'music_path': p, 'start_point': start_point, 'length': 441000, 'frames' : info.num_frames}
            counter += 1


# In[8]:


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


# # Data Loaders

# In[9]:


class AudioDataGenerator(Dataset):
    def __init__(self, data, sample_rate=HDEMUCS_HIGH_MUSDB.sample_rate, segment_length = 10):
        self.data = data
        self.sample_rate = sample_rate
        self.segment_length = sample_rate * segment_length

    def __len__(self):
        return len(self.data)
    
    def load_audio(self, path, start_point, filename):
        audio_tensors = []
        file = filename
        segment, _ = torchaudio.load(f"{path}/{file}", frame_offset=start_point, num_frames=self.segment_length)
        audio_tensors.append(segment)
        return torch.cat(audio_tensors, dim=0)

    def load_roll(self, path, start_point, frames):
        midi = path + '/mixture.wav.mid'
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
        roll_tensor = self.load_roll(audio_path, start_point, sample['frames'])
        return mixture_tensor, drum_tensor, roll_tensor


# ## Lightning Data Module

# In[10]:


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

# In[11]:


class DrumDemucs(pl.LightningModule):
    def __init__(self):
        super(DrumDemucs, self).__init__()

        self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 4096],
            hop_sizes=[256, 512, 1024],
            win_lengths=[1024, 2048,4096],
            #scale="mel", 
            #n_bins=128,
            sample_rate=44100,
            device="cuda"
        )

        self.loss_fn_2 = auraloss.time.SISDRLoss()

        self.loss_used = 0

        sources = ['drum',
                   # 'noise',
                   ]
        
        self.demucs_mixer =  torchaudio.models.HDemucs(
            sources=sources,
            audio_channels=7,
            depth=6,
        )

        self.flatten_0 = torch.nn.Conv1d(7, 256, 7, padding=3)
        self.flatten_1 = torch.nn.Conv1d(256, 2, 3, padding=1)
        


    def compute_loss(self, outputs, ref_signals):
        loss = self.loss_fn(outputs, ref_signals) + self.loss_fn_2(outputs, ref_signals)
        return loss

    def forward(self, audio, drumroll):
        # print(drumroll.size())
        # print(audio.size())

        # drumroll_flatten = self.flatten_0(drumroll)
        # drumroll_flatten = self.flatten_1(drumroll_flatten)

        to_mix = torch.cat([audio, drumroll], axis=1)
        
        # print(to_mix.size())

        out = self.demucs_mixer(to_mix)

        # print(out.size())

        #left = out[:, 0, 0, :]
        #right = out[:, 0, 1, :]

        #out_2 = torch.stack([left, right])
        
        out_2 = self.flatten_0(out.squeeze(1))
        out_2 = self.flatten_1(out_2)

        #print(out_2.size())
        

        return out_2
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        audio, drum, drumroll = batch
        
        outputs = self.forward(audio, drumroll)
        # print(outputs.size())

        if batch_idx % 256 == 0:
            input_signal = audio[0].cpu().detach().numpy().T
            generated_signal = outputs[0].cpu().detach().numpy().T
            drum_signal = drum[0].cpu().detach().numpy().T 
            wandb.log({'audio_input': [wandb.Audio(input_signal, caption="Input", sample_rate=44100)]})
            wandb.log({'audio_reference': [wandb.Audio(drum_signal, caption="Reference", sample_rate=44100)]})
            wandb.log({'audio_output': [wandb.Audio(generated_signal, caption="Output", sample_rate=44100)]})

            # # Create a Plotly figure
            # fig = go.Figure()
            
            # fig.add_trace(go.Scatter(y=drumroll[0].cpu().detach().numpy(), mode='lines', name=f'Signal'))
            
            # # Update layout for a better visualization
            # fig.update_layout(
            #     title='Tensor Visualization',
            #     xaxis_title='Sample',
            #     yaxis_title='Value',
            #     legend_title='Signal',
            #     margin=dict(l=20, r=20, t=40, b=20),
            #     hovermode='closest'
            # )

            # wandb.log({"Tensor Interactive Plot": wandb.Plotly(fig)})

        loss = self.compute_loss(outputs, drum)         

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    

    def configure_optimizers(self):
        # Define your optimizer and optionally learning rate scheduler here
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=64, gamma=0.1)
        return [optimizer], [scheduler]
        


# ## Lightning Callbacks

# In[12]:


class SaveModelEveryNSteps(pl.Callback):
    def __init__(self, save_step_frequency=256,):
        self.save_step_frequency = save_step_frequency
        self.save_path = "D://Github//phd-drum-sep//models//DrumSep//"
        os.makedirs(self.save_path , exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.save_step_frequency == 0:
            checkpoint_path = os.path.join(self.save_path, f"step_{trainer.global_step + 1}.ckpt")
            trainer.save_checkpoint(checkpoint_path)


# # Train Loop

# In[13]:


model = DrumDemucs()


# In[14]:


wandb_logger = WandbLogger(project='DrumDemucs', log_model='all')


# In[15]:


audio_data_module = AudioDataModule(all_scenes, batch_size=4, num_workers=0, persistent_workers=False)


# In[16]:


trainer = pl.Trainer(
    max_epochs=1000,
    accelerator="gpu", 
    devices=-1,
    logger=wandb_logger,
    callbacks=[SaveModelEveryNSteps()],
    accumulate_grad_batches=1,
    gradient_clip_val=5,
)


# In[ ]:


trainer.fit(model, audio_data_module)


# In[ ]:





# In[ ]:




