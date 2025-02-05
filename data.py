import random
import torch
import torch.utils
import torch.utils.data
import math
import numpy as np

class TAPSdataset:
    def __init__(self, 
                 datapair_list, 
                 segment=None, 
                 stride=None, 
                 shift=None, 
                 with_id=False,
                 with_text=False,
                 ):
        
        self.datapair_list = datapair_list
        self.segment = segment
        self.stride = stride
        self.shift = shift
        self.with_id = with_id
        self.with_text = with_text
        assert self.with_id if self.with_text else True, "with_id must be True if with_text is True"

        # Prepare lists for tm and am audio arrays
        tm_list, am_list = [], []
        for item in self.datapair_list:
            # Load throat and acoustic microphone audio array, and convert to tensor. Add channel dimension
            tm = item["audio.throat_microphone"]['array'].astype('float32')
            am = item["audio.acoustic_microphone"]['array'].astype('float32')
            id = item["speaker_id"] + "_" + item["sentence_id"]
            text = item["text"]
            length = tm.shape[-1]
            tm_list.append((tm, id, text, length))
            am_list.append((am, id, text, length))
        
        # Create Audioset objects for tm and am
        self.tm_set = Audioset(wavs=tm_list, segment=segment, stride=stride, with_id=with_id, with_text=with_text)
        self.am_set = Audioset(wavs=am_list, segment=segment, stride=stride, with_id=with_id, with_text=with_text)
        
    def __len__(self):
        # The length of the dataset is the number of tm_set samples
        return len(self.tm_set)

    def __getitem__(self, index):
        
        if self.with_text:
            tm, id, text = self.tm_set[index]
            am, _, _ = self.am_set[index]
        elif self.with_id:
            tm, id = self.tm_set[index]
            am, _ = self.am_set[index]
        else:
            tm = self.tm_set[index]
            am = self.am_set[index]
        
        # If shift is specified, randomly pick an offset for tm and am
        if self.shift:
            t = am.shape[-1] - self.shift
            # Ensure shift is even and enough frames remain
            assert self.shift % 2 == 0 and t > 0
            offset = random.randint(0, self.shift)
            # Cut both tm and am with the chosen offset
            am = am[..., offset:offset+t]
            tm = tm[..., offset:offset+t]
        
        # If tapsId is True, return the file ID as well
        if self.with_text:
            return tm, am, id, text
        elif self.with_id:
            return tm, am, id
        else:
            return tm, am

class Audioset:
    def __init__(self, wavs=None, segment=None, stride=None, with_id=False, with_text=False):
        # Store the file list and hyperparameters
        self.wavs = wavs
        self.num_examples = []
        self.segment = segment
        self.stride = stride or segment
        self.with_id = with_id
        self.with_text = with_text
        
        # Calculate how many segments (examples) each file can produce
        for _, _, _, wav_length in self.wavs:
            # If no fixed segment length is provided or the file is shorter, only 1 example
            if segment is None or wav_length < segment:
                examples = 1
            else:
                # Otherwise, calculate how many segments fit given stride
                examples = int(math.ceil((wav_length - self.segment) / (self.stride)) + 1)
            self.num_examples.append(examples)

    def __len__(self):
        # The total length is the sum of all examples across files
        return sum(self.num_examples)

    def __getitem__(self, index):
        # Iterate through files and find which file/segment corresponds to 'index'
        for (wav, id, text, _), examples in zip(self.wavs, self.num_examples):
            # If index is larger than current file's examples, skip to the next file
            if index >= examples:
                index -= examples
                continue
            
            # Otherwise, compute the offset based on stride and index
            offset = self.stride * index if self.segment else 0
            # Decide how many frames to load (full file if segment is None)
            num_frames = self.segment if self.segment else len(wav)
            # Slice the waveform
            wav = wav[offset:offset+num_frames]
            # If the loaded waveform is shorter than the segment length, pad it
            if self.segment:
                wav = np.pad(wav, (0, num_frames - wav.shape[-1]), 'constant')
                
            # Add channel dimension
            wav = np.expand_dims(wav, axis=0)
            
            if self.with_text:
                return wav, id, text
            elif self.with_id:
                return wav, id
            else:
                return wav
            

class StepSampler(torch.utils.data.Sampler):
    def __init__(self, length, step):
        # Save the total length and sampling step
        self.step = step
        self.length = length
        
    def __iter__(self):
        # Return indices at intervals of step
        return iter(range(0, self.length, self.step))
    
    def __len__(self):
        # Length is how many indices we can produce based on the step
        return self.length // self.step
    

def validation_collate_fn(batch):
    assert len(batch[0]) == 2, "This collate function only works with 2-tuple batches"
    if len(batch) == 1:
        tm, am = batch[0]
        tm = np.expand_dims(tm, axis=1)
        am = np.expand_dims(am, axis=1)
        tm = torch.from_numpy(tm)
        am = torch.from_numpy(am)
        return tm, am, None
    
    tm, am = zip(*batch)
    
    # Squeeze channel dimension
    tm = [np.squeeze(inp) for inp in tm]
    am = [np.squeeze(inp) for inp in am]
    
    max_length = max(seq.shape[0] for seq in tm)
    padded_tm = np.full((len(tm), max_length), 0.0, dtype=np.float32)
    padded_am = padded_tm.copy()
    mask = padded_tm.copy()
    
    for i, (seq_tm, seq_am) in enumerate(zip(tm, am)):
        padded_tm[i, :seq_tm.shape[0]] = seq_tm
        padded_am[i, :seq_am.shape[0]] = seq_am
        mask[i, :seq_tm.shape[0]] = 1.0

    # Add channel dimension
    padded_tm = np.expand_dims(padded_tm, axis=1)
    padded_am = np.expand_dims(padded_am, axis=1)
    mask = np.expand_dims(mask, axis=1)
    
    padded_tm = torch.from_numpy(padded_tm)
    padded_am = torch.from_numpy(padded_am)
    mask = torch.from_numpy(mask)
    
    return padded_tm, padded_am, mask