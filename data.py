import random
import torch
import torch.utils
import torch.utils.data
import torchaudio
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def filename2tapsID(filename):
    # p##_u##_mic.wav -> p##_u##
    return filename.split('/')[-1][:7]

class TAPSdataset:
    def __init__(self, 
                 datapair_list, 
                 segment=None, 
                 stride=None, 
                 shift=None, 
                 tapsId=False
                 ):
        
        self.datapair_list = datapair_list
        self.segment = segment
        self.stride = stride
        self.shift = shift
        self.tapsId = tapsId

        # Prepare lists for tm and am audio files
        tm_list, am_list = [], []
        for line in self.datapair_list:
            # Each line is assumed to have format "tm|am"
            tm, am = line.split('|')
            # Store tuples of (file_path, number_of_frames_in_that_file)
            tm_list.append((tm, torchaudio.info(tm).num_frames))
            am_list.append((am, torchaudio.info(am).num_frames))
        
        # Create Audioset objects for tm and am
        self.tm_set = Audioset(files=tm_list, segment=segment, stride=stride, with_path=tapsId)
        self.am_set = Audioset(files=am_list, segment=segment, stride=stride, with_path=tapsId)
        
    def __len__(self):
        # The length of the dataset is the number of tm_set samples
        return len(self.tm_set)

    def __getitem__(self, index):
        # If tapsId is True, also retrieve the file name
        if self.tapsId:
            tm, tm_file = self.tm_set[index]
            am, am_file = self.am_set[index]
            assert filename2tapsID(tm_file) == filename2tapsID(am_file), f"File mismatch: {tm_file} vs {am_file}"
            tapsId = filename2tapsID(tm_file)
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
        if self.tapsId:
            return tm, am, tapsId
        else:
            return tm, am

class Audioset:
    def __init__(self, files=None, segment=None, stride=None, with_path=False):
        # Store the file list and hyperparameters
        self.files = files
        self.num_examples = []
        self.segment = segment
        self.stride = stride or segment
        self.with_path = with_path
        
        # Calculate how many segments (examples) each file can produce
        for _, file_length in self.files:
            # If no fixed segment length is provided or the file is shorter, only 1 example
            if segment is None or file_length < segment:
                examples = 1
            else:
                # Otherwise, calculate how many segments fit given stride
                examples = int(math.ceil((file_length - self.segment) / (self.stride)) + 1)
            self.num_examples.append(examples)

    def __len__(self):
        # The total length is the sum of all examples across files
        return sum(self.num_examples)

    def __getitem__(self, index):
        # Iterate through files and find which file/segment corresponds to 'index'
        for (file, _), examples in zip(self.files, self.num_examples):
            # If index is larger than current file's examples, skip to the next file
            if index >= examples:
                index -= examples
                continue
            
            # Otherwise, compute the offset based on stride and index
            offset = self.stride * index if self.segment else 0
            # Decide how many frames to load (full file if segment is None)
            num_frames = self.segment if self.segment else -1
            
            # Load audio from the offset for num_frames
            wav, _ = torchaudio.load(file, frame_offset=offset, num_frames=num_frames)
            
            # If the loaded waveform is shorter than the segment length, pad it
            if self.segment:
                wav = F.pad(wav, (0, num_frames - wav.shape[-1]))
            
            # Return (wav, file) if with_path is True, otherwise just return wav
            return (wav, file) if self.with_path else wav

            

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
    
    if len(batch) == 1:
        tm, am, Id = batch[0]
        return tm.unsqueeze(0), am.unsqueeze(0), None, Id
    
    tm, am, Id = zip(*batch)
    
    tm = [inp.clone().detach().squeeze() for inp in tm]
    am = [inp.clone().detach().squeeze() for inp in am]
           
    padded_tm = pad_sequence(tm, batch_first=True, padding_value=0.0).unsqueeze(1)
    padded_am = pad_sequence(am, batch_first=True, padding_value=0.0).unsqueeze(1)
    
    mask = torch.zeros(padded_tm.shape, dtype=torch.float32)
    for i, length in enumerate([inp.size(0) for inp in tm]):
        mask[i, :, :length] = 1
    
    return padded_tm, padded_am, mask, Id

class StepSampler(torch.utils.data.Sampler):
    def __init__(self, length, step):
        self.step = step
        self.length = length
        
    def __iter__(self):
        return iter(range(0, self.length, self.step))
    
    def __len__(self):
        return self.length // self.step
