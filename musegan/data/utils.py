"""Midi dataset."""

from typing import Tuple
from torch import Tensor
import torch
from torch.utils.data import Dataset
import numpy as np
from music21 import midi, converter, note, stream, duration, tempo

class LPDDataset(Dataset):
    """LPDDataset. Multitrack

    Parameters
    ----------
    path: str
        Path to dataset.
    """

    def __init__(self, path: str) -> None:
        """Initialize."""
        dataset = np.load(path, allow_pickle=True, encoding="bytes")
        self.data_binary = dataset["arr_0"]

    def __len__(self) -> int:
        """Return the number of samples in dataset."""
        return len(self.data_binary)

    def __getitem__(self, index: int) -> Tensor:
        """Return one sample from dataset.

        Parameters
        ----------
        index: int
            Index of sample.

        Returns
        -------
        Tensor:
            Sample.
        """
        return torch.from_numpy(self.data_binary[index]).float()


class MidiDataset(Dataset):
    """MidiDataset. One track

    Parameters
    ----------
    path: str
        Path to dataset.
    split: str, optional (default="train")
        Split of dataset.
    n_bars: int, optional (default=2)
        Number of bars.
    n_steps_per_bar: int, optional (default=16)
        Number of steps per bar.
    """

    def __init__(self, path: str, split: str = "train", n_bars: int = 2, n_steps_per_bar: int = 16) -> None:
        """Initialize."""
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        dataset = np.load(path, allow_pickle=True, encoding="bytes")
        
        # Check if split key exists
        if split not in dataset:
            raise KeyError(f"{split} is not a key in the dataset.")
        
        dataset = dataset[split]
        self.data_binary, self.data_ints, self.data = self.__preprocess__(dataset)

    def __len__(self) -> int:
        """Return the number of samples in dataset."""
        return len(self.data_binary)

    def __getitem__(self, index: int) -> Tensor:
        """Return one sample from dataset.

        Parameters
        ----------
        index: int
            Index of sample.

        Returns
        -------
        Tensor:
            Sample.
        """
        return torch.from_numpy(self.data_binary[index]).float()

    def __preprocess__(self, data: np.ndarray) -> Tuple[np.ndarray]:
        """Preprocess data.

        Parameters
        ----------
        data: np.ndarray
            Data.

        Returns
        -------
        Tuple[np.ndarray]:
            Data binary, data ints, preprocessed data.
        """
        data_ints = []
        for x in data:
            # Skip rows with NaN values
            skip_rows = np.where(np.isnan(x).any(axis=1))[0]
            if skip_rows.size > 0:
                first_valid = skip_rows[0]
                x = x[first_valid:]  # Truncate the data
            
            if self.n_bars * self.n_steps_per_bar < x.shape[0]:
                data_ints.append(x[:self.n_bars * self.n_steps_per_bar])
        
        data_ints = np.array(data_ints)
        self.n_songs = data_ints.shape[0]
        self.n_tracks = data_ints.shape[2]
        data_ints = data_ints.reshape([self.n_songs, self.n_bars, self.n_steps_per_bar, self.n_tracks])
        
        max_note = 83
        mask = np.isnan(data_ints)
        data_ints[mask] = max_note + 1
        data_ints = data_ints.astype(int)
        
        num_classes = max_note + 2  # max_note + 1 and adding 1 for the mask
        data_binary = np.eye(num_classes)[data_ints]
        data_binary[data_binary == 0] = -1
        data_binary = np.delete(data_binary, max_note + 1, axis=-1)  # Removing the last class
        data_binary = data_binary.transpose([0, 3, 1, 2])  # [songs, classes, bars, steps]

        return data_binary, data_ints, data


def binarise_output(output: np.ndarray) -> np.ndarray:
    """Binarize output.

    Parameters
    ----------
    output: np.ndarray
        Output array.
    """
    max_pitches = np.argmax(output, axis=-1)
    return max_pitches


def postprocess(output: np.ndarray, n_tracks: int = 4, n_bars: int = 2, n_steps_per_bar: int = 16) -> stream.Score:
    """Postprocess output.

    Parameters
    ----------
    output: np.ndarray
        Output array.
    n_tracks: int, (default=4)
        Number of tracks.
    n_bars: int, (default=2)
        Number of bars.
    n_steps_per_bar: int, (default=16)
        Number of steps per bar.
    """
    parts = stream.Score()
    parts.append(tempo.MetronomeMark(number=66))
    max_pitches = binarise_output(output)
    
    midi_note_score = np.vstack([max_pitches[i].reshape([n_bars * n_steps_per_bar, n_tracks]) for i in range(len(output))])
    
    for i in range(n_tracks):
        last_x = int(midi_note_score[:, i][0])
        s = stream.Part()
        dur = 0
        
        for idx, x in enumerate(midi_note_score[:, i]):
            x = int(x)
            if (x != last_x or idx % 4 == 0) and idx > 0:
                n = note.Note(last_x)
                n.duration = duration.Duration(dur)
                s.append(n)
                dur = 0
            last_x = x
            dur += 0.25  # Assuming each step is a quarter note
        
        # Handle the last note
        if last_x != -1:  # Only append if last_x is valid
            n = note.Note(last_x)
            n.duration = duration.Duration(dur)
            s.append(n)
        
        parts.append(s)
    
    return parts
