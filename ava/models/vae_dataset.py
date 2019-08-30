"""
Data stuff for animal vocalization syllables.

Contains
--------
- function get_syllable_partition
- function get_syllable_data_loaders
- class SyllableDataset
"""
__author__ = "Jack Goffinet"
__date__ = "November 2018 - August 2019"


import h5py
import joblib
import numpy as np
import os
from scipy.interpolate import interp1d, interp2d
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter
from scipy.signal import stft
import torch
from torch.utils.data import Dataset, DataLoader

EPSILON = 1e-12


def get_syllable_partition(dirs, split, shuffle=True):
	"""
	Partition the set filenames into a random test/train split.

	Parameters
	----------
	dirs : list of strings
		List of directories containing saved syllable hdf5 files.

	split : float
		Portion of the hdf5 files to use for training, 0 < split <= 1.0

	shuffle : bool, optional
		Whether to shuffle the hdf5 files. Defaults to True.

	Returns
	-------
	partition : dictionary
		Contains two keys, 'test' and 'train', that map to lists of .hdf5 files.
		Defines the random test/train split.
	"""
	assert(split > 0.0 and split <= 1.0)
	# Collect filenames.
	filenames = []
	for dir in dirs:
		filenames += get_hdf5s_from_dir(dir)
	# Reproducibly shuffle.
	filenames = sorted(filenames)
	if shuffle:
		np.random.seed(42)
		np.random.shuffle(filenames)
		np.random.seed(None)
	# Split.
	index = int(round(split * len(filenames)))
	return {'train': filenames[:index], 'test': filenames[index:]}


def get_syllable_data_loaders(partition, batch_size=64, shuffle=(True, False), \
	num_workers=4):
	"""
	Return a pair of DataLoaders given a test/train split.

	Parameters
	----------
	partition : dictionary
		Test train split: a dictionary that maps the keys 'test' and 'train'
		to disjoint lists of .hdf5 filenames containing syllables.

	batch_size : int, optional
		Batch size of the returned Dataloaders. Defaults to 32.

	shuffle : tuple of bools, optional
		Whether to shuffle data for the train and test Dataloaders,
		respectively. Defaults to (True, False).

	num_workers : int, optional
		How many subprocesses to use for data loading. Defaults to 3.

	Returns
	-------
	dataloaders : dictionary
		Dictionary mapping two keys, 'test' and 'train', to respective
		torch.utils.data.Dataloader objects.
	"""
	sylls_per_file = get_sylls_per_file(partition)
	train_dataset = SyllableDataset(filenames=partition['train'], \
		transform=numpy_to_tensor, sylls_per_file=sylls_per_file)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, \
		shuffle=shuffle[0], num_workers=num_workers)
	if not partition['test']:
		return {'train':train_dataloader, 'test':None}
	test_dataset = SyllableDataset(filenames=partition['test'], \
		transform=numpy_to_tensor, sylls_per_file=sylls_per_file)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, \
		shuffle=shuffle[1], num_workers=num_workers)
	return {'train':train_dataloader, 'test':test_dataloader}



class SyllableDataset(Dataset):
	"""torch.utils.data.Dataset for animal vocalization syllables"""

	def __init__(self, filenames, sylls_per_file, transform=None):
		"""
		Create a torch.utils.data.Dataset for animal vocalization syllables.

		Parameters
		----------
		filenames : list of strings
			List of hdf5 files containing syllable spectrograms.

		sylls_per_file : int
			Number of syllables in each hdf5 file.

		transform : None or function, optional
			Transformation to apply to each item. Defaults to None (no
			transformation)
		"""
		self.filenames = filenames
		self.sylls_per_file = sylls_per_file
		self.transform = transform

	def __len__(self):
		return len(self.filenames) * self.sylls_per_file

	def __getitem__(self, index):
		result = []
		single_index = False
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True
		for i in index:
			# First find the file.
			load_filename = self.filenames[i // self.sylls_per_file]
			file_index = i % self.sylls_per_file
			# Then collect fields from the file.
			with h5py.File(load_filename, 'r') as f:
				try:
					spec = f['specs'][file_index]
				except:
					print(file_index, self.sylls_per_file)
					print(i // self.sylls_per_file, len(self.filenames))
					print(len(f['specs']))
					print(load_filename)
					quit()
			if self.transform:
				spec = self.transform(spec)
			result.append(spec)
		if single_index:
			return result[0]
		return result



def get_sylls_per_file(partition):
	"""Open an hdf5 file and see how many syllables it has."""
	key = 'train' if len(partition['train']) > 0 else 'test'
	assert len(partition[key]) > 0
	filename = partition[key][0] # Just grab the first file.
	with h5py.File(filename, 'r') as f:
		sylls_per_file = len(f['specs'])
	return sylls_per_file


def numpy_to_tensor(x):
	"""Transform a numpy array into a torch.FloatTensor"""
	return torch.from_numpy(x).type(torch.FloatTensor)


def get_hdf5s_from_dir(dir):
	"""
	Return a sorted list of all hdf5s in a directory.

	Note
	----
	plotting.data_container relies on this.
	"""
	return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if \
		is_hdf5_file(f)]


def get_wavs_from_dir(dir):
	return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if \
		is_wav_file(f)]


def is_hdf5_file(filename):
	"""Is the given filename an hdf5 file?"""
	return len(filename) > 5 and filename[-5:] == '.hdf5'


def is_wav_file(filename):
	return len(filename) > 4 and filename[-4:] == '.wav'



if __name__ == '__main__':
	pass


###
