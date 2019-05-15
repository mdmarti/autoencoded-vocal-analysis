"""Dataset for birdsongs."""

__author__ = "Jack Goffinet"
__date__ = "April 2019"


from os import listdir, sep
from os.path import join
import random

import h5py

import numpy as np
from scipy.signal import stft
from scipy.interpolate import interp1d
from skimage.transform import resize

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


EPSILON = 1e-9


def get_partition(dirs, split):
	"""

	"""
	assert split > 0.0 and split <= 1.0, "Invalid split: "+str(split)
	filenames = []
	for dir in dirs:
		filenames += [join(dir, i) for i in listdir(dir) if i[-5:] == '.hdf5']
	np.random.seed(42)
	np.random.shuffle(filenames)
	np.random.seed(None)
	index = int(round(split * len(filenames)))
	return {'train': filenames[:index], 'test': filenames[index:]}


def get_spec(audio, f_ind, start_frame, stop_frame, p, fs):
	"""
	Get a spectrogram.
	"""
	f, t, spec = stft(audio[start_frame:stop_frame], fs=fs)
	spec = np.log(np.abs(spec) + EPSILON)
	spec -= p['spec_thresh']
	spec[spec < 0.0] = 0.0
	# Switch to mel frequency spacing.
	if p['mel']:
		new_f = np.linspace(mel(p['min_freq']), mel(p['max_freq']), p['num_freq_bins'], endpoint=True)
		new_f = inv_mel(new_f)
		new_f[0] = f[0] # Correct for numerical errors.
		new_f[-1] = f[-1]
	else:
		new_f = np.linspace(f[0], f[-1], p['num_freq_bins'], endpoint=True)
	new_spec = np.zeros((p['num_freq_bins'], spec.shape[1]), dtype='float')
	for j in range(spec.shape[1]):
		interp = interp1d(f, spec[:,j], kind='cubic')
		new_spec[:,j] = interp(new_f)
	new_spec = resize(new_spec, (p['num_freq_bins'], p['num_time_bins']), anti_aliasing=True, mode='reflect')
	spec = new_spec
	f = new_f
	# Normalize.
	spec *= 0.9 / (np.max(spec) + EPSILON)
	spec += 0.05
	return spec, f_ind, t[1] - t[0]


def get_data_loaders(partition, params, batch_size=64, num_time_bins=128, \
			shuffle=(True, False)):
	songs_per_file = params['songs_per_file']
	train_dataset = SongDataset(filenames=partition['train'], \
			params=params,
			transform=ToTensor(),
			songs_per_file=songs_per_file)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, \
			shuffle=shuffle[0], num_workers=3)
	if not partition['test']:
		return train_dataloader, None
	test_dataset = SongDataset(filenames=partition['test'], \
			params=params,
			transform=ToTensor(),
			songs_per_file=songs_per_file)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, \
			shuffle=shuffle[1], num_workers=3)
	return train_dataloader, test_dataloader



class SongDataset(Dataset):
	"""Dataset for birdsongs"""

	def __init__(self, filenames, params, songs_per_file=1000, transform=None):
		self.filenames = filenames
		self.p = params
		self.songs_per_file = songs_per_file
		self.transform = transform


	def __len__(self):
		return len(self.filenames) * self.songs_per_file


	def __getitem__(self, index, start_frame=None, start_time=None):
		result = []
		single_index = False
		try:
			iterator = iter(index)
			if start_frame is None:
				start_frame = [None] * len(index)
			if start_time is None:
				start_time = [None] * len(index)
		except TypeError:
			index = [index]
			start_frame = [start_frame]
			start_time = [start_time]
			single_index = True
		for j, i in enumerate(index):
			# First find the file.
			load_filename = self.filenames[i // self.songs_per_file]
			file_index = i % self.songs_per_file
			# Then collect fields from the file.
			with h5py.File(load_filename, 'r') as f:
				sample = {
					'audio': f['audio'][file_index],
					'time': f['time'][file_index],
					'file_time': f['file_time'][file_index],
					'filename': str(f['filename'][file_index]),
					'fs' : f['fs'][file_index]
				}
				audio_frames =  int(sample['fs'] * self.p['spec_dur'])
				if start_frame[j] is None and start_time[j] is None:
					start_frame[j] = random.randint(0, len(sample['audio']) - audio_frames)
				elif start_time[j] is not None:
					start_frame[j] = int(round(sample['fs'] * start_time[j]))
				spec, _, _ = get_spec(sample['audio'], None, start_frame[j], start_frame[j]+audio_frames, self.p, sample['fs'])
				sample['spec'] = spec
				del sample['audio']
			if self.transform:
				sample = self.transform(sample)
			result.append(sample)
		if single_index:
			return result[0]
		return result



class ToTensor(object):

	def __call__(self, sample):
		spec = sample['spec']
		spec = torch.from_numpy(spec).type(torch.FloatTensor)
		sample['spec'] = spec
		return sample



def mel(a):
	return 1127 * np.log(1 + a / 700)


def inv_mel(a):
	return 700 * (np.exp(a / 1127) - 1)



if __name__ == '__main__':
	pass


###
