"""
Useful functions for segmenting.

"""
__date__ = "August 2019 - July 2020"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from scipy.signal import stft
from scipy.io import wavfile


EPSILON = 1e-9



def get_spec(audio, p):
	"""
	Get a spectrogram.

	Much simpler than ``ava.preprocessing.utils.get_spec``

	Raises
	------
	- ``AssertionError`` if ``len(audio) < p['nperseg']``.

	Parameters
	----------
	audio : numpy array of floats
		Audio
	p : dict
		Spectrogram parameters. Should the following keys: `'fs'`, `'nperseg'`,
		`'noverlap'`, `'min_freq'`, `'max_freq'`, `'spec_min_val'`,
		`'spec_max_val'`

	Returns
	-------
	spec : numpy array of floats
		Spectrogram of shape [freq_bins x time_bins]
	dt : float
		Time step between time bins.
	f : numpy.ndarray
		Array of frequencies.
	"""
	assert len(audio) >= p['nperseg'], \
			"len(audio): " + str(len(audio)) + ", nperseg: " + str(p['nperseg'])
	f, t, spec = stft(audio, fs=p['fs'], nperseg=p['nperseg'], \
		noverlap=p['noverlap'])
	i1 = np.searchsorted(f, p['min_freq'])
	i2 = np.searchsorted(f, p['max_freq'])
	f, spec = f[i1:i2], spec[i1:i2]
	spec = np.log(np.abs(spec) + EPSILON)
	spec -= p['spec_min_val']
	spec /= p['spec_max_val'] - p['spec_min_val']
	spec = np.clip(spec, 0.0, 1.0)
	return spec, t[1]-t[0], f


def clean_segments_by_hand(audio_dirs, orig_seg_dirs, new_seg_dirs, p, \
	nrows=4, ncols=4, shoulder=0.1, img_filename='temp.pdf'):
	"""
	Plot spectrograms and ask for accept/reject input.

	The accepted segments are taken from `orig_seg_dirs` and copied to
	`new_seg_dirs`.

	Note
	----
	* This will not overwrite existing segmentation files.

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	orig_seg_dirs : list of str
		Original segment directories.
	new_seg_dirs : list of str
		New segment directories.
	p : dict
		Parameters. Should the following keys: `'fs'`, `'nperseg'`,
		`'noverlap'`, `'min_freq'`, `'max_freq'`, `'spec_min_val'`,
		`'spec_max_val'`
	nrows : int, optional
		Number of rows of spectrograms to plot. Defaults to ``4``.
	ncols : int, optional
		Number of columns of spectrograms to plot. Defaults to ``4``.
	shoulder : float, optional
		Duration of audio to plot on either side of segment. Defaults to `0.1`.
	img_filename : str, optional
		Where to write images. Defaults to ``'temp.pdf'``.
	"""
	# Make new directories, if needed.
	for new_seg_dir in new_seg_dirs:
		if not os.path.exists(new_seg_dir):
			os.makedirs(new_seg_dir)
	# Collect all the filenames.
	audio_fns, orig_seg_fns = get_audio_seg_filenames(audio_dirs, orig_seg_dirs)
	temp_dict = dict(zip(orig_seg_dirs, new_seg_dirs))
	new_seg_fns = []
	for orig_seg_fn in orig_seg_fns:
		a,b = os.path.split(orig_seg_fn)
		new_seg_fns.append(os.path.join(temp_dict[a], b))
	for new_seg_fn in new_seg_fns:
		assert not os.path.isfile(new_seg_fn), "File already exists: " + \
				new_seg_fn
	# Collect all of the segments.
	all_onsets, all_offsets = [], []
	all_audio_fns, all_orig_seg_fns, all_new_seg_fns = [], [], []
	gen = zip(audio_fns, orig_seg_fns, new_seg_fns)
	for audio_fn, orig_seg_fn, new_seg_fn in gen:
		segs = np.loadtxt(orig_seg_fn).reshape(-1,2)
		header = "Onsets/offsets cleaned by hand from " + orig_seg_fn
		np.savetxt(new_seg_fn, np.array([]), header=header)
		onsets, offsets = segs[:,0], segs[:,1]
		all_onsets += onsets.tolist()
		all_offsets += offsets.tolist()
		all_audio_fns += [audio_fn]*len(segs)
		all_orig_seg_fns += [orig_seg_fn]*len(segs)
		all_new_seg_fns += [new_seg_fn]*len(segs)
	# Loop through the segments, asking for accept/reject descisions.
	index = 0
	while index < len(all_onsets):
		print("orig_seg_fn:", all_orig_seg_fns[index])
		num_specs = min(len(all_onsets) - index, nrows*ncols)
		_, axarr = plt.subplots(nrows=nrows, ncols=ncols)
		axarr = axarr.flatten()
		# Plot spectrograms.
		for i in range(num_specs):
			if i == 0 or all_audio_fns[index+i] != all_audio_fns[index+i-1]:
				audio_fn = all_audio_fns[index+i]
				orig_seg_fn = all_orig_seg_fns[index+i]
				new_seg_fn = all_new_seg_fns[index+i]
				# Get spectrogram.
				fs, audio = wavfile.read(audio_fn)
				assert fs == p['fs'], "Found fs="+str(fs)+", expected fs="+\
						str(p['fs'])
				spec, dt, f = get_spec(audio, p)
			onset, offset = all_onsets[index+i], all_offsets[index+i]
			i1 = max(0, int((onset - shoulder) / dt))
			i2 = min(spec.shape[1], int((offset + shoulder) / dt))
			t1 = max(0, onset-shoulder)
			t2 = min(len(audio)/fs, offset+shoulder)
			plt.sca(axarr[i])
			plt.imshow(spec[:,i1:i2], origin='lower', aspect='auto', \
					extent=[t1, t2, f[0]/1e3, f[-1]/1e3])
			plt.title(str(i))
			plt.axis('off')
			plt.axvline(x=onset, c='r')
			plt.axvline(x=offset, c='r')
		plt.tight_layout()
		plt.savefig(img_filename)
		plt.close('all')
		# Collect user input. NOTE: HERE
		response = 'invalid response'
		while not _is_valid_response(response, num_specs):
			response = input("[Good]? Or list bad spectrograms: ")
		if response == '':
			good_indices = [index+i for i in range(num_specs)]
		else:
			responses = [int(i) for i in response.split(' ')]
			for i in range(num_specs):
				if i not in responses:
					good_indices.append(index+i)
			good_indices = np.unique(np.array(good_indices, dtype='int')).tolist()
		# Copy the good segments.
		for i in range(num_specs):
			if index + i in good_indices:
				with open(all_new_seg_fns[index+i], 'ab') as f:
					seg = np.array([all_onsets[index+i], all_offsets[index+i]])
					np.savetxt(f, seg, fmt='%.5f')
		index += num_specs


		#
		# good_indices = []
		# i = 0
		# while i < len(onsets):
		# 	num_specs = min(len(onsets) - i, 16)
		# 	_, axarr = plt.subplots(nrows=4, ncols=4)
		# 	axarr = axarr.flatten()
		# 	for j in range(num_specs):
		# 		onset, offset = onsets[i+j], offsets[i+j]
		# 		i1 = max(0, int((onset - shoulder) / dt))
		# 		i2 = min(spec.shape[1], int((offset + shoulder) / dt))
		# 		t1 = max(0, onset-shoulder)
		# 		t2 = min(len(audio)/fs, offset+shoulder)
		# 		plt.sca(axarr[j])
		# 		plt.imshow(spec[:,i1:i2], origin='lower', aspect='auto', \
		# 				extent=[t1, t2, f[0]/1e3, f[-1]/1e3])
		# 		plt.title(str(j))
		# 		plt.axis('off')
		# 		# plt.xlabel('Time (s)')
		# 		plt.axvline(x=onset, c='r')
		# 		plt.axvline(x=offset, c='r')
		# 	plt.tight_layout()
		# 	plt.savefig(img_filename)
		# 	plt.close('all')
		# 	response = 'invalid response'
		# 	while not _is_valid_response(response, num_specs):
		# 		response = input("[Good]? Or list bad spectrograms: ")
		# 	if response == '':
		# 		good_indices += [i+j for j in range(num_specs)]
		# 	else:
		# 		responses = [int(k) for k in response.split(' ')]
		# 		for j in range(num_specs):
		# 			if j not in responses:
		# 				good_indices.append(i+j)
		# 	i += num_specs
		# good_indices = np.array(good_indices, dtype='int')
		# onsets, offsets = onsets[good_indices], offsets[good_indices]
		# combined = np.stack([onsets, offsets]).T
		# np.savetxt(new_seg_fn, combined, fmt='%.5f', header=header)


def copy_segments_to_standard_format(orig_seg_dirs, new_seg_dirs, seg_ext, \
	delimiter, usecols, skiprows, max_duration=None):
	"""
	Copy onsets/offsets from SAP, MUPET, or Deepsqueak into a standard format.

	Note
	----
	- `delimiter`, `usecols`, and `skiprows` are all passed to `numpy.loadtxt`.

	Parameters
	----------
	orig_seg_dirs : list of str
		Directories containing original segments.
	new_seg_dirs : list of str
		Directories for new segments.
	seg_ext : str
		Input filename extension.
	delimiter : str
		Input filename delimiter.
	usecols : tuple
		Input file onset and offset column.
	skiprows : int
		Number of rows to skip.
	max_duration : {None, float}, optional
		Maximum segment duration. If None, no max is set. Defaults to `None`.
	"""
	assert len(seg_ext) == 4
	for orig_seg_dir, new_seg_dir in zip(orig_seg_dirs, new_seg_dirs):
		if not os.path.exists(new_seg_dir):
			os.makedirs(new_seg_dir)
		seg_fns = [os.path.join(orig_seg_dir,i) for i in \
				os.listdir(orig_seg_dir) if len(i) > 4 and i[-4:] == seg_ext]
		for seg_fn in seg_fns:
			segs = np.loadtxt(seg_fn, delimiter=delimiter, skiprows=skiprows, \
					usecols=usecols).reshape(-1,2)
			if max_duration is not None:
				new_segs = []
				for seg in segs:
					if seg[1]-seg[0] < max_duration:
						new_segs.append(seg)
				if len(new_segs) > 0:
					segs = np.stack(new_segs)
				else:
					segs = np.array([])
			new_seg_fn = os.path.join(new_seg_dir, os.path.split(seg_fn)[-1])
			new_seg_fn = new_seg_fn[:-4] + '.txt'
			header = "Onsets/offsets copied from "+seg_fn
			np.savetxt(new_seg_fn, segs, fmt='%.5f', header=header)


def write_segments_to_audio(in_audio_dirs, out_audio_dirs, seg_dirs, n_zfill=3,\
	verbose=True):
	"""
	Write each segment as its own audio file.

	Parameters
	----------
	in_audio_dirs : list of str
		Where to read audio.
	out_audio_dirs : list of str
		Where to write audio.
	seg_dirs : list of str
		Where to read segments.
	n_zfill : int, optional
		For filename formatting. Defaults to ``3``.
	verbose : bool, optional
		Deafults to ``True``.
	"""
	if verbose:
		print("Writing segments to audio,", len(in_audio_dirs), "directories")
	for in_dir, out_dir, seg_dir in zip(in_audio_dirs, out_audio_dirs, seg_dirs):
		seg_fns = [j for j in sorted(os.listdir(seg_dir)) if _is_txt_file(j)]
		audio_fns = [os.path.join(in_dir, j[:-4]+'.wav') for j in seg_fns]
		seg_fns = [os.path.join(seg_dir, j) for j in seg_fns]
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		for seg_fn, audio_fn in zip(seg_fns, audio_fns):
			segs = np.loadtxt(seg_fn).reshape(-1,2)
			if len(segs) == 0:
				continue
			fs, audio = wavfile.read(audio_fn)
			for j in range(segs.shape[0]):
				num_samples = int(round(fs * (segs[j,1]-segs[j,0])))
				i1 = int(round(fs * segs[j,0]))
				out_audio = audio[i1:i1+num_samples]
				out_audio_fn = os.path.split(audio_fn)[-1][:-4]
				out_audio_fn += '_' + str(j).zfill(n_zfill) + '.wav'
				out_audio_fn = os.path.join(out_dir, out_audio_fn)
				wavfile.write(out_audio_fn, fs, out_audio)
	if verbose:
		print("\tDone.")


def get_audio_seg_filenames(audio_dirs, seg_dirs):
	"""
	Return lists of audio and corresponding segment filenames.

	Parameters
	----------
	audio_dirs : list of str
		Audio directories
	seg_dirs : list of str
		Corresponding segmenting directories

	Returns
	-------
	audio_fns : list of str
		Audio filenames
	seg_fns : list of str
		Corresponding segment filenames
	"""
	audio_fns, seg_fns = [], []
	for audio_dir, seg_dir in zip(audio_dirs, seg_dirs):
		temp_fns = [i for i in sorted(os.listdir(audio_dir)) if \
				_is_audio_file(i)]
		audio_fns += [os.path.join(audio_dir, i) for i in temp_fns]
		temp_fns = [i[:-4] + '.txt' for i in temp_fns]
		seg_fns += [os.path.join(seg_dir, i) for i in temp_fns]
	return audio_fns, seg_fns


def softmax(arr, t=0.5):
	"""Softmax along first array dimension."""
	temp = np.exp(arr/t)
	temp /= np.sum(temp, axis=0) + EPSILON
	return np.sum(np.multiply(arr, temp), axis=0)


def _read_onsets_offsets(filename):
	"""
	A wrapper around numpy.loadtxt for reading onsets & offsets.

	Parameters
	----------
	filename : str
		Filename of a text file containing one header line and two columns.

	Returns
	-------
	onsets : numpy.ndarray
		Onset times.
	offsets : numpy.ndarray
		Offset times.
	"""
	arr = np.loadtxt(filename, skiprows=1)
	if len(arr) == 0:
		return [], []
	if len(arr.shape) == 1:
		arr = arr.reshape(1,2)
	assert arr.shape[1] == 2, "Found invalid shape: "+str(arr.shape)
	return arr[:,0], arr[:,1]


def _is_audio_file(filename):
	return len(filename) > 4 and filename[-4:] == '.wav'


def _is_txt_file(filename):
	return len(filename) > 4 and filename[-4:] == '.txt'


def _is_valid_response(response, num_specs):
	if response == '':
		return True
	try:
		responses = [int(i) for i in response.split(' ')]
		return min(responses) >= 0 and max(responses) < num_specs
	except:
		return False



if __name__ == '__main__':
	pass


###
