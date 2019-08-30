"""
Minimal working example.

0) Define directories and parameters.
1) Tune segmenting parameters.
2) Segment.
	2.5) Clean segmenting decisions.
3) Tune preprocessing parameters.
4) Preprocess.
5) Train a generative model on these syllables.
6) Plot and analyze.

"""
__author__ = "Jack Goffinet"
__date__ = "December 2018 - August 2019"


from itertools import repeat
from joblib import Parallel, delayed
import numpy as np
import os

from ava.data.data_container import DataContainer
from ava.models.vae import X_SHAPE
from ava.models.vae import VAE
from ava.models.window_vae import VAE as WindowVAE
from ava.models.vae_dataset import get_syllable_partition, \
	get_syllable_data_loaders
from ava.model.window_vae_dataset import get_warped_window_data_loaders
from ava.preprocessing.preprocessing import get_spec, process_sylls, \
	tune_preprocessing_params
from ava.segmenting.refine_segments import refine_segments_pre
from ava.segmenting.segmenting import tune_segmenting_params, segment
from ava.segmenting.amplitude_segmentation_v2 import get_onsets_offsets


#########################################
# 0) Define directories and parameters. #
#########################################
mouse_params = {
	'get_spec': get_spec,
	'num_freq_bins': X_SHAPE[0],
	'num_time_bins': X_SHAPE[1],
	'delimiter': ',',
	'skiprows': 1,
	'usecols': (1,2),
	'segment': {
		'max_dur': 0.2,
		'min_freq': 30e3,
		'max_freq': 110e3,
		'nperseg': 1024, # FFT
		'noverlap': 0, # FFT
		'spec_min_val': 2.0,
		'spec_max_val': 6.0,
		'fs': 250000,
		'th_1':1.5,
		'th_2':2.0,
		'th_3':2.5,
		'min_dur':0.03,
		'max_dur':0.2,
		'freq_smoothing': 3.0,
		'smoothing_timescale': 0.007,
		'softmax': False,
		'temperature':0.5,
		'algorithm': get_onsets_offsets,
		'seg_extension': '.txt',
	},
	'preprocess': {
		'sliding_window': False,
		'max_dur': 0.2,
		'min_freq': 30e3,
		'max_freq': 110e3,
		'num_freq_bins': X_SHAPE[0],
		'num_time_bins': X_SHAPE[1],
		'nperseg': 1024, # FFT
		'noverlap': 512, # FFT
		'spec_min_val': 2.0,
		'spec_max_val': 6.0,
		'fs': 250000,
		'mel': False, # Frequency spacing
		'MAD': True,
		'time_stretch': True,
		'within_syll_normalize': False,
		'seg_extension': '.txt',
		'delimiter': '\t',
		'skiprows': 0,
		'usecols': (1,2),
		'max_num_syllables': None, # per directory
		'sylls_per_file': 20,
		'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
				'spec_max_val', 'max_dur'),
		'int_preprocess_params': ('nperseg',),
		'binary_preprocess_params': ('time_stretch', 'mel', 'within_syll_normalize', 'MAD'),
	},
}


zebra_finch_params_sliding_window = {
	'sliding_window': True,
	'window_length': 0.08,
	'get_spec': get_spec,
	'num_freq_bins': X_SHAPE[0],
	'num_time_bins': X_SHAPE[1],
	'min_freq': 400,
	'max_freq': 10e3,
	'nperseg': 512, # FFT
	'noverlap': 256, # FFT
	'mel': True, # Frequency spacing
	'spec_min_val': 2.0,
	'spec_max_val': 6.5,
	'time_stretch': False,
	'within_syll_normalize': False,
	'seg_extension': '.txt',
	'delimiter': '\t',
	'skiprows': 0,
	'usecols': (0,1),
	'max_dur': 1e9, # Big number
	'max_num_syllables': None, # per directory
	# 'sylls_per_file': 20,
	'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
			'spec_max_val', 'max_dur'),
	'int_preprocess_params': ('nperseg',),
	'binary_preprocess_params': ('time_stretch', 'mel', 'within_syll_normalize')
}


zebra_finch_params = {
	'sliding_window': False,
	'get_spec': get_spec,
	'num_freq_bins': X_SHAPE[0],
	'num_time_bins': X_SHAPE[1],
	'min_freq': 400,
	'max_freq': 10e3,
	'nperseg': 512, # FFT
	'noverlap': 256, # FFT
	'mel': True, # Frequency spacing
	'spec_min_val': 2.0,
	'spec_max_val': 6.5,
	'time_stretch': True,
	'within_syll_normalize': False,
	'seg_extension': '.txt',
	'delimiter': ' ',
	'skiprows': 0,
	'usecols': (0,1),
	'max_dur': 0.2,
	'max_num_syllables': None, # per directory
	'sylls_per_file': 5,
	'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
			'spec_max_val', 'max_dur'),
	'int_preprocess_params': ('nperseg',),
	'binary_preprocess_params': ('time_stretch', 'mel', 'within_syll_normalize')
}

# root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/'
# params = zebra_finch_params_sliding_window
# audio_dirs = [os.path.join(root, i) for i in ['songs/DIR', 'songs/UNDIR']]
# template_dir = root + 'templates'
# spec_dirs = [root+'h5s']
# proj_dirs = [root+'song_window/proj/']
# model_filename = root + 'song_window/checkpoint_201.tar'
# plots_dir = root + 'song_window/plots/'
# feature_dirs = None
# seg_dirs = None

root = '/media/jackg/Jacks_Animal_Sounds/mice/MUPET/'
audio_dirs = [root + i for i in ['C57', 'DBA']]
seg_dirs = [root + i for i in ['temp_seg_C57', 'temp_seg_DBA']]
proj_dirs = [root + i for i in ['temp_proj_C57', 'temp_proj_DBA']]
new_seg_dirs = [root + i for i in ['clean_seg_C57', 'clean_seg_DBA']]
spec_dirs = [root + i for i in ['temp_C57_spec', 'temp_DBA_spec']]
params = mouse_params
model_filename = root + 'checkpoint_020.tar'

dc = DataContainer(projection_dirs=proj_dirs, \
	spec_dirs=spec_dirs, plots_dir=root, model_filename=model_filename)

"""
##################################
# 1) Tune segmenting parameters. #
##################################
seg_params = tune_segmenting_params(audio_dirs, params['seg_params'])
params['seg_params'] = seg_params
"""

"""
###############
# 2) Segment. #
###############
n_jobs = min(len(audio_dirs), os.cpu_count()-1)
gen = zip(audio_dirs, seg_dirs, repeat(params['seg_params']))
Parallel(n_jobs=n_jobs)(delayed(segment)(*args) for args in gen)
"""

"""
####################################
# 2.5) Clean segmenting decisions. #
####################################
refine_segments(seg_dirs, audio_dirs, new_seg_dirs, params['segment'])
quit()
"""

"""
#####################################
# 3) Tune preprocessing parameters. #
#####################################
preprocess_params = tune_preprocessing_params(audio_dirs, new_seg_dirs, \
		params['preprocess'])
params['preprocess'] = preprocess_params
quit()
"""

"""
##################
# 4) Preprocess. #
##################
n_jobs = min(3, os.cpu_count()-1)
gen = zip(audio_dirs, new_seg_dirs, spec_dirs, repeat(params['preprocess']))
Parallel(n_jobs=n_jobs)(delayed(process_sylls)(*args) for args in gen)
quit()
"""

"""
###################################################
# 5) Train a generative model on these syllables. #
###################################################
model = VAE(save_dir=root)
partition = get_syllable_partition(spec_dirs, split=1)
num_workers = min(7, os.cpu_count()-1)
loaders = get_syllable_data_loaders(partition, num_workers=num_workers)
loaders['test'] = loaders['train']
model.train_loop(loaders, epochs=201, test_freq=None)
quit()
"""

########################
# 6) Plot and analyze. #
########################
from ava.plotting.tooltip_plot import tooltip_plot_DC
from ava.plotting.latent_projection import latent_projection_plot_DC

latent_projection_plot_DC(dc)
tooltip_plot_DC(dc, num_imgs=2000)




if __name__ == '__main__':
	pass


###