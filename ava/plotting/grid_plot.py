"""
Plot a grid of images.

"""
__author__ = "Jack Goffinet"
__date__ = "July-August 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os



def indexed_grid_plot(dc, indices, ax=None, save_and_close=True, \
	filename='grid.pdf'):
	"""
	TO DO: use this to access grid_plot.
	"""
	specs = dc.request('specs')
	result = []
	for row in indices:
		result.append([specs[j] for j in row])
	grid_plot(np.array(result), os.path.join(dc.plots_dir, filename), ax=ax, \
		save_and_close=save_and_close)


def grid_plot(specs, filename, gap=3, ax=None, save_and_close=True):
	"""
	Parameters
	----------
	specs : numpy.ndarray
		...

	filename : str
		Save the image here.
	"""
	if type(gap) == type(4):
		gap = (gap,gap)
	a, b, c, d = specs.shape
	dx, dy = d+gap[1], c+gap[0]
	height = a*c + (a-1)*gap[0]
	width = b*d + (b-1)*gap[1]
	img = np.zeros((height, width))
	for j in range(a):
		for i in range(b):
			img[j*dy:j*dy+c,i*dx:i*dx+d] = specs[-j-1,i]
	for i in range(1,b):
		img[:,i*dx-gap[1]:i*dx] = np.nan
	for j in range(1,a):
		img[j*dy-gap[0]:j*dy,:] = np.nan
	if ax is None:
		ax = plt.gca()
	ax.imshow(img, aspect='equal', origin='lower', interpolation='none',
		vmin=0, vmax=1)
	ax.axis('off')
	if save_and_close:
		plt.tight_layout()
		plt.savefig(filename)
		plt.close('all')



if __name__ == '__main__':
	pass


###