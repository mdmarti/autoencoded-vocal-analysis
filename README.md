## Autoencoded Vocal Analysis
#### Generative modeling of animal vocalizations
Current version: 0.2.1

See our [preprint](https://doi.org/10.1101/811661) on bioRxiv for details.

See `examples/` for usage.

To build package:
```
$ cd path/to/autoencoded-vocal-analysis
$ python setup.py sdist bdist_wheel
```

To build docs:
```
$ cd path/to/autoencoded-vocal-analysis/docs
$ make html
$ open build/html/index.html
```

Dependencies:
* Python 3
* [PyTorch](https://pytorch.org)
* [Joblib](https://joblib.readthedocs.io/)
* [UMAP](https://umap-learn.readthedocs.io/)
* [affinewarp](https://github.com/ahwillia/affinewarp)
* [Bokeh](https://docs.bokeh.org/en/latest/)
* [Sphinx read-the-docs theme](https://sphinx-rtd-theme.readthedocs.io/en/stable/)

Issues and/or pull requests are appreciated!

See also:
* [Animal Vocalization Generative Network](https://github.com/timsainb/AVGN), a
	nice repo by Tim Sainburg for clustering birdsong syllables and generating
	syllable interpolations.
* [DeepSqueak](https://github.com/DrCoffey/DeepSqueak) and
	[MUPET](https://github.com/mvansegbroeck/mupet), MATLAB packages for
	detecting and classifying rodent ultrasonic vocalizations.
