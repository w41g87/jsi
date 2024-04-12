# Info

```jsi.py``` files contain all necessary function, ```generate.ipynb``` provides some simple examples for generating heatmaps using the provided inputs; ```optimize.ipynb``` provides examples for approximating a target jsi using the gradient descent optimizer; ```nn.ipynb``` provides examples of dataset generation and training with neuro-network; ```misc.ipynb``` provides other relevant examples.

# Prerequisite

The following is deprecated, use [tensorflow](https://www.tensorflow.org/install), or [tensorflow with GPU](https://www.tensorflow.org/install/pip)

We recommend you to use tools like [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to create a virtual environment for tensorflow gpu installation.

<del>
<p>To enable CUDA acceleration, you will need an Nvidia GPU.

You will need the following packages to enable CUDA acceleration:

- CUDA Toolkit from [Nvidia](https://developer.nvidia.com/cuda-toolkit)
- cuda-python package from [here](https://nvidia.github.io/cuda-python/install.html)

For OpenCl implementation:

- OpenCl SDK for your graphics card (This is usually included in your driver installation)
- pyopencl package from [here](https://documen.tician.de/pyopencl/misc.html#installing-from-pypi-wheels)

To enable interactive matplotlib figures, install [ipympl](https://matplotlib.org/ipympl/)</p>
</del>

# Change Log

## v 4.2.1
- Expanded the complexity of the trained model

## v 4.2.0
- Custom loss class
- jsi calculation kernel object
- pretrained example model for mnist dataset

## v 4.1.1
- Exponential optimization lr decay as default
- Clipped optimization gradient to avoid blowups
### Bugfixes
- Import library corrections

## v 4.1.0
- NN integration

## v 4.0.1
- Integrated JSI generation
### Bugfixes
- Calibrated derivation against 1-node analytical result

## v 4.0.0
- Multi-ring optimization

## v 3.2.2
- Added checkpoint for segmented training
- Added CLI utilities to train/resume training and generate image to file
- Optimized CLI compatibility
    ### Bugfixes
      - Fixed wrong coupling length calculation when padding is considered

## v 3.1.2
- Adopted static graph in lieu of eager execution
- Code compartmentalization and speedups

## v 3.1.1
### Bugfixes
    - Fixed loss calculation blowing up
    - Fixed tensorflow model descrepancies

## v 3.1.0
- Approximation algorithm using Adam gradient descdent

## v 3.0.1
### Bugfixes
    - Fixed heatmap representation having rows flipped

## v 3.0.0
- Tensorflow integration in preperation for NN training
- Code speedup
- Code cleanup
- Removed CUDA and OpenCl methods

## v 2.2.5
- Added k-space calculation with defined period

## v 2.2.4
### Bugfixes
    - Fixed index error for k-space eigenvalue calculations

## v 2.2.3
- Added eigenvalue calculation for real and k-space
    ### Bugfixes
     - Fixed invalid warning method

## v 2.2.2
- Enhanced plot readability
    ### Bugfixes
     - Patched CPU computing
     - Warning messages now correctly display when optional packages are missing
     - Fixed inconsistent output of the CUDA accelerator due to unsynchronized host-to-device memory copying
     - Heatmap now correctly displays on large input nodes

## v 2.2.1
- Added nodes-dependent external coupling function
- Added phase plot to spot anomalies (phase should be either 0 or pi since the JSI should be real)

## v 2.2.0
- Added OpenCl acceleration to accomodate more graphical processors
- Added concurrency to CPU calculation
- All function declarations now reside in the ```jsi.py``` module
- Integrated and streamlined function calls
- Automatically falls back to slower modules if acceleration modules are not found

## v 2.1.0
- Optimized CUDA computation by removing divergent branch and instruction overhead

## v 2.0.0
- Added CUDA acceleration
- Streamlined input format, allowing arbitrary coupling lengths








