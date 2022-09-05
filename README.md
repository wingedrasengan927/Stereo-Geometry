A comprehensive tutorial on Stereo Geometry and Stereo Rectification with Examples.
## Setting up
Assuming you've anaconda installed, create a virtual environment and install dependencies. 

### Create Virtual Environment
```
conda create -n stereo-geometry python=3.6 anaconda
conda activate stereo-geometry
```
### Clone and Install dependencies
```
git clone https://github.com/wingedrasengan927/Stereo-Geometry.git
cd Stereo-Geometry
pip install -r requirements.txt
```
There are two main libraries we'll be using:
<br>[**pytransform3d**](https://github.com/rock-learning/pytransform3d): This library has great functions for visualizations and transformations in the 3D space.
<br>[**ipympl**](https://github.com/matplotlib/ipympl): It makes the matplotlib plot interactive allowing us to perform pan, zoom, and rotation in real time within the notebook which is really helpful when working with 3D plots.

**Note:** If you're using Jupyter Lab, please install the Jupyter Lab extension of ipympl from [here](https://github.com/matplotlib/ipympl)

ipympl can be accessed in the notebook by including the magic command `%matplotlib widget`

### Article
The code follows [**this article**](https://medium.com/p/7f368b09924a)
