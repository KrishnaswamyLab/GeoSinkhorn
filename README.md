# GeoSinkhorn
Code for the paper Geodesic Sinkhorn for Fast and Accurate Optimal Transport on Manifolds. 

> [!NOTE]  
>This repository is still in development.

### Installation
You can install the library from [PyPI](https://pypi.org/project/geosink/) by running:
```bash
pip install geosink
``` 
Or using Git, by first cloning the repository and running:
```bash
pip install -e .
```
If you want to use the pre existing graph tools, run:
```bash
pip install -e .['graph']
```
To run the tests, you will need additional packages. Install them by running:
```bash
pip install -e .['dev']
```


### Minimal Example
You can reproduce this example in the following notebook [![notebook](https://img.shields.io/static/v1?label=Run%20in&message=Google%20Colab&color=orange&logo=Google%20Cloud)](https://colab.research.google.com/drive/1Y_CHGb49aVXgTPtnD-Yf8GX_PYYwXwYx?usp=sharing).


We build a graph between two Gaussian distributions and compute the distance between two signals on that graph.
```python
import numpy as np
from geosink.sinkhorn import GeoSinkhorn 
from geosink.heat_kernel import laplacian_from_data

# Generate data and build graph.
data0 = np.random.normal(0, 1, (100, 5))
data1 = np.random.normal(5, 1, (100, 5))
data = np.concatenate([data0, data1], axis=0)
lap = laplacian_from_data(data, sigma=1.0)

# instantiate the GeoSinkhorn class
geo_sinkhorn = GeoSinkhorn(tau=5.0, order=10, method="cheb", lap=lap)

# create two signals
m_0 = np.zeros(200,)
m_0[:100] = 1
m_0 = m_0 / np.sum(m_0)
m_1 = np.zeros(200,)
m_1[100:] = 1
m_1 = m_1 / np.sum(m_1)

# compute the distance between the two signals
dist_w = geo_sinkhorn(m_0, m_1, max_iter=500)
print(dist_w)
```
Note that it is also possible to provide a graph instance directly to the `GeoSinkhorn` class with `GeoSinkhorn(tau=1.0, order=10, method="cheb", graph=graph)`. The `graph` must have a Laplacian attribute `graph.L`. We suggest using a sparse Laplacian (e.g. in COO format) for better performance.

### How to Cite

If you find this code useful in your research, please cite the following paper (expand for BibTeX):
<details>
<summary>
Huguet, G., Tong, A., Zapatero, M. R., Tape, C. J., Wolf, G., & Krishnaswamy, S. (2023). Geodesic Sinkhorn for fast and accurate optimal transport on manifolds. In 2023 IEEE 33rd International Workshop on Machine Learning for Signal Processing (MLSP).
</summary>

```bibtex
@inproceedings{huguet2023geodesic,
  title={Geodesic Sinkhorn for fast and accurate optimal transport on manifolds},
  author={Huguet, Guillaume and Tong, Alexander and Zapatero, Mar{\'\i}a Ramos and Tape, Christopher J and Wolf, Guy and Krishnaswamy, Smita},
  booktitle={2023 IEEE 33rd International Workshop on Machine Learning for Signal Processing (MLSP)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```
