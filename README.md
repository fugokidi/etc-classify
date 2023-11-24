# Privacy-Preserving Image Classification Using an Isotropic Network

This repository is the official implementation of [Privacy-Preserving Image Classification Using an Isotropic Network](https://ieeexplore.ieee.org/document/9760030). 


The main idea of this paper is to show EtC encrypted images can be learned by
modern networks like ViT.

* EtC classification on ConvMixer is based on [ConvMixer](https://github.com/locuslab/convmixer).
* EtC classification on ViT is based on [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch).

Please follow the above repositories for setup.

Applying EtC on each image batch is

```python
from encryption import EtC
etc = EtC(14, 16) # the image size is 224 and EtC uses block size 16 => 16 * 14 = 224

# then in your training loop
for X, y in dataloader:
	X = etc(X)
```
