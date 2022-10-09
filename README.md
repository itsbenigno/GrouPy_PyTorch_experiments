# Experiments on GrouPy using PyTorch

Convolutional neural networks have proven to be very powerful models of sensory data. 
A large amount of empirical evidence supports the notion that convolutional weight sharing is important for good predictive performance.
Convolutional weight sharing is effective because there is a translation symmetry in most perception tasks.
Although convolutions are equivariant to translation, they are not equivariant to reflection or rotation.
It may therefore be useful to encode a form of rotational symmetry in the architecture of a neural network.
This could reduce the redundancy of learning to detect the same patterns in different orientations (freeing up model capacity) and the need for extensive data augmentation.
Cohen’s paper shows how convolutional networks can be generalized to exploit larger groups of symmetries, including rotations and reflections.

The goal of this project is to reproduce (using Pytorch) the original experiments made by Cohen, 
highlighting the strengths and investigating the shortcomings of G-CNN, 
and also to propose a new approach that may help to cope with G-CNN weakness.
