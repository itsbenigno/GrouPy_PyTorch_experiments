# * Input 32 × 32 RGB image
# * 3 × 3 conv. 96 ReLU
# * 3 × 3 conv. 96 ReLU
# * 3 × 3 max-pooling stride 2
# * 3 × 3 conv. 192 ReLU
# * 3 × 3 conv. 192 ReLU
# * 3 × 3 max-pooling stride 2
# * 3 × 3 conv. 192 ReLU
# * 1 × 1 conv. 192 ReLU
# * 1 × 1 conv. 10 ReLU
# * global averaging over 6 × 6 spatial dimensions
# * 10 (or 100)-way softmax
