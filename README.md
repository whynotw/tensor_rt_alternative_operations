# TensorRT alternative operations
## Alternative ways to create tensorRT unsupported operations

**TensorRT** is a deep learning framework for accelerating inference. But many operations are not supported, including **upsampling** and **concatenating**. In this repo, alternative ways to create unsupported operations (**upsampling** and **concatenating**) are implemented with **pytorch**. You can translate the codes to **tensorRT** for your deep learning model.

### Prerequisites
```
numpy
pytorch==0.5.0
```

---

### 1. Upsampling

Use `python upsample.py`

```
input shape is (1, 2, 3, 3)
[[[[ 0.  1.  2.]
   [ 3.  4.  5.]
   [ 6.  7.  8.]]

  [[ 9. 10. 11.]
   [12. 13. 14.]
   [15. 16. 17.]]]]
```
Two ways to upsample the input matrix:
1. `torch.nn.functional.interpolate` **(unsupported by tensorRT)**
2. `torch.nn.ConvTranspose2d` with proper wieght initialization **(supported by tensorRT)**

The results are the same as follows.
```
output shape is (1, 2, 6, 6)
[[[[ 0.  0.  1.  1.  2.  2.]
   [ 0.  0.  1.  1.  2.  2.]
   [ 3.  3.  4.  4.  5.  5.]
   [ 3.  3.  4.  4.  5.  5.]
   [ 6.  6.  7.  7.  8.  8.]
   [ 6.  6.  7.  7.  8.  8.]]

  [[ 9.  9. 10. 10. 11. 11.]
   [ 9.  9. 10. 10. 11. 11.]
   [12. 12. 13. 13. 14. 14.]
   [12. 12. 13. 13. 14. 14.]
   [15. 15. 16. 16. 17. 17.]
   [15. 15. 16. 16. 17. 17.]]]]
```

---

### 2. Concatenating

Use `python concatenate.py`

```
inputs shape are (1, 2, 3, 3)
[[[[ 0.  1.  2.]
   [ 3.  4.  5.]
   [ 6.  7.  8.]]

  [[ 9. 10. 11.]
   [12. 13. 14.]
   [15. 16. 17.]]]]
   
[[[[  0.  -1.  -2.]
   [ -3.  -4.  -5.]
   [ -6.  -7.  -8.]]

  [[ -9. -10. -11.]
   [-12. -13. -14.]
   [-15. -16. -17.]]]]
```

Two ways to concatenate the input matrices:
1. `torch.cat` **(unsupported by tensorRT)**
2. `torch.nn.Conv2d` with proper wieght initialization **(supported by tensorRT)**

The results are the same as follows.

```
output shape is (1, 4, 3, 3)
[[[[  0.   1.   2.]
   [  3.   4.   5.]
   [  6.   7.   8.]]

  [[  9.  10.  11.]
   [ 12.  13.  14.]
   [ 15.  16.  17.]]

  [[  0.  -1.  -2.]
   [ -3.  -4.  -5.]
   [ -6.  -7.  -8.]]

  [[ -9. -10. -11.]
   [-12. -13. -14.]
   [-15. -16. -17.]]]]
```
