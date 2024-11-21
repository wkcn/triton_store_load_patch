import triton_sl
import torch
import numpy as np

dtype = np.float16
ttype = torch.float16
a = torch.arange(100, 110).to(dtype=ttype)
p = a.data_ptr()
p = (p + np.arange(4) * 2).astype(np.uint64) 
mask = np.array([True, True, True, True], dtype=np.bool)
other = np.zeros(4, dtype=np.int64)

b = triton_sl.load(p, mask, other, np.dtype(dtype))
print(b)
