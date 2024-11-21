#include <atomic>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <type_traits>

namespace py = pybind11;


PYBIND11_MODULE(triton_sl, m) {
  m.def("load",
        [](py::array_t<uint64_t> ptr, py::array_t<bool> mask, py::array other,
           py::dtype ret_dtype) -> py::array {
          int numel = ptr.size();
          auto shape =
              std::vector<ptrdiff_t>(ptr.shape(), ptr.shape() + ptr.ndim());
          py::array ret(ret_dtype, py::array::ShapeContainer{numel});
          py::array_t<uint64_t> reshaped_ptr = ptr.reshape({numel});
          py::array_t<bool> reshaped_mask = mask.reshape({numel});
          py::array reshaped_others = other.reshape({numel});
          for (size_t i = 0; i < ptr.size(); ++i) {
            if (reshaped_mask.at(i))
              memcpy(ret.mutable_data(i),
                     reinterpret_cast<void *>(reshaped_ptr.at(i)),
                     ret_dtype.itemsize());
            else
              memcpy(ret.mutable_data(i), reshaped_others.data(i),
                     ret_dtype.itemsize());
          }
          return ret.reshape(shape);
        });

  m.def("store",
        [](py::array_t<uint64_t> ptr, py::array value, py::array_t<bool> mask) {
          int numel = ptr.size();
          py::array_t<uint64_t> reshaped_ptr = ptr.reshape({numel});
          py::array_t<int8_t> reshaped_mask = mask.reshape({numel});
          py::array reshaped_value = value.reshape({numel});
          for (size_t i = 0; i < ptr.size(); ++i) {
            if (reshaped_mask.at(i)) {
              memcpy(reinterpret_cast<void *>(reshaped_ptr.mutable_at(i)),
                     reshaped_value.data(i), value.dtype().itemsize());
            }
          }
        });
}
