g++ --std=c++17 -O3 -fopenmp -lpthread -shared -I include -fPIC `python3 -m pybind11 --includes` sl.cpp -o triton_sl`python3-config --extension-suffix`
