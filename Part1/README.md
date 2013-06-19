Machine Learning Series Part I
=====

These are the source files that go along with [Part 1](http://www.joediv.com/series-create-a-machine-learning-library-in-c-part-1-linear-regression-gradient-descent/)
of my Create A Machine Learning Library in C++ Series.

main.cpp requires that Armadillo C++ Library is installed  in
order to compile. The library can be found at their website:

http://arma.sourceforge.net/

The Mac Accelerate Framework, or LAPACK/BLAS may also be required to
perform certain functions.

Example of linking to Armadillo and Accelerate Framework:

```bash
g++ main.cpp -framework Accelerate -larmadillo -o learning
```
