Machine Learning Series Part 2
=====

These are the source files that go along with [Part 2](http://www.joediv.com/series-create-a-machine-learning-library-in-c-part-2-logistic-regression-regularization/)
of my Create A Machine Learning Library in C++ Series.

main2.cpp requires that Armadillo C++ Library is installed  in
order to compile. The library can be found at their website:

http://arma.sourceforge.net/

The Mac Accelerate Framework, or LAPACK/BLAS may also be required to
perform certain functions.

Example of linking to Armadillo and Accelerate Framework:

```bash
g++ main2.cpp -framework Accelerate -larmadillo -o learning
```
