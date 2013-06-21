Machine Learning Series Part 5
=====

These are the source files that go along with [Part 5](http://www.joediv.com/series-create-a-machine-learning-library-in-c-part-5-xnor-gate-with-neural-network)
of my Create A Machine Learning Library in C++ Series.

main5.cpp requires that Armadillo C++ Library is installed  in
order to compile. The library can be found at their website:

http://arma.sourceforge.net/

The Mac Accelerate Framework, or LAPACK/BLAS may also be required to
perform certain functions.

Example of linking to Armadillo and Accelerate Framework:

```bash
g++ main5.cpp -framework Accelerate -larmadillo -o learning
```

