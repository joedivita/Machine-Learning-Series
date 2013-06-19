Machine Learning Series Part 3
=====

These are the source files that go along with [Part 3](http://www.joediv.com/series-create-a-machine-learning-library-in-c-part-3-advanced-minimization-algorithms/)
of my Create A Machine Learning Library in C++ Series.

main3.cpp requires that the Armadillo C++ & NLopt Libraries
are installed in order to compile. The libraries can be
found at their respective websites:

* http://arma.sourceforge.net
* http://ab-initio.mit.edu/wiki/index.php/NLopt

The Mac Accelerate Framework, or LAPACK/BLAS may also be required to
perform certain functions.

Example of linking to Armadillo, Accelerate Framework, & NLopt:

```bash
g++ main3.cpp -framework Accelerate -larmadillo -lnlopt -lm -o learning
```
