Machine Learning Series Part 4
=====

These are the source files that go along with [Part 4](http://www.joediv.com/series-create-a-machine-learning-library-in-c-part-4-start-building-the-class/)
of my Create A Machine Learning Library in C++ Series.

main4.cpp requires that the Armadillo C++ & NLopt Libraries
are installed in order to compile. The libraries can be
found at their respective websites:

* http://arma.sourceforge.net/
* http://ab-initio.mit.edu/wiki/index.php/NLopt

The Mac Accelerate Framework, or LAPACK/BLAS may also be required to
perform certain functions.

First you must compile the smartclass files as shown below:

```bash
g++ -c smartclass.cc -o smartclass.o
```

Then compile with smartclass.o and link to Armadillo, Accelerate Framework, & NLopt:

```bash
g++ main4.cpp smartclass.o -framework Accelerate -larmadillo -lnlopt -lm -o learning
```
