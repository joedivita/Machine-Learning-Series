Machine-Learning-Series
=====

This is a repo for all the files used in my series: [Create A Machine Learning Library in C++](http://www.joediv.com/courseras-ng-stanford-machine-learning-course-in-c).

The series follows along with Professor Andrew Ng's Stanford Course "Machine Learning" on Coursera, and implements each section in C++.  The series then attempts to organize all of the learning algorithms taught in the course into an easy to use C++ library.

Resources
=====

Dependencies
-----

Compiling many of the files in this repo require that the folowing libraries are installed:

* [Armadillo C++ Linear Algebra Library](http://arma.sourceforge.net/)
* [NLopt C++ Non-Linear Optimization Library](http://ab-initio.mit.edu/wiki/index.php/NLopt)
* [LAPACK](http://www.netlib.org/lapack/)/[BLAS](http://www.netlib.org/blas/) OR if using a Mac the [Accelerate Framework](https://developer.apple.com/performance/accelerateframework.html)

Compiling
-----

Example compiling Part3 & linking with Armadillo, NLopt, & the Accelerate Framework: 

g++ main3.cpp -framework Accelerate -larmadillo -lnlopt -lm -o learning

Questions / Comments
-----

Send all inquiries to: joediv31@gmail.com

For more information visit: http://joediv31.com
