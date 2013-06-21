//smartdata.hpp file for Part 4 in Create A Machine Learning Library in C++
//Copyright 2013 Joseph DiVita 
 
#ifndef SMARTDATA_HPP
#define SMARTDATA_HPP

#include <string>
#include <armadillo>
#include <vector>

namespace smartdata {

    enum loadMethod{
        CSV = 0
    };
    
    enum algType{
        LINEAR_REGRESSION = 0,
        LOGISTIC_REGRESSION = 1
    };
    
    enum minFunc{
        LGFBS = 0,
        GRADIENT_DESCENT = 1,
        NORMAL_EQ = 2
    };
    
    typedef struct{
        int intVal;
        float floatVal;
        double doubleVal;
        std::string stringVal;
    }result;
    
    class dataset{
    
    public:
        
        //public variables
        bool normalize;
        double lambda;
        algType cur_algorithm;
        arma::mat X;
        arma::mat y;
        arma::mat theta;
        
        //public functions
        dataset();
        dataset(const std::string &dataFile, const loadMethod &lmethod);
        void loadDataSet(const std::string &dataFile, const loadMethod &lmethod);
        void setAlgorithm(const algType &algorithmType);
        void setMinimization(const minFunc &minTechnique);
        void setMinimization(const minFunc &minTechnique, const double alpha, const int iterations);
        void learn();
        void teach(const std::vector<double> &Xnew, const std::vector<double> &ynew);
        void teach(const std::vector< std::vector<double> > &Xnew, const std::vector< std::vector<double> > &ynew);
        result predict(const std::vector<double> &testSet);
        arma::mat computeCost();
        arma::mat computeGradient();
        
    private:
        
        //private variables
        minFunc cur_minFunc;
        arma::mat mu;
        arma::mat sigma;
        double alpha;
        int iterations;
        int n;
        int m;
        
        //private functions
        void normalizeX();
        void unormalizeX();
        void loadInDataFromCSV(const std::string &filename);
        void gradientDescent();
        void normalEquation();
        arma::mat sigmoid(const arma::mat &z);
        
    };
}

#endif




