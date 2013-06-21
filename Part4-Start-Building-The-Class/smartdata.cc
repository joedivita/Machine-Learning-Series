//smartdata.cc file for Part 4 in Create A Machine Learning Library in C++
//Copyright 2013 Joseph DiVita 
 
#include <fstream>
#include <sstream>
#include <utility>
#include <nlopt.hpp>
#include <iostream>
#include "smartdata.hpp"

//Constructor
smartdata::dataset::dataset()
{
    //set defaults
    dataset::normalize = true;
    dataset::lambda = 1.0;
    dataset::cur_minFunc = LGFBS;
}

//Contstuctor w/ load data
smartdata::dataset::dataset(const std::string &dataFile, const loadMethod &lmethod)
{
    //set defaults
    dataset::normalize = true;
    dataset::lambda = 1.0;
    dataset::cur_minFunc = LGFBS;
    
    if(lmethod == CSV)
        dataset::loadInDataFromCSV(dataFile);
    
    //initialize theta
    dataset::theta = arma::mat(dataset::X.n_cols, 1);
}

void smartdata::dataset::loadDataSet(const std::string &dataFile, const loadMethod &lmethod)
{
    if(lmethod == CSV)
        dataset::loadInDataFromCSV(dataFile);
    
    dataset::theta = arma::mat(dataset::X.n_cols, 1);
}

void smartdata::dataset::setAlgorithm(const algType &algorithmType)
{
    dataset::cur_algorithm = algorithmType;
}

void smartdata::dataset::setMinimization(const minFunc &minTechnique)
{

    if(minTechnique == GRADIENT_DESCENT)
        std::cout<<"GRADIENT_DESCENT requires passing two additional parameters: double alpha, int iterations - use: setMinimization(const smartdata::minFunc &minTechnique, const double alpha, const int iterations)"<<std::endl;
    else
        dataset::cur_minFunc = minTechnique;
}

void smartdata::dataset::setMinimization(const minFunc &minTechnique, const double alpha, const int iterations)
{
    if(minTechnique != GRADIENT_DESCENT)
    {
       std::cout<<"Two many parameters passed for this minimization function - use: setMinimization(const smartdata::minFunc &minTechnique)"<<std::endl; 
    }
    else
    {
        dataset::cur_minFunc = minTechnique;
        dataset::alpha = alpha;
        dataset::iterations = iterations;
    }
}

double myvfunc(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
    smartdata::dataset *myDataSet = (smartdata::dataset *)my_func_data;
    
    myDataSet->theta = arma::conv_to<arma::mat>::from(x);
    
    if (!grad.empty()) {
        arma::mat gradient = myDataSet->computeGradient();
        typedef std::vector<double> stdvec;
        grad = arma::conv_to<stdvec>::from(gradient);
    }
    
    arma::mat cost = myDataSet->computeCost();
    
    return cost(0,0);
}

void smartdata::dataset::learn()
{
    if(dataset::normalize == true)
        dataset::normalizeX();
    
    if(dataset::cur_minFunc == LGFBS)
    {
        nlopt::opt opt(nlopt::LD_LBFGS,dataset::X.n_cols);
       
        dataset *C = this;
        
        double minf = 0;
        std::vector<double> tempTheta(dataset::n);
    
        opt.set_min_objective(myvfunc, C);
        opt.set_ftol_rel(1e-14);
        nlopt::result myResult = opt.optimize(tempTheta, minf);
        
        dataset::theta = arma::conv_to<arma::mat>::from(tempTheta);
        
    }
    else if(dataset::cur_minFunc == GRADIENT_DESCENT)
    {
        dataset::gradientDescent();
    }
    else if(dataset::cur_minFunc == NORMAL_EQ)
    {
        dataset::normalEquation();
    }
}

void smartdata::dataset::teach(const std::vector<double> &Xnew, const std::vector<double> &ynew)
{
    
    dataset::m++;
    
    if(dataset::normalize == true)
        unormalizeX();
    
    //1. Convert Xnew & ynew to arma::mats
    arma::mat teachX = arma::join_cols(arma::mat(1,1).ones(), arma::conv_to<arma::mat>::from(Xnew)).t();
    arma::mat teachy = arma::conv_to<arma::mat>::from(ynew);
    
    dataset::X.insert_rows(dataset::X.n_rows, teachX);
    dataset::y.insert_rows(dataset::y.n_rows, teachy);

}

void smartdata::dataset::teach(const std::vector< std::vector<double> > &Xnew, const std::vector< std::vector<double> > &ynew)
{
    
    if(dataset::normalize == true)
        unormalizeX();

    for(int i=0; i<Xnew.size(); i++)
    {
        arma::mat teachX = arma::join_cols(arma::mat(1,1).ones(), arma::conv_to<arma::mat>::from(Xnew[i])).t();
        arma::mat teachy = arma::conv_to<arma::mat>::from(ynew[i]);
        
        dataset::X.insert_rows(dataset::X.n_rows, teachX);
        dataset::y.insert_rows(dataset::y.n_rows, teachy);
        
        dataset::m++;
    }
}

smartdata::result smartdata::dataset::predict(const std::vector<double> &testSet)
{
    smartdata::result myResult;
    arma::mat inputMatrix = arma::conv_to<arma::mat>::from(testSet);
    
    inputMatrix = arma::join_cols(arma::mat(1,1).ones(), inputMatrix);
  
    if(dataset::normalize == true)
    {
        inputMatrix = (inputMatrix-dataset::mu.t())/dataset::sigma.t();
        inputMatrix.row(0).ones();
    }
    
    if(dataset::cur_algorithm == smartdata::LINEAR_REGRESSION)
    {
        arma::mat rawResult = dataset::theta.t() * inputMatrix;
        myResult.doubleVal = rawResult(0,0);
    }
    else if(dataset::cur_algorithm == smartdata::LOGISTIC_REGRESSION)
    {
        arma::mat rawResult = dataset::sigmoid(dataset::theta.t()*inputMatrix);
        myResult.doubleVal = rawResult(0,0);
    }
    
    return myResult;
}

void smartdata::dataset::loadInDataFromCSV(const std::string &filename)
{
    std::ifstream myfile;
    myfile.open(filename.c_str());
    
    dataset::m = 0;
    dataset::n = 0;
    
    std::vector<std::string> allLines;
    
    while(myfile.good())
    {
        std::string line;
        std::getline(myfile, line);
        allLines.push_back(line);
        dataset::m++;
    }
    
    dataset::m = dataset::m-1;
    
    std::istringstream iss2(allLines[0]);
    
    while(iss2.good())
    {
        std::string section2;
        std::getline(iss2, section2, ',');
        dataset::n++;
    }
    
    dataset::y = arma::mat(dataset::m, 1);
    dataset::X = arma::mat(dataset::m, dataset::n);
    dataset::X.col(0).ones();

    for(int i=0; i<dataset::m; i++)
    {
        std::istringstream iss(allLines[i]);
        std::string section;
        
        for(int k=1; k<=dataset::n; k++)
        {
            std::getline(iss, section, ',');
            
            if(k==dataset::n)
                y(i,0) = std::atof(section.c_str());
            else
                X(i,k) = std::atof(section.c_str());
        }
    }

}


//Utility functions:

void smartdata::dataset::normalizeX()
{
    arma::mat avgMatrix = arma::mat(dataset::X.n_rows, dataset::X.n_cols);
    arma::mat stdDevMatrix = arma::mat(dataset::X.n_rows, dataset::X.n_cols);
    
    dataset::mu = (arma::sum(dataset::X,0))/(dataset::X.n_rows);
    dataset::sigma = arma::stddev(dataset::X, 0, 0);

    for(int i=0; i<(dataset::X.n_rows); i++)
    {
        avgMatrix.row(i) = dataset::mu;
        stdDevMatrix.row(i) = dataset::sigma;
    }
    
    dataset::X = (dataset::X - avgMatrix)/stdDevMatrix;
    dataset::X.col(0).ones();
    
}

void smartdata::dataset::unormalizeX()
{
    arma::mat avgMatrix = arma::mat(dataset::X.n_rows, dataset::X.n_cols);
    arma::mat stdDevMatrix = arma::mat(dataset::X.n_rows, dataset::X.n_cols);
    
    for(int i=0; i<(dataset::X.n_rows); i++)
    {
        avgMatrix.row(i) = dataset::mu;
        stdDevMatrix.row(i) = dataset::sigma;
    }
    
    dataset::X = (dataset::X % stdDevMatrix) + avgMatrix;
    dataset::X.col(0).ones();
}

//changes based on algorithm
arma::mat smartdata::dataset::computeCost()
{
    arma::mat regTheta = dataset::theta;
    regTheta.row(0).zeros();
    
    if(dataset::cur_algorithm == smartdata::LINEAR_REGRESSION)
        return ((double)1/(2*(dataset::X.n_rows)))*(dataset::X*dataset::theta-dataset::y).t()*(dataset::X*dataset::theta-dataset::y) + (dataset::lambda/(2.0*dataset::X.n_rows))*regTheta.t()*regTheta;
    else if(dataset::cur_algorithm == smartdata::LOGISTIC_REGRESSION)
        return ((double)-1/dataset::X.n_rows)*(dataset::y.t()*arma::log(dataset::sigmoid(dataset::X*dataset::theta)) + (1-dataset::y.t())*arma::log(1-dataset::sigmoid(dataset::X*dataset::theta))) + (dataset::lambda/(2.0*dataset::X.n_rows))*regTheta.t()*regTheta;
    else
        return arma::mat(X.n_rows,1);
}

//changes based on algorithm
arma::mat smartdata::dataset::computeGradient()
{
    arma::mat regTheta = dataset::theta;
    regTheta.row(0).zeros();
    
    if(dataset::cur_algorithm == smartdata::LINEAR_REGRESSION)
        return ((double)1/(dataset::X.n_rows))*dataset::X.t()*(dataset::X*dataset::theta-dataset::y) + ((double)dataset::lambda/dataset::X.n_rows)*regTheta;
    else if(dataset::cur_algorithm == smartdata::LOGISTIC_REGRESSION)
        return ((double)1/(dataset::X.n_rows)) * dataset::X.t() * (dataset::sigmoid(dataset::X * dataset::theta) - dataset::y) + ((double)dataset::lambda/dataset::X.n_rows)*regTheta;
    else
        return arma::mat(1,1);
}

void smartdata::dataset::gradientDescent()
{
    dataset::theta = arma::mat(dataset::X.n_cols, 1).zeros();
    
    for(int i=0; i<dataset::iterations; i++)
        dataset::theta = dataset::theta - dataset::alpha * dataset::computeGradient();
}

void smartdata::dataset::normalEquation()
{
    arma::mat A = arma::mat(dataset::X.n_cols,dataset::X.n_cols).ones();
    arma::mat B = arma::diagmat(A);
    B(0,0) = 0;
    dataset::theta = arma::pinv(dataset::X.t()*dataset::X + dataset::lambda*B)*dataset::X.t()*dataset::y;
}

arma::mat smartdata::dataset::sigmoid(const arma::mat &z)
{
    return (1.0 / (1.0 + arma::exp(-z)));
}


