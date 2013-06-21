//main2.cpp file for Part 2 in Create A Machine Learning Library in C++
//Copyright 2013 Joe DiVita

#include <iostream>
#include <vector>
#include <cmath>
#include <armadillo>

using namespace std;
using namespace arma;

const double lambda = 1.0;

void normalize(mat &X, mat &mu, mat &sigma)
{
    mat avgMatrix = mat(X.n_rows, X.n_cols);
    mat stdDevMatrix = mat(X.n_rows, X.n_cols);

    mu = (sum(X,0))/(X.n_rows);
    sigma = stddev(X, 0, 0);

    for(int i=0; i<(X.n_rows); i++)
    {
        avgMatrix.row(i) = mu;
        stdDevMatrix.row(i) = sigma;
    }
    
    X = (X - avgMatrix)/stdDevMatrix;
    
    X.col(0).ones();

    cout<<"Normalized X:"<<endl;
    cout<<X<<endl;
}

mat sigmoid(const mat &z)
{
    return (1.0 / (1.0 + exp(-z)));
}

mat computeCost(const mat &X, const mat &y, const mat &theta)
{
    mat regTheta = theta;
    regTheta.row(0).zeros();
    
    return ((double)-1/X.n_rows)*(y.t()*log(sigmoid(X*theta)) + (1-y.t())*log(1-sigmoid(X*theta))) + (lambda/(2.0*X.n_rows))*regTheta.t()*regTheta;
}

mat computeGradient(const mat &X, const mat &y, const mat &theta)
{
    mat regTheta = theta;
    regTheta.row(0).zeros();
    
    return ((double)1/(X.n_rows)) * X.t() * (sigmoid(X * theta) - y) + ((double)lambda/X.n_rows)*regTheta;
}

mat gradientDescent(const mat &X, const mat &y)
{
    const double alpha = 0.01;
    const int iterations = 20000;
    
    mat theta = mat(X.n_cols, 1).zeros();
    
    for(int i=0; i<iterations; i++)
    {
        theta = theta - alpha * computeGradient(X,y,theta);
    }
    
    return theta;
}

int main(){
    
    //Create our testdata
    mat X = mat(5,2);
    
    //Initialize the y-intercept column to all ones
    X.col(0).ones();
    
    //Our test data for tumor sizes
    X(0,1) = 10;
    X(1,1) = 8;
    X(2,1) = 8;
    X(3,1) = 2;
    X(4,1) = 1;
    
    //Our test data for malignancy (1 = malignant, 0 = benign)
    mat y = mat(5,1);
    y(0,0) = 1;
    y(1,0) = 1;
    y(2,0) = 1;
    y(3,0) = 0;
    y(4,0) = 0;
    
    mat mu = mat(1,X.n_cols);
    mat sigma = mat(1,X.n_cols);
    
    //Normalize X
    normalize(X, mu, sigma);
    
    //Run gradient descent to obtain theta
    mat theta = gradientDescent(X,y);
    
    mat testSet = mat(2,1);
    double x1;
    
    cout<<"Enter size of tumor: ";
    cin>>x1;
    
    testSet(0,0) = 1;
    
    //Normalize user input
    testSet(1,0) = (x1-mu(0,1))/(sigma(0,1));
    
    //Obtain the % confidence that the tumor is malignant by implementing the hypothesis function
    mat predictionConfidence = sigmoid(theta.t()*testSet);
    
    //If the % confidence is greater than or equal to 50%, predict malignancy, otherwise predict it is benign
    string result = "Malignant";
    double predictionConfidenceNum = predictionConfidence(0,0);
    
    if(predictionConfidenceNum < 0.5)
    {
        result = "Benign";
        predictionConfidenceNum = 1-predictionConfidenceNum;
    }
    
    cout<<"Result of this tumor is: "<<result<<" with "<<100.0*predictionConfidenceNum<<"% confidence"<<endl;
        
    return 0;
}
