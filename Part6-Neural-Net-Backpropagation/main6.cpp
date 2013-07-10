//main6.cpp file for Part 6 in Create a Machine Learning Library in C++
//Copyright 2013 Joe DiVita

#include <iostream>
#include <armadillo>
#include <nlopt.hpp>

using namespace std;
using namespace arma;

const int s1 = 2;
const int s2 = 4;
const int s3 = 1;

const double m = 5;

const double lambda = 0.01;

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
mat sigmoid(const mat &z)
{
    return (1.0 / (1.0 + exp(-z)));
}

double myvfunc(const vector<double> &x, vector<double> &grad, void *my_func_data)
{
    
    //Extract and recast our X & y mats stored in my_func_data
    mat* pC = (mat *)my_func_data;
    mat Xmat = pC[0];
    mat Y = pC[1];
    
    //Create an arma::mat called theta, converted from type vector<double>
    mat theta = conv_to<mat>::from(x);
    
    //get back Theta1, and Theta2
    mat theta1 = theta.rows(0, s2*(s1+1)-1);
    mat theta2 = theta.rows(theta1.n_rows, theta1.n_rows + (s3*(s2+1))-1);
    theta1.reshape(s2, s1+1);
    theta2.reshape(s3, s2+1);
    
    //Calculate Regularization term for cost function
    double regularization = ((double)lambda / (2.0*m)) * (accu(square(theta1.cols(1, theta1.n_cols-1))) + accu(square(theta2.cols(1, theta2.n_cols-1))));
    
    double cost = 0;
    
    //Initialize the gradients for Theta1 & 2
    mat Theta1_grad = mat(theta1.n_rows, theta1.n_cols).zeros();
    mat Theta2_grad = mat(theta2.n_rows, theta2.n_cols).zeros();
    
    //Loop through all training sets
    for(int i =0; i<m; i++)
    {
        //Forward Propagate to get a3
        mat a1 = Xmat.row(i);
        a1.insert_cols(0,1.0); //add bias unit
        
        mat a2 = sigmoid(theta1*a1.t());
        a2.insert_rows(0,1.0); //add bias unit
        
        mat a3 = sigmoid(theta2*a2);
        
        //Get Y in correct format -- doesn't matter in this example because Y only has one output val -- depends on how you input y from your training data
        mat y = Y.row(i);
        y = y.t();
        
        //Sum the costs through the training set
        mat costMat = (y.t()*log(a3) + (1-y.t())*log(1-a3));
        cost = cost + costMat(0,0);
        
        //Back propagate and cacluate deltas
        mat delta3 = a3-y;
        mat delta2 = (theta2.t()*delta3)%(a2%(1-a2));
        delta2 = delta2.rows(1,delta2.n_rows-1);
        
        //Sum up gradients (Delta_Cap)
        Theta1_grad = Theta1_grad + delta2*a1;
        Theta2_grad = Theta2_grad + delta3*a2.t();
        
    }
    
    //Ad regularization to gradients
    Theta1_grad = (1.0/m)*Theta1_grad + (lambda/m)*(theta1);
    Theta2_grad = (1.0/m)*Theta2_grad + (lambda/m)*(theta2);
    
    //Remove regularization for Bias term gradients
    Theta1_grad.col(0) = Theta1_grad.col(0) - (lambda/m)*(theta1.col(0));
    Theta2_grad.col(0) = Theta2_grad.col(0) - (lambda/m)*(theta2.col(0));
    
    //Unroll the gradients into one vector
    Theta1_grad.reshape(Theta1_grad.n_elem, 1);
    Theta2_grad.reshape(Theta2_grad.n_elem, 1);
    mat unrolledGradient = join_cols(Theta1_grad,Theta2_grad);
    
    //Finish computing cost function
    cost = cost * (-1.0/m);
    cost = cost + regularization;
    
    
    if (!grad.empty()) {
        
        //Set nlopt min function gradient vector equal to the unrolled version of our gradients
        typedef vector<double> stdvec;
        grad = conv_to<stdvec>::from(unrolledGradient);
        
    }
    
    
    return cost;
    
}
int main(int argc,char **argv)
{
    
    //Training Data - 5 examples
    mat X = mat(5,2);
    
    X(0,0) = 1.0;
    X(0,1) = 1.0;
    X(1,0) = 3.0;
    X(1,1) = 4.0;
    X(2,0) = 10.0;
    X(2,1) = 10.0;
    X(3,0) = 12.0;
    X(3,1) = 11.0;
    X(4,0) = 2.0;
    X(4,1) = 3.0;
    
    mat Y = mat(5,1);
    
    Y(0,0) = 0.0;
    Y(1,0) = 0.0;
    Y(2,0) = 1.0;
    Y(3,0) = 1.0;
    Y(4,0) = 0.0;
    
    //THETAS
    mat theta1 = randu<mat>(s2, s1+1);
    mat theta2 = randu<mat>(s3, s2+1);
    
    //Unroll the thetas and join together;
    theta1.reshape(theta1.n_elem, 1);
    theta2.reshape(theta2.n_elem, 1);
    
    mat unrolledTheta = join_cols(theta1,theta2);
    
    //Initialize our opt object with the LBFGS algorithm and number of parameters
    int totalParams = unrolledTheta.n_rows;
    
    nlopt::opt opt(nlopt::LD_LBFGS,totalParams);
    
    //Put X & y into a variable to be passed as extra data
    mat C[2];
    C[0] = X;
    C[1] = Y;
    
    //create a starting point for theta & randomize the initial values
    vector<double> testTheta(totalParams);
    for(int i=0; i<totalParams; i++)
        testTheta[i] = fRand(-1.0, 1.0);
    
    //Assign objective as minimizing myvfunc (cost function) and pass in &C as extra data
    opt.set_min_objective(myvfunc, &C);
    
    //Set stopping tolerance
    opt.set_ftol_rel(1e-4);

    //variable to hold returned minimum cost
    double minf = 0;
    
    //run the optimize function to optimize theta
    nlopt::result myResult = opt.optimize(testTheta, minf);
    
    cout<<"Cost: "<<minf<<endl;
    
    //convert vector<double> testTheta to type mat
    mat testTheta2 = conv_to<mat>::from(testTheta);
    
    theta1 = testTheta2.rows(0, s2*(s1+1)-1);
    theta2 = testTheta2.rows(theta1.n_rows, theta1.n_rows + (s3*(s2+1))-1);
    
    theta1.reshape(s2, s1+1);
    theta2.reshape(s3, s2+1);
    
    //Try a test set
    mat test = mat(1,2);
    test(0,0) = 11.0;
    test(0,1) = 11.0;
    test.insert_cols(0, mat(test.n_rows, 1).ones());
    
    //forward propagate to find prediction
    mat a2 = sigmoid(test*theta1.t());
    a2.insert_cols(0, mat(a2.n_rows, 1).ones());
    
    mat a3 = sigmoid(a2*theta2.t());
    
    cout<<"Prediction: "<<a3<<endl;
    
    
    return 0;
}
