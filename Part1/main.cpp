//main.cpp file for Part 1 in Create a Machine Learning Library in C++
//Copyright 2013 Joe DiVita

#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <armadillo>

using namespace std;
using namespace arma;

//Function that computes the cost function, J(theta), for a given vector theta.
mat computeCost(const mat &X, const mat &y, const mat &theta)
{
    return ((double)1/(2*(X.n_rows)))*(X*theta-y).t()*(X*theta-y);
}

//Our Gradient Descent Function --> returns a vector theta, of our fitted model parameters.
mat gradientDescent(const mat &X, const mat &y)
{
    //gradient descent constants
    const double alpha = 0.01;
    const int iterations = 20000;
    
    mat theta = mat(X.n_cols, 1).zeros();
    
    for(int i=0; i<iterations; i++)
    {
        theta = theta - alpha*((double)1/(X.n_rows))*X.t()*(X*theta-y);
        
        //cout<<computeCost(X, y, theta)<<endl;
        //uncomment the above line if you want to to see how the cost converges.
    }
    
    return theta;
}

//Our Normal Equation Function --> returns a vector theta, of our fitted model parameters.
mat normalEquation(const mat &X, const mat &y)
{
    return pinv(X.t()*X)*X.t()*y;
}

//Feature Normalization --> returns a mxn Normalized X
void normalize(mat &X, mat &mu, mat &sigma)
{
    //these matrices need to be same demensions as X & will hold the average and standard deviation for each column of X.
    mat avgMatrix = mat(X.n_rows, X.n_cols);
    mat stdDevMatrix = mat(X.n_rows, X.n_cols);
    
    //mu vector that holds the average for each feature, sigma is a vector the standard deviation for each feature.
    mu = (sum(X,0))/(X.n_rows);
    sigma = stddev(X, 0, 0);
    
    //fill avgMatrix & rangeMatrix to n rows
    for(int i=0; i<(X.n_rows); i++)
    {
        //avg of each col in X (copied to n rows)
        avgMatrix.row(i) = mu;
        //std dev of each col in X (copied to n rows)
        stdDevMatrix.row(i) = sigma;
    }
    
    //Note: '/' sign is ELEMENT-WISE division, not matrix division
    X = (X - avgMatrix)/stdDevMatrix;
    
    //x0 does not get normalized, refill to a column of ones
    X.col(0).ones();
    
    //output the Normalized version of X.
    cout<<"Normalized X:"<<endl;
    cout<<X<<endl;
}

int main(int argc, const char * argv[])
{
    /*
     THIS SECTION PERTAINS TO READING IN OUR TRAINIG DATA FROM A OUR TRAINING DATA FILE, WHICH CONTAINS COMMA SEPERATED VALUES OF OUR FEATURES, WITH THE LAST COLUMN BEING THE "ANSWER"
     */
    
    //open the file containing our training dataset
    ifstream myfile;
    myfile.open("housingInfo.txt");
   
    //vector to store each row in
    vector<string> allLines;
    
    //initialize # of training examples to 0
    int m = 0;
    
    while(myfile.good())
    {
        string line;
        getline(myfile, line);
        
        //add each row to allLines
        allLines.push_back(line);
        
        //count the number of training examples (rows in text file)
        m++;
    }
    
    //compensate for extra \n in at bottom of text file.
    m = m-1;
    
    //initialize # of features to 0
    int n = 0;
    
    //analyze each line in in the txt file to calculate # of features
    istringstream iss2(allLines[0]);
    
    while(iss2.good())
    {
        string section2;
        getline(iss2, section2, ',');
        
        //count number of features
        n++;
    }

    //initialize y to mx1 vector
    mat y = mat(m, 1);

    //initialize X to mxn vector
    mat X = mat(m, n);
    
    //first column of X is a column of all ones, this is feature x0
    X.col(0).ones();
    
    //fill up y & X with values from training set
    for(int i=0; i<m; i++)
    {
        istringstream iss(allLines[i]);
        
        string section;
        
        int temp_n = 0;
        
        for(int k=1; k<=n; k++)
        {
            getline(iss, section, ',');
            
            if(k==n)
                y(i,0) = atof(section.c_str());
            else
                X(i,k) = atof(section.c_str());
        }
    }
    
    /*
     THIS ENDS THE SECTION OF LOADING TRAINING DATA IN FROM TEXT FILE
    */

    cout<<""<<endl;
    
    //print out X
    cout<<"Loaded X:"<<endl;
    cout<<X<<endl;
    
    //we initialize mu and sigma to 1xn matricies, we will need this variables later to noramlize the inputs of #of beds/baths that the user enters when making predictions.
    mat mu = mat(1,X.n_cols);
    mat sigma = mat(1,X.n_cols);
    
    normalize(X,mu,sigma);
    
    //get value for theta by using gradient descent.
    cout<<"Gradient Descent: "<<endl;
    mat theta = gradientDescent(X, y);
    cout<<theta<<endl;
    
    //get value for theta by using the normal equation.
    cout<<"Normal Equation:"<<endl;
    theta = normalEquation(X,y);
    cout<<theta<<endl;
    
    /*********************
    Use our new learning algorithm to predict housing prices based on our training data:
    *********************/
    
    //initialize an empty vector to hold the features for our example
    mat testSetFeatures = mat(n,1);
    
    double x1 = 0;
    double x2 = 0;
    
    cout<<"Enter number of bedrooms: ";
    cin>>x1;
    cout<<"Enter number of bathrooms: ";
    cin>>x2;
 
    //add the x0 column and initialize to 1
    testSetFeatures(0,0) = 1.0;
    
    //We need to normalize these features (since our theta is based of normalized values)
    testSetFeatures(1,0) = (x1-mu(0,1))/(sigma(0,1));
    testSetFeatures(2,0) = (x2-mu(0,2))/(sigma(0,2));
    
    //Implement the linear regression hypothesis function to make predictions:
    mat prediction = theta.t() * testSetFeatures;
    
    cout<<"Predicted House Price: $"<<prediction<<endl;
   
    return 0;
}

