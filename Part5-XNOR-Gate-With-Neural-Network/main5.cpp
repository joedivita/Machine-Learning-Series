//main5.cpp file for Part 5 in Create a Machine Learning Library in C++
//Copyright 2013 Joe DiVita

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

mat sigmoid(const mat &z)
{
    return (1.0 / (1.0 + exp(-z)));
}

void addBias(mat &inputs)
{
    mat inputBias = mat(1,1);
    inputBias(0,0) = 1.0;
    inputs.insert_rows(0, inputBias);
}

void reduceNoise(mat &value)
{
    //clean up the noise a bit
    if(value(0,0) < 0.05)
        value(0,0) = 0.0;
    if(value(0,0) > 0.95)
        value(0,0) = 1.0;
}

mat AND_Gate(const mat &inputs)
{
    mat theta_AND = mat(1, inputs.n_rows);
    theta_AND(0,0) = -30.0;
    
    for(int i=1; i<(inputs.n_rows); i++)
        theta_AND(0,i) = 20.0;
    
    mat AND_result = reduceNoise(sigmoid(theta_AND*inputs));
    
    return AND_result;
}

mat OR_Gate(const mat &inputs)
{
    mat theta_OR = mat(1, inputs.n_rows);
    theta_OR(0,0) = -10.0;
    
    for(int i=1; i<(inputs.n_rows); i++)
        theta_OR(0,i) = 20.0;
    
    mat OR_result = reduceNoise(sigmoid(theta_OR*inputs));
    
    return OR_result;
}

mat NOT_AND_NOT_Gate(const mat &inputs)
{
    mat theta_NOT_AND_NOT = mat(1, inputs.n_rows);
    theta_NOT_AND_NOT(0,0) = 10.0;
    
    for(int i=1; i<(inputs.n_rows); i++)
        theta_NOT_AND_NOT(0,i) = -20.0;
    
    mat NOT_AND_NOT_result = reduceNoise(sigmoid(theta_NOT_AND_NOT*inputs));
    
    return NOT_AND_NOT_result;
}

int main()
{
    mat inputLayer = mat(2,1);
    cout<<"Enter first input: ";
    cin>>inputLayer(0,0);
    cout<<"Enter second input: ";
    cin>>inputLayer(1,0);
    
    addBias(inputLayer);
    
    mat hiddenLayer1 = mat(2,1);
    hiddenLayer1.row(0) = AND_Gate(inputLayer);
    hiddenLayer1.row(1) = NOT_AND_NOT_Gate(inputLayer);
    
    addBias(hiddenLayer1);
    
    mat outputLayer = mat(1,1);
    outputLayer.row(0) = OR_Gate(hiddenLayer1);
    
    cout<<"XNOR Result: "<<outputLayer(0,0)<<endl;
    
    return 0;
}

