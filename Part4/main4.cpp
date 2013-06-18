#include <iostream>
#include "smartdata.hpp"

using namespace std;

int main()
{
    smartdata::dataset myDataSet("housingInfo.txt", smartdata::CSV);
    myDataSet.setAlgorithm(smartdata::LINEAR_REGRESSION);
    myDataSet.learn();
    
    vector<double> myInputs(2);
    
    myInputs[0] = 4;
    myInputs[1] = 3;
    
    double result = myDataSet.predict(myInputs).doubleVal;
    
    cout<<result<<endl;
    
    vector< vector<double> > moreDataX(2);
    vector< vector<double> > moreDataY(2);
    
    vector<double> X1(2);
    vector<double> y1(1);
    
    vector<double> X2(2);
    vector<double> y2(1);
    
    X1[0] = 2;
    X1[1] = 1;
    y1[0] = 80000;
    
    X2[0] = 6;
    X2[1] = 5;
    y2[0] = 740000;
    
    moreDataX[0] = X1;
    moreDataX[1] = X2;
    moreDataY[0] = y1;
    moreDataY[1] = y2;
    
    myDataSet.teach(moreDataX, moreDataY);
    myDataSet.learn();
    
    result = myDataSet.predict(myInputs).doubleVal;
    
    cout<<result<<endl;
    
    
    return 0;
}
