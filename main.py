# -*- coding: utf-8 -*-
"""
@author: Vishakha Prakash Ratnakar
"""
Log_accuracy_list = []
sklearn_accuracy_list= []


# for loop for iterating 10 times
for i in range(1,11): 
    
    import pandas as pd
    import numpy as np
    from final_code import LogisticRegression
    from final_code import accuracy
    from final_code import z_score
    from sklearn.model_selection import train_test_split

#-----------------------------------------------------------------------------
#importing data and spliting into training and testing data
#-----------------------------------------------------------------------------
    df = pd.read_csv(r"D:\Semester 2\deep learning\blobs300.csv",sep='\t')
    
    # takes random data in each iteration
    df = df.sample(frac=1).reset_index(drop=True)

    df.columns =['fire','year','temp','humidity','rainfall','drought_code'
                 ,'buildup_index','day','month','wind_speed']


    dataset_data = df.drop(df.columns[[0]], axis=1)
    
    # Perform feature scaling on dataset
    dataset_data = z_score(dataset_data)
    dataset_data=np.array(dataset_data, dtype=np.float32)

    # performing cleaing operation on labels and converting yes and no
    # into 0's and 1's

    dataset_labels = df['fire']
    dataset_labels = df['fire'].str.strip()
    dataset_labels_dummies = pd.get_dummies(dataset_labels)  
    dataset_labels = dataset_labels_dummies.iloc[:, 1]
    
    # taking the training data size of 2/3 from total dataset
    X_train, X_test, y_train,y_test = train_test_split(dataset_data,dataset_labels,test_size = 0.334)
    #train_size = int(0.6667 * len(df))
    
    #Spliting into training and test data
    #X_train = dataset_data[:train_size]
    #X_test = dataset_data[train_size:]

    #y_train = dataset_labels[:train_size]
    #y_test = dataset_labels[train_size:]
    
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()


#------------------------------------------------------------------------------
#          Accuracy using Constructed Logistic Regression classifier
#------------------------------------------------------------------------------    
    regressor = LogisticRegression( learning_rate=0.001,no_iterations=4500)
    costs = regressor.fit_function( X_train, y_train)
    
    #probabilities is used to store the sigmoid function values which are used for plotting ROC curve
    test_prediction,probabilities = regressor.predict(X_test) 
    
    # calculating accuracy
    Log_accuracy=accuracy(y_test,test_prediction)
    #print('Test Accuracy from our code',Log_accuracy)

    #Log_accuracy_list is used to store the accuarcies for each iteration
    Log_accuracy_list.append(Log_accuracy*100)
    
#------------------------------------------------------------------------------
#        Accuracy using SKLearn code for Logistic Regression
#------------------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression

    logistic =LogisticRegression()
    logistic.fit(X_train,y_train)
    predictions_train = logistic.predict(X_test)
    sklearn_accuracy=accuracy_score(y_test,predictions_train)
    #print('Test Accuracy from sklearn code',sklearn_accuracy)
    
 #sklearn_accuracy_list is used to store the accuarcies for each iteration
    sklearn_accuracy_list.append(sklearn_accuracy*100)
    
    
    print('\n')
#------------------------------------------------------------------------------
#code to load predicted results and actual test results in excel file
#------------------------------------------------------------------------------
    
    #from pandas import Series, ExcelWriter
    df_actual_predicted_op = pd.DataFrame({ 'Acutal_Test_Results':y_test,'Predicted_Test_Results':test_prediction,'SKLearn_Predicted_Test_Results':predictions_train})

    # create excel writer object
    if(i==1):
        writer = pd.ExcelWriter('D:\Semester 1\MAchine learning\Assignment 2\Result_output.xlsx',engine='openpyxl',mode='w')
        df_actual_predicted_op.to_excel(writer, sheet_name='sheet '  + str(i))
        writer.save()
    else: 
        writer = pd.ExcelWriter('D:\Semester 1\MAchine learning\Assignment 2\Result_output.xlsx',engine='openpyxl',mode='a')
        df_actual_predicted_op.to_excel(writer, sheet_name='sheet '  + str(i))
        writer.save()
   
    # save the excel

print('Test results and predicted results are  written successfully to Excel File.')
#writer.close()

#------------------------------------------------------------------------------
#code to load predicted results and actual test results in excel file
#------------------------------------------------------------------------------

df_Accuracy = pd.DataFrame({'Accuracy_Score_Logistic':Log_accuracy_list,'Accuracy_Score_SKLearn':sklearn_accuracy_list})

# create excel writer object
Accuracy_writer = pd.ExcelWriter('D:\Semester 1\MAchine learning\Assignment 2\Accuracy_output.xlsx')
df_Accuracy.to_excel(Accuracy_writer)

# save the excel
Accuracy_writer.save()
#Accuracy_writer.close()
print('Accuracy score is written successfully to Excel File.')


#------------------------------------------------------------------------------
#print average of accuracy
#------------------------------------------------------------------------------
from statistics import mean
print(df_Accuracy,"\n")

print('The mean accuracy of our classifier after 10 iteration',mean(Log_accuracy_list))
print('\nThe mean accuracy of SKLearn classifier after 10 iteration',mean(sklearn_accuracy_list))

#------------------------------------------------------------------------------
#ROC curve of our classifier
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from sklearn import metrics

lw = 2
fpr, tpr, _ = metrics.roc_curve(y_test,  probabilities.T)
auc = metrics.roc_auc_score(y_test, probabilities.T)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC curve of our classifier')
plt.show()

#------------------------------------------------------------------------------
#ROC curve of SKklearn classifier
#------------------------------------------------------------------------------
y_pred_proba = logistic.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC curve of SKlearn')
plt.show()

#-----------------------------------------------------------------------------
#Plot for cost vs numver of iterations
#-----------------------------------------------------------------------------
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title('Cost reduction over time')
plt.show()


