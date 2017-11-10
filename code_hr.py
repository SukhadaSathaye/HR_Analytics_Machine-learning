import numpy as np # Data arrays made easy
import pandas as pd # Data tables made easy with Pandas

import matplotlib.pyplot as plt # For graph
import matplotlib as matplot # For graph
import seaborn as sns # For data visualization
# Built-in magic command function for displaying graph inline
#%matplotlib inline


df = pd.DataFrame.from_csv('/Users/ashwinhabbu/PycharmProjects/Python_project/HR_comma_sep.csv', index_col=None)

# Renaming certain columns for better readability
# Can rename the columns in this way
df = df.rename(columns={'satisfaction_level': 'satisfaction',
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })
print('Column rename done')
df.head()
# Move the reponse variable "turnover" to the front of the table
front = df['turnover'] #Creating a data column from df['turnover']
df.drop(labels=['turnover'], axis=1, inplace=True) # axis=1 means that we are referring to col and not row
#inplace = True means you are not assigning the df to another df it is done inplace
df.insert(0, 'turnover', front)
#df.head()

corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#sns.plt.title('Heatmap of Correlation Matrix')
emp_population_satisfaction = df['satisfaction'].mean()
emp_turnover_satisfaction = df[df['turnover']==1]['satisfaction'].mean()


print( 'The mean for the employee population is: ' + str(emp_population_satisfaction) )
print( 'The mean for the employees that had a turnover is: ' + str(emp_turnover_satisfaction) )



#scipy is used for statistics
import scipy.stats as stats
stats.ttest_1samp(a=  df[df['turnover']==1]['satisfaction'], # Sample of Employee satisfaction who had a Turnover
                  popmean = emp_population_satisfaction)  # Employee Population satisfaction mean


degree_freedom = len(df[df['turnover']==1])
LQ = stats.t.ppf(0.025,degree_freedom)  # Left Quartile
RQ = stats.t.ppf(0.975,degree_freedom)  # Right Quartile
print ('The t-distribution left quartile range is: ' + str(LQ))
print ('The t-distribution right quartile range is: ' + str(RQ))


from sklearn.cluster import KMeans

# Graph and create 3 clusters of Employee Turnover
kmeans = KMeans(n_clusters=3,random_state=2)
kmeans.fit(df[df.turnover==1][["satisfaction","evaluation"]])

kmeans_colors = ['green' if c == 0 else 'blue' if c == 2 else 'red' for c in kmeans.labels_]

fig = plt.figure(figsize=(10, 6))
plt.scatter(x="satisfaction",y="evaluation", data=df[df.turnover==1],
            alpha=0.25,color = kmeans_colors)
plt.xlabel("Satisfaction")
plt.ylabel("Evaluation")
plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],color="black",marker="X",s=100)
plt.title("Clusters of Employee Turnover")
plt.show()

#The above code we got it from one of the kernels of Kaggle but it did not explain
#why didHigh performing who were highly satisfied left the company
# Analyzed data of employees who were highly satisfied and high performing yet left the companuy

df_turnover_1=df[df['turnover']==1].sort_values(['evaluation','satisfaction'],ascending=False)
df_turnover_0=df[df['turnover']==0].sort_values(['evaluation','satisfaction'],ascending=False)




eval_1=df_turnover_1['evaluation'].head(10).mean()
sat_1=df_turnover_1['satisfaction'].head(10).mean()
pc_1=df_turnover_1['projectCount'].head(10).mean()
yac_1=df_turnover_1['yearsAtCompany'].head(10).mean()
y=df_turnover_1['averageMonthlyHours'].head(10).mean()/100


eval_0=df_turnover_0['evaluation'].head(10).mean()
sat_0=df_turnover_0['satisfaction'].head(10).mean()
pc_0=df_turnover_0['projectCount'].head(10).mean()
yac_0=df_turnover_0['yearsAtCompany'].head(10).mean()
x=df_turnover_0['averageMonthlyHours'].head(10).mean()/100



print(eval_0,eval_1)

arr_1=[eval_1,sat_1,pc_1,yac_1,y]
arr_0=[eval_0,sat_0,pc_0,yac_0,x]






print('The evaluation and satisfaction of employees who left the company :',eval_1,sat_1)

index = np.arange(5)

fig, ax = plt.subplots()

#index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, arr_1, bar_width,
                 alpha=opacity,
                 color='b',

                 error_kw=error_config,
                 label='Left the company')

rects2 = plt.bar(index + bar_width, arr_0, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='Did not leave')

plt.ylim([0,10])
plt.xlabel('Metric')
plt.ylabel('Average')
plt.title('Comparison of top performers')
plt.xticks(index + bar_width / 2, ('Rating', 'Satisfaction', 'Project_count', 'Tenure', 'Working hours'))
plt.legend()
plt.tight_layout()
plt.show()




#The following graphs shows differences in number of hours of top performing
#employees who left vs who did not leave

df_turnover_1=df[df['turnover']==1].sort_values(['evaluation','satisfaction'],ascending=False)
df_turnover_0=df[df['turnover']==0].sort_values(['evaluation','satisfaction'],ascending=True)


x=df_turnover_0['averageMonthlyHours'].head(10)
y=df_turnover_1['averageMonthlyHours'].head(10)


eval_1=df_turnover_1['evaluation'].head(10).mean()
sat_1=df_turnover_1['satisfaction'].head(10).mean()

print('The evaluation and satisfaction of employees who left the company :',eval_1,sat_1)

index = np.arange(10)

fig, ax = plt.subplots()

#index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, y, bar_width,
                 alpha=opacity,
                 color='b',

                 error_kw=error_config,
                 label='Left the company')

rects2 = plt.bar(index + bar_width, x, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='Did not leave')

plt.ylim([0,400])
plt.xlabel('Top performers')
plt.ylabel('Average month hours')
plt.title('Comparison of top performers')
plt.xticks(index + bar_width / 2, ('1', '2', '3', '4', '5','6','7','8','9','10'))
plt.legend()
plt.tight_layout()
plt.show()




from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler

# Create dummy variables for the 'department' and 'salary' features, since they are categorical
department = pd.get_dummies(data=df['department'],drop_first=True,prefix='dep') #drop first column to avoid dummy trap
salary = pd.get_dummies(data=df['salary'],drop_first=True,prefix='sal')
df.drop(['department','salary'],axis=1,inplace=True)
df = pd.concat([df,department,salary],axis=1)


# Create base rate model
def base_rate_model(X) :
    y = np.zeros(X.shape[0])
    return y


# Create train and test splits
target_name = 'turnover'
X = df.drop('turnover', axis=1)
robust_scaler = RobustScaler()
X = robust_scaler.fit_transform(X)
y=df[target_name]
for i in range(10,55):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=i/100, random_state=123, stratify=y)
    print(i/10)

    # Check accuracy of base rate model
    y_base_rate = base_rate_model(X_test)
    from sklearn.metrics import accuracy_score
    acc_score=accuracy_score(y_test, y_base_rate)
    print("Base rate accuracy is %2.2f" % acc_score)

    # Check accuracy of Logistic Model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2', C=1)

    model.fit(X_train, y_train)
    print ("Logistic accuracy is %2.2f" % accuracy_score(y_test, model.predict(X_test)))


    # Using 10 fold Cross-Validation to train our Logistic Regression Model
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    modelCV = LogisticRegression(class_weight = "balanced")
    scoring = 'roc_auc'
    results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
    print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))

