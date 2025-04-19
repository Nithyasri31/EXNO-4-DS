# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/666b79e5-b3be-4d6e-8f2b-9f91a79cb938)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/6edfa8a8-ac2c-49ea-9f51-b670a6424ecf)
```
max_val=np.max(np.abs(df[['Height','Weight']]))
max_val
```
![image](https://github.com/user-attachments/assets/017f8126-de8e-4f5c-be2f-8e473ca8f2c0)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/008bddab-6e80-412b-b966-07380ca2366c)
```
from sklearn.preprocessing import Normalizer
nm=Normalizer()
df[['Height','Weight']]=nm.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/2c4208ae-2d26-476d-98fb-a06cf9ae94ab)
```
from sklearn.preprocessing import MaxAbsScaler
mas=MaxAbsScaler()
df[['Height','Weight']]=mas.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/d96dfde1-c7f1-460a-a628-fccef65bd4ad)
```
from sklearn.preprocessing import RobustScaler
rs=RobustScaler()
df[['Height','Weight']]=rs.fit_transform(df[['Height','Weight']])
df.head(5)
```
![image](https://github.com/user-attachments/assets/34210956-dbb2-4fac-a5cd-b87b521bd867)
```
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/ddd6f829-028b-47dd-b5a9-eb9bf9472690)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
contingency_table
```
![image](https://github.com/user-attachments/assets/1fc19d8e-7934-458a-b8de-595d096b126e)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print('Chi-square statistic:',chi2)
print('p-value:',p)
```
![image](https://github.com/user-attachments/assets/0fe7a457-ab47-4746-8b6a-8fd7be0e126d)
```
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
import pandas as pd
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': ['A', 'B', 'C', 'A', 'B'],
    'Feature3': [5, 4, 3, 2, 1],
    'Target': [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

df
```
![image](https://github.com/user-attachments/assets/4c62059b-f167-48d3-be0e-645937b81ad1)
```
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
print('Selected features:',x_new)
```
![image](https://github.com/user-attachments/assets/7ee7d7e8-baa0-4ebc-9517-c0f1ebdb5e4e)
```
selectedFeatureIndices=selector.get_support(indices=True)
 selectedFeatures=x.columns[selectedFeatureIndices]
 print('Selected features:',selectedFeatures)
```
![image](https://github.com/user-attachments/assets/1f3b51ea-3d41-485a-9f9e-d63697e11f43)

# RESULT:
       Feature Scaling and feature selection process has been successfully performed on the data set.

