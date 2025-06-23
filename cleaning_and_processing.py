import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt


# import the data set and getting the basic information

df =pd.read_csv("/Volumes/SSD/Titanic-Dataset.csv")

print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

# by observing the data,i observe that age,cabins and embarked have null values
# so i will fill the null values with mean for age and mode for embarked and cabins

df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# after filling the null values, i will check again for null values
print("\nAfter filling null values:")
print(df.head())



print(df.isnull().sum())

# categorial encoding can be done in sex and embarked columns

df['Sex']=df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# checking by printing the sex and embarked columns
print("\nAfter encoding:")

print("_____for emabrked column____")

print(df['Embarked'])


print("____sex column____")
print(df['Sex'])



# normalizing the data
# we will normalize the age and fare columns 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
# checking the normalized data
print("\nAfter normalization:")
print(df[['Age', 'Fare']].head())

# visualizing the outliers in the data

sns.boxplot(x=df['Fare'])
plt.show()


# removing the utliers in the fare columns



# Let's check Fare column outliers
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")

# Compute fences
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

print(f"Lower fence: {lower_fence}")
print(f"Upper fence: {upper_fence}")

# Remove outliers
df_no_outliers = df[(df['Fare'] >= lower_fence) & (df['Fare'] <= upper_fence)]

print(f"Original data size: {df.shape[0]}")
print(f"Data size after removing outliers: {df_no_outliers.shape[0]}")

# Save cleaned data
df_no_outliers.to_csv("titanic_no_outliers.csv", index=False)

# Visualizing the cleaned data
sns.boxplot(x=df_no_outliers['Fare'])
plt.title("Fare after removing outliers")
plt.show()









