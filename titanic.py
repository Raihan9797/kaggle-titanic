# %%
import pandas as pd
import numpy as np

# %%
# get data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
gender_submission = pd.read_csv('data/gender_submission.csv')


# %%
# train EDA
train # 891 rows, 12 cols
# %%
train.columns
# %%
"""
Use this to write stuff instead of markdown.
Trade off between faster typing and no markdown? I choose no markdown

### Basic Exploration
1. 891 rows, 12 columns ie 891 people, 12 variables about each person

2. Columns Looking at the Kaggle Site
    1. Survived" column, 1 = Yes, 0 = No
    2. Pclass: ticket class, 1 = 1st, 2 = 2nd, 3 = 3rd
    3. Sex: male or female
    4. Age: in years
    5. Sibsp: # of siblings or spouses aboard the Titanic
    6. Parch: # of parents or children aboard the Titanic
    7. Ticket: ticket number
    8. Fare: passenger fare
    9. Cabin: cabin number
    10. Embarked: port of embarkation, C = Cherbourg, Q = Queenston, S = Southampton
    11. PassengerId
    12. Name
"""

# %%
import missingno
# %%
missingno.matrix(train)
# %%
train.describe()
# age has 714 / 891
train.Cabin.describe() # 204 / 891
# %%
train.isnull().sum()
# 177 na for Age
# 687 for Cabin
# 2 for embarked

# %%
## gender_submission EDA
gender_submission
# %%
"""
this is how the submission is supposed to look like
its called gender because only the females are assumed to have survived; might make sense: when there is an emergency, they usually save the women and children first.
"""

# %%
test
# 418 rows, 11 cols
# only 11 because survived is not included for test set!
# %%
""" DATA ANALYSIS 
DBourke splits analysis into 2 portions
1. Discrete (or discretized) variables
- objects are usually categorical. In our eg, [Name, Sex, Ticket, Cabin, Embarked] are objects
- ints are usually categorical too

2. continuous variables
- usually floats

"""
train.dtypes

# %%
### creating dataframes first before filtering out
df_bin = pd.DataFrame() # for discretised continuous variables
df_con = pd.DataFrame() # for continuous variables

# %%
import seaborn as sns 
import matplotlib.pyplot as plt
# %%
### VAR 1: SURVIVED ### 
fig = plt.figure(figsize = (20,1))
sns.countplot(y = 'Survived', data= train)
print(train['Survived'].value_counts())

## 342 survived, 549 died
# %%

### add this var to both of our empty dataframes
df_bin['Survived'] = train['Survived']
df_con['Survived'] = train['Survived']
df_bin.head()

# %%
### VAR 2: PCLASS ### 
### Key: 1 = 1st class... 3 = 3rd class
sns.set()
sns.displot(train['Pclass']) # distribution plot!
print('----------counting the categories----------')
print(train.Pclass.value_counts())

print('----------checking for null values----------')
print(train.Pclass.isnull().sum())


# %%
### add to our sub dataframes
df_bin['Pclass'] = train['Pclass']
df_con['Pclass'] = train['Pclass']

# %%
### FEATURE: NAME ###
train.Name.value_counts() # lists all the names and their counts. 891 rows means that everyone has different names!


# %%
"""
1. we can shorten the names by removing the mister or mrs
2. we wont be using the names for this because theres too many unique values. Also, its unlikely that names affect whether u survived (unless ofc you are some famous person?)
"""

# %%
### FEATURE: SEX ###
train.Sex.value_counts()
sns.barplot(y='Sex', x='Survived', data=train)
# %%


# %%
