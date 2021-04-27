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
sns.countplot(y= 'Sex', data = train)
# %%
# no null values
train.Sex.isnull().sum()


# %%
# overlapping bar chart
sns.displot(
    data=train, x="Sex", hue="Survived", alpha=.6, height=6
)

# %%
### trying to plot a grouped bar chart
train.Survived.value_counts()

# %%
### USE A DISTRIBUTION PLOT
"""
of those who died, more were male
of those who survived, more were female
"""
sns.displot(train, x="Survived", hue="Sex", multiple="dodge")

# %%
# add Sex to the subset dataframes
df_bin['Sex'] = train['Sex']
df_bin['Sex'] = np.where(df_bin['Sex'] == 'female', 1, 0) # change sex to 0 for male and 1 for female

df_con['Sex'] = train['Sex']
# %%
df_bin

# %%
# How does the Sex variable look compared to Survival?
# We can see this because they're both binarys.
fig = plt.figure(figsize=(10, 10))
sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'], kde_kws={'label': 'Survived'})
sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'], kde_kws={'label': 'Did not survive'})

# %%
### FEATURE: AGE ###
train.Age.isnull().sum() # 177 / 891 null vals
missingno.matrix(train)
# q. how would i fill up these empty values?
# a. i would get the average age and then fill them with that

# %%
train.Age.describe() # average is 29.699


# %%
x = train
x.Age.fillna(30.0, inplace = True)

# %%
# count has increased to 891
x.Age.describe()

# %%
### creating a function that will plot the graphs we need ###
def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20,5), use_bin_df = False):
    """
    function that plot counts and distribution of a label variable and target var side by side
    """
    if use_bin_df:
    
    else:




# %%
train.SibSp.value_counts()
"""
the frequency of the number of siblings and spouses each person had in the ship
1. majority had 0 siblings or spouses with them on the ship
2. the second biggest group only had 1; maybe couples.
"""
sns.displot(train, x='SibSp')




# %%
"""
1. from the graph, we can see that if you have 1 SibSp, you were more likely to survive as compared to other groups
2. we should try to find the proportion: how to aggregate?
"""
sns.displot(train, x='SibSp', hue='Survived', multiple = "dodge")
