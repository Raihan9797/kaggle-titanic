# Kaggle Titanic dataset
Doing this to recap how to do Data Analysis. Learning:
1. EDA: Exploratory Data Analysis
- Data Visualisation
- Summary Statistics

## Useful Links
1. [Kaggle Link](https://www.kaggle.com/c/titanic/data)
2. [Daniel Bourke Tutorial](https://www.youtube.com/watch?v=f1y9wDDxWnA)
3. [Ken Jee Tutorial](https://www.youtube.com/watch?v=I3FBJdiExcg)


# Things I've learnt:
1. VSCode Interactive Python is another alternative to Jupyter Notebook. Easier to code but no markdown which is an okay tradeoff esp since I use Vim

2. Checking missing values
- train.describe(); use the count() and compare between the number of rows to see the number of missing values for each column
- missingno.matrix(df) will be a visual representation of values that are available. Its quite good
- train.isnull().sum()  # return a count of null vals
- df['Colname'].value_counts() # returns counts of the column

3. Visualizing data
- seaborn is just another wrapper for matplotlib
- plotly is more interactive and nicer but it seaborn is supposed to be faster
- how to do grouped bar chart? Use a distribution plot, multiple = 'dodge'
sns.displot(train, x="Survived", hue="Sex", multiple="dodge")

4. Transforming Data
    1. How to get value_counts for grouped data? Need to use unstack() or pivot_table(). Need to search more on that
    2. Change data types from int to bool as categorical (eg. Survived)
    3. Using df.filter(), groupby() 


5. Creating functions to graph data:
- Especially useful for EDA. As we go through each feature/col, we are basically repeating a lot of steps. Creating function to do this will help makes things a lot faster

6. The reason we create empty dataframes, df_bin, df_con:
- In cases where you already know what you want, this might NOT be useful; all you have to do is recreate a df with the columns u want
- eg. new_df = old_df[['columns u want']]
- In this example where we look through all the columns, its better to start from zero, and add columns as we EDA!

7. distplot() is deprecated, use displot() instead. Why doesnt the labels show?

8. Using Jupyter notebook
    1. Getting extremely hard to navigate the interactive notebook
    2. Installing jupyternotebook
        pip install jupyterlab
        jupyter-lab
    3. Navigate to the folder you want, use cmd and then run the above commands

9. getting jupyter working dir
import os
os.getcwd()


10. Changing numeric 1/0 to categorical yes/no for df columns using map()
> train['s2'] = train['Survived'].apply(lambda x: 'yes' if x==1 else 'no')
> train['s2'] = train['Survived'].map({True:'yes', False:'no'})
- 1 and 0 are basically True and False which is why the mapping works


11. Why we might NOT want to change 1/0 to a categorical yes/no
- in the dbourke notebook, he changes the Sex from male/female to 1/0. Im guessing its because it will be easier in the long run for a machine learning model to process

12. Visualizing types of data
1. Categorical: distribution plot, countplot
2. Numeric: histogram, boxplot, violint plot
    boxplots don't allow u to see accurately if the distribution is bimodal! vs violin and histogram

13. you should not value_counts() for continuous data because it could be alot and is not meaningful. Better to count in buckets like when u use a histogram!


14. replacing null vals in a column
* if the col is of type int, be sure to change it back
> train['Age'] = train['Age'].fillna(value= 30)
> train['Age'] = train['Age'].astype(int)



15. converting continuous numeric column to discretized values
> df_bin['Age'] = pd.cut(train['Age'], 10) # bucketed/binned into different categories


16. Seaborn distplot is deprecated; use displot() or histplot(). Read through the docs and understand how to replicated the graphs shown!
- The kde doesnt really match what is shown though?


17. Subplotting: using the new displot() creates an empty subplot for no reason
- updated version
```
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize = figsize)
        sns.countplot( y=target_column, data=data, ax = axs[0])
        sns.histplot(data=train,
                    x = target_column,
                    kde=True,
                    multiple='dodge',
                    hue= label_column,
                    ax = axs[1],
                    stat = 'count'
                    )
```
- still different: the density is not the same; and the kde graph doesnt start the same


18. Jupyter lab autocompletion not working
> %config Completer.use_jedi = False
- after that press <tab> to auto complete; won't be able to intellisense for u.
> !pip3 install jedi==0.17.2
- install this version instead; for me, doesnt intellisense


19. Drop na rows
> df_con = df_con.dropna(subset = ['Embarked'])

20. Feature Encoding: One Hot Encoding.
> df_bin_enc = pd.get_dummies(df_bin, columns=one_hot_cols)
- Using pd.get_dummies to basically get one hot encoding

- DBourke changes from Label Encoder to OneHotEncoder because Label Encoder might implicitly create some kind of ordering even when there isn't any. Check this (example)[https://contactsunny.medium.com/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621]

21. OneHotEncoder vs pd.get_dummies:
- Conclusion: use OneHotEncoder if you are going to do ML. Read (this)[https://stackoverflow.com/questions/36631163/what-are-the-pros-and-cons-between-get-dummies-pandas-and-onehotencoder-sciki]