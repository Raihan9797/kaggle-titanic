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

22. (Video)[https://www.youtube.com/watch?v=9yl6-HEY7_s] using One Hot Encoder

23. (Video)[https://www.youtube.com/watch?v=irHhDMbw3xo] the benefits of One Hot Encoder, along with column_transformer and pipeline.
- Using this one hot encoder, we are able to use the column_transfrom to encode specific columns only.
- Using the column_transform, we are able to use this in a pipeline so you dont have to keep preprocessing new input!!! Just re-use the pipeline! Because it took in the initial one hot encoder, it will keep the correct categories
    - Eg. Your training set has M and F, but in your test set, there is only F. If you pd.get_dummies(test_set), it will only think the F exists! Using the OneHotEncoder, this will not happen!

24. Current downside of using OneHotEncoder: it is converted to an array
- technically is fine, but not human readable because there are no columns to read from

25. Basic splitting of train and test data
X_train = df.drop('label_col', axis = 1)
y_train = df.label_col

26. Using sklearn algos
- Very easy, just fit
- use model.score(x_train, y_train) to get the accuracy of the model, BUT ONLY TRAINED ONCE.
- can use model_selection.cross_val_predict() to get y_pred using cross validation
- can use metrics.accuracy_score to get cross validated accuracy
- overall work ok even without any parameter tuning!


27. timing functions
> start_time = time.time()
> time_diff = time.time() - start_time

28. Using Catboost: open source gradient boosting library
- Issues with ipywidget: can't display the graph of logloss as we train the model on vscode. Changed to jupyterlab, still doesnt work. Turns out, jupyter notebook != jupter lab. Used the classic notebook to run.
> python -m jupyter notebook

29. Precision vs recall
- (how to remember)[https://towardsdatascience.com/finally-remember-precision-and-recall-94b4d481f9bf]

30. y_true and y_pred for sklearn
- In all sklearn libraries, **make sure to put y_true then y_pred**
> print(classification_report(y_train , y_pred))

31. Important classification metrics
> from sklearn.metrics import classification_report, confusion_matrix

32. The sklearn confusion matrix
- Assuming that you follow the input correctly (y_true, y_pred), the confusion matrix will create an array in which the column names are the predicted values, while the rows (aka index) are the true values. To make it explicit, you can create a dataframe and rename the labels like this:
```
pd.DataFrame(confusion_matrix(y_train, gbt_pred),
          columns = ['Predicted: 0', 'Predicted: 1'],
           index = ['Acutal: 0', "Actual: 1"])
```

33. This tutorial didnt actually cover train-test split. Fortunately, you already know how to do this (because its a function in sklearn) so we can skip this step. 

34. Submitting your predictions: make sure your test set has been preprocessed the same way!!


# Possible Extensions (from the last part of my jupyter notebook)
Some reccomendations by DBourke, and my own reflections

1. The Age feature
- All I did was just use the mean() to fill up the age. Just by doing that I was able to get Age to the 2nd highest feature importance.
- DBourke suggest reading up on the interpolate() function of Pandas.
- Another way I saw on Youtube, was to basically use ML model that would fill up the values. An example of this could be kmeans clustering, those with similar attributes could be given similar Ages in this case.

2. What to do with Name.
Initially, i thought we should just remove, the Mr. Mrs etc. But DBourke actually recommends focusing on those! My reasoning for this was that Mr and Mrs are already in Sex.
- But other prefixes could come up: Titles like Dr. etc that might affect how a person is likely to be saved!!

3. Cabin feature
- Is there a way to see if they had a cabin or not? DBourke didnt answer this but i guess using the Pclass? and ticket number?

4. Combine SibSp and Parch to see if the person was alone.
Actually a very good idea which I did think about but never articulated. I would think those that are not alone might actually have less chance of surviving? If you are not alone, one of you is more likely to survive imo.

5. Hyperparameter tuning.
- Dbourke reccomends hyperopt library. We could also use GridSearch and Randomized search iirc.
- https://github.com/hyperopt/hyperopt

6. As we have seen when trying to do the submission, it was actually quite annoying to do all the preprocessing steps again for the test set. Should really consider using the pipeline feature on sklearn.