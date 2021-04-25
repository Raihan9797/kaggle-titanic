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