## Classification with Python
### Purpose:
This project is meant for demonstrating data classification skills with Python in Jupyter Notebook. The example dataset to be used contains information on properties of wines (feature variables) and the grape variety (target variable). The aim is to investigate the data and to utilize feature variables to find the most accurate machine learning model to predict the classifications of the grape variety. The fitted model could then be given new wine data and be used to predict grape varieties. (Note: no wine expertise, dataset topic only for project convenience and technical purposes.)
### Activities:
The libraries used are Pandas, Matplotlib, and Seaborn for investigating and visualizing the data, SciPy for testing statistical significance, as well as Scikit-learn for the machine learning models.

While the dataset is available in .data format from the original source, it was converted into .xlsx using Excel's Power Query Editor due to easier understanding before beginning the analysis with Python. Column headers were also created for the dataset while converting it, according to the wine properties listed in the original source.

Choosing feature variables was done based on correlation coefficients, primarily in the context of the target variable while also taking into consideration the correlation among the feature variables themselves to avoid model confusion.

A few iterations of testing different variables and model parameters was conducted to reach the best accuracy levels for the models. The final selections are presented and reasoned in the analysis, along with the accuracy results as well as further presentation of the most accurate model's performance with the data.
