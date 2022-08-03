# Ganesh Kale
### Python | Data Science | ML | NLP | Deep Learning

I am senior data scientist with over 14+ years of experience using predictive modeling, data processing 
and data mining in various fields. I worked as Data engineer, data analyst and data scientist in banking and financial service organizations.
My major experience is in building natural language processing solutions for banking and financial products with the help of machine learning and deep learning techniques.
I have experience in managing high-performing teams in fast-paced organizations. leveraging data science skills to solve challenging business problems.
 
I have completed my masters in mathematics from Pune University India and masters in data science from Bellevue university NE USA. 

# Contact:

email: ganeshbkale@gmail.com

# Professional Summary:

## [Project 1: News Categorization](https://github.com/gkdsc2020/dsc550_data_mining/blob/main/week10_final_project_GaneshKale.ipynb):

##### Background: 

For long time, this process of categorization news was done manually by people and used to allot news to respective section(category). With digitalization of news paper, the news gets updated every moment and allocating to them to appropriate category can be cumbersome task.

**How this problem was solved**: To avoid manual news categorization, with help of latest technology, Natural Language Processing and Machine Leanring, this problem is tackled to classify and predict which category a piece of news will fall into based on the news headline and short description.

**How this model would help news editors and customers?**

A machine learning model is built using supervised machine learning techniques, that learns from existing news headlines and short description and predict the news category appropriately. With the help of this model the news categorization can be automated and it would save manual work and help users to read the news of their interest in right section.

##### Modeling: 

Following machine learning classification algorithms were used to train and evaluate model.

- Naïve Bayes Classifier 
- Linear Support Vector Machine
- Logistic Regression
- XGBoost Classifier

##### Conclusion: 

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework.
This model provided highes accuracy score among all of the three and is finalized as baseline model for this project.

#### News Categorization Distribution:

![image](https://user-images.githubusercontent.com/70597635/180676600-2740d5f9-e934-446b-82f8-4f2bbc8fe9ef.png)



## [Project 2: Walmart Sales Forecasting](https://github.com/gkdsc2020/dsc630_predictive_analytics/blob/main/dsc630_final_project.ipynb):

##### Background: 

Sales Forecasting is the process of using a company’s sales records over the past few years to predict the short-term or long-term sales performance of the company in the future. Sales forecasting is a globally conducted corporate practice where number of objectives are identified, action-plans are chalked out as well as budgets and resources are allotted to them.
Business Sales Executives often find themselves scrambling for answers when it comes to sales forecasting during business reviews. The Sales Forecast Model will help sales executives to find such answers upfront and be ready with numbers and predictions to share with leaderships team. 

#### Problem Statement:

The goal of this analysis is to predict future sales for the Walmart stores based on the varying features and events.
•	Build the Machine Learning model that would learn from past records, events and predict the accurate outcomes.
•	Predict the Sales forecast for Store and its departments on specific week of the year considering if it is before holiday or after holiday.

**How this model would help?**

The baselined model which produced best accuracy score can be used to predict the sales forecast for any given store and its departments. The model can be deployed to production system and simple application can be built to predict sales.

##### Modeling: 

These ML algorithms are used to train the model and evaluated using Weighted mean absolute error. The model with lowest RMSE score and best accuracy score is baselined. Following are the list of ML algorithms are used to train the model – 
•	KNN Regression	
•	Decision Tree
•	Random Forest
•	Gradient Boosting Machine (XGBoost Regressor)
•	ARIMA - Auto Regressive Integrated Moving Average

#### Result:
<img width="543" alt="image" src="https://user-images.githubusercontent.com/70597635/182530011-c271a212-fc24-4d65-a9a6-3eef983ea973.png">

##### Conclusion: 

Comparing the accuracy of different models, it turns out that XGBoost regressor with accuracy score 97.7% and Root Mean Squared Error 3463 is the best model for this project and is baselined. The model can be deployed to production system and simple application can be built to predict sales. The Application would accept values such store number, department number, week of the year, size of the store, is holiday in the week, average temperature, unemployment rate in that week, fuel price etc. and based on these values it would predict the sales value for that store and departments.

#### Feature Correlations:

![image](https://user-images.githubusercontent.com/70597635/182530545-e78b614a-074e-40f7-845f-958f03c5d254.png)

