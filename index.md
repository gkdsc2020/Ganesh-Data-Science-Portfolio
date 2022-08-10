## Table of Contents:

 * Project 1: News Categorization
 * Project 2: Walmart Sales Forecasting
 * Project 3: Next Word Prediction-Language Model

{:toc}


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

 * Project 1: News Categorization
 * Project 2: Walmart Sales Forecasting
 * Project 3: Next Word Prediction-Language Model

{:toc}


## [Project 1: News Categorization](https://github.com/gkdsc2020/dsc550_data_mining/blob/main/week10_final_project_GaneshKale.ipynb):
 
##### Background: 

For long time, this process of categorization news was done manually by people and used to allot news to respective section(category). With              digitalization of news paper, the news gets updated every moment and allocating to them to appropriate category can be cumbersome task.
 

**How this problem was solved**: To avoid manual news categorization, with help of latest technology, Natural Language Processing and Machine           Leanring, this problem is tackled to classify and predict which category a piece of news will fall into based on the news headline and short             description.
 
 
**How this model would help news editors and customers?**

A machine learning model is built using supervised machine learning techniques, that learns from existing news headlines and short description and       predict the news category appropriately. With the help of this model the news categorization can be automated and it would save manual work and help     users to read the news of their interest in right section.
 
 
##### Modeling: 

Following machine learning classification algorithms were used to train and evaluate model.

 - Naïve Bayes Classifier 
 - Linear Support Vector Machine
 - Logistic Regression
 - XGBoost Classifier

##### Conclusion: 

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine              learning     algorithms under the Gradient Boosting framework.
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

- KNN Regression	
- Decision Tree
- Random Forest
- Gradient Boosting Machine (XGBoost Regressor)
- ARIMA - Auto Regressive Integrated Moving Average

#### Result:
<img width="543" alt="image" src="https://user-images.githubusercontent.com/70597635/182530011-c271a212-fc24-4d65-a9a6-3eef983ea973.png">

##### Conclusion: 

Comparing the accuracy of different models, it turns out that XGBoost regressor with accuracy score 97.7% and Root Mean Squared Error 3463 is the best model for this project and is baselined. The model can be deployed to production system and simple application can be built to predict sales. The Application would accept values such store number, department number, week of the year, size of the store, is holiday in the week, average temperature, unemployment rate in that week, fuel price etc. and based on these values it would predict the sales value for that store and departments.

#### Feature Correlations:

![image](https://user-images.githubusercontent.com/70597635/182530545-e78b614a-074e-40f7-845f-958f03c5d254.png)



## [Project 3: Next Word Prediction-Language Model](https://github.com/gkdsc2020/dsc680-applied-ds/blob/main/dsc680_project1_GaneshKale.pdf):

##### Background: 

Language modeling involves predicting the next word in a sequence given the sequence of words already present. A language model is a key element in many natural language processing models such resolving customers inquiries through chat or answering the questions through emails.
In customer Service business especially in messaging or Chats or email supports, customer representative often struggle to response fast if they have limited knowledge of business area wherein the inquiry is about and need to respond fast for better service and improved customer satisfaction.


#### Problem Statement:

Business Stakeholder wanted to build model that would learn from previously provided chat or email resolution history and suggest the next word when representative start providing resolution to the customers inquiry.

• Build the model that would predict the next word based on previous context.

##### Modeling: 

The model is built using Recurrent neural networks (RNN) and TensorFlow and Keras.

Text data is cleaned and transformed in vector form using sklearns vecorizer. 

The sequential neural network model in trained using LSTM(Long-Short Term memory) network. It is a variety of recurrent neural networks (RNNs) that are capable of learning long-term dependencies, especially in sequence prediction problems.

##### Conclusion: 

The model is predicting the next words based on given input sample text, the generate next word or series of words can be handled through argument as configuration and changed based on model performance.
The next word prediction language model is generating the relevant word and can be used in production to help agent type response fast.



## [Project 4: Transaction Categorization](https://github.com/gkdsc2020/dsc680-applied-ds/blob/main/project2_transaction_categorization_GaneshKale.ipynb):

##### Background: 

Transaction categorization is the ability to recognize the purpose of a transaction based on its description. For long, this process was done manually but now technology can do it efficiently.
This project is focused on Natural Language Processing using the power of Machine Learning to predict which category a transaction will fall into, given the description of the transaction.

#### Problem Statement:

Goal - Build a model to predict transaction categories using the 10 (ten) distinct categories that a transaction may fall into. The categories are as follows:

1. Communication Services
2. Education
3. Entertainment
4. Finance
5. Health and Community Service
6. Property and Business Services
7. Retail Trade
8. Services to Transport
9. Trade, Professional and Personal Services
10. Travel

##### Modeling: 

The model is built using NLP and Classification Machine Learnings.
The text data is converted into vectors using sklearns countvecotrizer and then SMOTE technique is used to baance the target classes.
Below are the list of classifications algorithms are used - 

- Naïve Bayes Classifier
- Support Vector Machine
- XGBoost Classifier

**Model Summary**:

Model       | Name	Accuracy Score

XGBoost	    | 69%

SVM	        | 69%

Naïve Bayes | 66%


##### Conclusion: 

After building and evaluating multiple classification models, we can see that the XGBoost has better accuracy which is 69% and since it uses gradient boosting it provides better result than other classifier models.

Based on this we can recommend XGBoost Classifier model as final model for our project to predict the transactions categories.

#### Transaction Category Distribution:

![image](https://user-images.githubusercontent.com/70597635/183001165-15214d9e-800b-4126-bb38-2f032cc41316.png)


## [Project 5: Credit Card Customers - Predict Churning](https://github.com/gkdsc2020/dsc530edapython/blob/main/dsc530_term_project.ipynb):

##### Background: 

The objective of this project was to perform the different statistical techniques as part of exploratory data analysis using python as programming language. A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction.


#### Problem Statement:

The objective of this data analysis project is to better understand the attributes that are making impact on customer's decision to leave the credit card company and after finalizing the attributes another goal is to identify such customers who are going to leave.

- Identify the key features/factors that driving customer to drop off.
- Identify the customers having high likelihood of attrition

**How to tackle Business Problem**

In order to tackle the business problem, following steps are performed:
- Exploratory Data Analysis - Explore the different features with help of visualization and statistical summary. This activity will provide more           insights about the it and we can choose the feature that could be influencing customers to drop off.
- Feature correlations - This activity will help us identify features which have more impact on customers decision.
- Modeling - Selecting right algorithm for model and evaluating the model result.

##### Exploratory Data Analysis:
As part of exploratory data analysis, different statistical techniques and are used to explore the data and different charts are used for visualization.

- Indentify missing values and Visualize missing values in the features
- Distribution of Numerical Variables - to see distribution and detect Outliers
- Outliers Detections in fatures using Box plots
- Display Unique and Duplicate values in feayures
- Data Transformations
- Compare CDFs
- Muliti-colinearity of Features


##### Modeling: 

The focus of this project more on Exploratory Data Analysis, so only Logistic Regression algorithm is used to predict the churn in customers.


##### Conclusion: 

The above stated business problems were tackled using exploratory data analysis and regression analysis. As part of exploratory data analysis, different statistical techniques and are used to explore the data and different charts are used for visualization.
The data validated for any missing or duplicate values, the data distribution is checked for any outliers and removed them appropriately. Different distributions are checked for few key variables such as PMF, CDF, analytical distribution such as logarithmic distribution, checked for the correlation, covariance between the variables, plotted scatter charts to see correlation visually. This exploratory analysis helped to identify features that are impacting customers decision to leave or stay with bank.
Finally, Regression analysis was performed using the key features to build the model to predict the customers who will drop-off and who will not.
This exploratory analysis helped understand the different features, their relationship and impact and regression analysis helped to build prediction model more accurately. 
Overall, with this analysis we were able to solve the business problem that was stated to identify features impacting customer to leave or stay and predicting churn.


#### CDFs of attrited customer Vs not attrited customers for Total Trans Counts:
![image](https://user-images.githubusercontent.com/70597635/183262722-baa55f61-f130-4aae-a5db-88d4d341e5b5.png)


## [Project 6: Data Prep and EDA for Housing Price Prediction](https://github.com/gkdsc2020/dsc540_data_preperation/blob/main/Final_Project.ipynb):

##### Background: 

House Prices in USA are booming, and house prices will continue to race ahead, at nearly twice the pace predicted before this year. This is what we hear or read when talk about housing market in USA.
To make data ready by collecting from different sources and cleaning, transforming, and merging all these data to make it final dataset in the ready format for machine learning algorithms so predicting house price model can be developed by training and validating on housing market dataset. To build predicting house price model, we should consider different factors such as house information, and facilities available in neighborhood such school, hospitals etc. 

#### Data Sources:

As part of this project, Austin, Texas housing market data is collected from 3 difference sources such as csv file with house related information, json file for hospital in neighborhood and tabular data for school information from website.

- **CSV File**: Austin TX area house information, such as address, year built, sale price, No. of bedroom, bathrooms, lot area etc.
- **Tabular Data**: The neighborhood school information will be pulled from below websites, since the website has school information such as address,     type of school, student per teacher ratio, ratings and percentile, staar awards ratings etc. but we dont need grade level rating so we are going to     remove them and clean data and at last will be joined with main data set based in zipcode value.
- **Json File**: The neighborhood public health locations information is pulled from data.gov using API. The data is in json format, and it will have     public health locations information such as facility name, address, hours of operation, website etc. 


##### Conclusion: 

As part of this project, I have implemented all the data prep and EDA techniques to make final dataset in ready format so that it will be passed to ML algorithms to make house price prediction model. There are few features kept in the data set such as urls, owner names, address for information purpose and can be ignored before feeding to ML algorithms.


#### Mulit-colinearity of Features:

![image](https://user-images.githubusercontent.com/70597635/183263251-d625eeef-3701-495c-b70f-898b66d31ce8.png)


## [Project 7: Data Visualization]():

##### Background: 

Airline travel is most preferred one but with recent accidents, Media made hype about it and started bad publicity and tried prove that air travel is not safest way of travel. As airline compay employee, stakeholders and Leadership goup asked to perform analysis on this area and provide different kind of visualization to prove or disprove this statement made by media.

#### Problem Statement:

As media raised concern over air travel safety, as employee of airline company need to solve this problem thru visualization - 

- Is air travel safe?
- Are there any impact on airline profits due to bad publicity?

#### Tableau Dashboard:

The visualization dashboard was built for internal communication purpose to provide the facts about the airline travel is safe and there is no adverse impact because of negative publicity made by media that it is dangerous to travel.
This dashboard considers the historical data of worldwide airlines and fatal incidents, fatalities happened specific to airlines and in the year, also captured financial data to show that there is not impact on overall profit and public demand to airline travel.

![image](https://user-images.githubusercontent.com/70597635/183280181-658cd846-7a05-4b60-9d9d-2b4a98723a3e.png)

##### [Blog Post](link):

The blog post is written with the help of data and charts those were presented in Dashboard. Since blog is for external audience in short for public so I made sure it gives some background about the research performed and what is the result of research and how we resolve the conflict of bad publicity by sharing the realistic data and figures.

##### [Infographics](https://github.com/gkdsc2020/dsc640_presentation_viz/blob/main/week9_10_Air%20Travel%20Safety%20Infographics.pdf):

Infographics are used to convey complicated data in a simple visual format. They are also visual tools to tell stories. Visual information graphics help people understand information quickly and more accurately. Since this topic was to convey message to public that airline is still safest way of travel, the infographics would provide them what them need to know from it.

 
##### Conclusion:

Based on the analysis, we were able to clarify two things are:

- Airline travel is still safest way of travel.
- There is no impact on net profit and public demand of air travel is increasing and will improve sales.


## [Project 8: R - Predict House Prices in Austin TX](https://github.com/gkdsc2020/dsc520statfords/blob/main/dsc520_final_project.Rmd):

##### Background: 
House Prices in USA are booming and house prices will continue to race ahead, at nearly twice the pace predicted before this year. Buying house is very critical job, one should aware of lots of things before buying house and when buying house nobody sure about when is the right time to buy house and wants to have some tool that would consider all the factors determining house price and predict the house price. Predicting the house price is challenging but doable and with help of machine learning algorithms this can be achieved. 


#### Problem Statement:

To predict the house prices from given house features and area features, the main business problems are:

- Identify the features that impact house Price.
- Predict the house prices based on identified features.


##### Modeling: 

R provides comprehensive support for multiple linear regression, Multiple regression is an extension of linear regression into relationship between more than two variables. In simple linear relation we have one predictor and one response variable, but in multiple regression we have more than one predictor variable and one response variable.

**lm()** function creates the relationship model between the predictor and the response variable.


##### Conclusion: 

Overall, by performing the exploratory data analysis and regression analysis, we were able to handle the problem statement mentioned above. With the help of exploratory data analysis the first problem statement was resolved to identify the features that are impacting the overall house prices. Regression analysis helped to tackle second question wherein we need to predict the house price based on given features of house and property area. The Multiple Linear Regression model built to tackle second question on predicting house price and based on the result summary the model we built is statistically significant and predict house price.
After all, the above performed analysis on house price prediction helped to solve both of the business problems.

#### House Price Distribution:

![image](https://user-images.githubusercontent.com/70597635/183263841-34a68ff2-7f5c-4baa-bfb3-48e244c553de.png)




## [Project 9: Weather App - Python](https://github.com/gkdsc2020/dsc510-intro-python/blob/main/week12_final_project.py):

##### Background: 

As part python programming, in this project weather app is developed using python language. This app to demonstrate the python language skill and how to use python features to read API, handle run time exceptions and make app user friendly.

This app gets user input as zip code or city name and brings the weather information by calling weather API and shares the requested weather information to user in appropriate format that user can understand it easily.



## [Project 10: eCommerce Future-Google Trend-Research](https://github.com/gkdsc2020/dsc500-intro-ds/blob/main/DSC_500_Project.pdf):

##### Background: 

E-Commerce or electronic commerce is the buying and selling of goods or services on the internet. The term covers a huge range of business processes, from payment processing to shipping and data management. From mobile shopping to online payment encryption and beyond, e-commerce encompasses a wide variety of data, systems, and tools for both online buyers and sellers. There are several e-Commerce models but we are going to focus on Business-to-Consumer (B2C) model.

To understand the Data points to prove theory that e-commerce business will grow in future we have used Google Trend analysis data and few online articles supporting the same, this how we have gathered data for analyzing this problem.


#### Problem Statement:

The goal of this project is to implement the Data Science methodologies to find the answer for the problem statement -

**What is the future of e-commerce and will it continue to grow in future?**


##### Conclusion: 

Ecommerce is an ever-expanding world. With intensifying purchasing power of global consumers, the proliferation of social media users, and the continuously progressing infrastructure and technology, the future of e-Commerce in 2020 and beyond is still more vibrant as ever.
We can conclude this analysis by saying the future of e-Commerce is bright and shine and it will continue to grow fast with this market trend in the future.

#### Google Trend – Google pay vs Apple pay:

![image](https://user-images.githubusercontent.com/70597635/183814789-5099c4d0-551e-4073-a61d-39c24177edff.png)


