# Predicting Customer Churn

## Overview
The primary objective of this project is to help the business reduce customer churn by identifying key factors that contribute to customers leaving the service. By analyzing customer data, we aim to provide actionable insights and recommendations that will enhance customer retention strategies, thereby fostering a more stable and loyal customer base. The business is facing challenges with customer retention, leading to increased costs and lost revenue. Understanding the reasons behind customer churn is crucial for developing strategies that improve customer satisfaction and loyalty. The process involves rigorous model evaluation to ensure robustness and reliability, ultimately supporting data-driven decision-making to reduce customer attrition and enhance long-term customer loyalty.

## Business Understanding

### Problem Statement
The business is facing challenges with customer retention, leading to increased costs and lost revenue. Understanding the reasons behind customer churn is crucial for developing strategies that improve customer satisfaction and loyalty
### Objective
The primary objective of this project is to help the business reduce customer churn by identifying key factors that contribute to customers leaving the service by developing a robust predictive model that accurately identifies customers at risk of churning, utilizing data preprocessing, feature engineering, and advanced machine learning techniques performance
### Stakeholders
* Customer Retention Team: Focused on understanding churn patterns to develop effective retention strategies.
* Executives: Interested in maintaining overall business performance and revenue stability.
* Customer Service Managers: Aim to maintain high levels of customer satisfaction and address issues that may lead to churn.

## Data Understanding

### Dataset
The dataset used in this project is sourced from Kaggle and provides comprehensive customer information. This dataset includes a variety of attributes related to customer demographics, service usage patterns, and customer support interactions, alongside indicators of customer churn. The customer churn dataset used in this project contains 3,333 records of customer information, focusing on attributes that may influence their likelihood to churn (leave the service). The dataset includes various features such as demographic details, service usage patterns, and customer service interactions. It is split into features like `State`, `Account Length`, `International Plan`, and `Churn`, with the target variable being whether the customer has churned. This dataset serves as the foundation for training and evaluating machine learning models aimed at predicting customer churn, enabling businesses to proactively address potential customer losses. We organize and process this dataset using dataframes to ensure efficient handling and analysis.

### Data Analysis
We first explored the dataset to understand its structure and contents. This involved examining descriptive statistics, checking for missing values, and identifying any outliers. Next, we conducted visualizations to gain insights into the distribution of key features, correlations between variables, and the proportion of churned customers. We also performed feature engineering to create new variables or transform existing ones to enhance the predictive power of our models. Additionally, we conducted statistical tests or exploratory data analysis to identify significant factors associated with churn. The data analysis phase aimed to uncover patterns and relationships within the data that could help us better understand customer churn behavior

### Data Cleaning
The data preparation process involved preliminary data cleaning to ensure the dataset's quality. The remaining tasks include encoding categorical variables, such as the presence of an international plan, into a numerical format suitable for machine learning models, and addressing class imbalance by applying techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the target variable's classes, ensuring the model performs well on both majority and minority classes.

## Visualizations
### Class imbalance
![image]()
The countplot visualizes the distribution of the target variable "Churn" in the SyriaTel Customer Churn dataset. The plot shows the number of customers who churned (denoted by 'True') and those who did not churn (denoted by 'False'). From the graph, it's evident that the dataset contains more instances of customers who did not churn compared to those who churned. This indicates an imbalance in the dataset

### Distribution of numerical features
![image]()
Each histogram provides insights into the distribution of a particular numerical feature, showing the frequency of values along the x-axis and the corresponding frequency density along the y-axis. This visualization helps in understanding the range, central tendency, and spread of each numerical feature in the dataset.

### Analyzing the relationship between numerical features and the target variable
![image]()
The plot visualizes the relationship between numerical features and the target variable 'churn' using boxplots. Each subplot represents a different numerical feature, and the boxplot illustrates the distribution of that feature's values across the two categories of the target variable: churn and non-churn. The x-axis denotes the target variable (churn), and the y-axis represents the values of the numerical feature. The boxplot shows the distribution of the feature's values within each category of the target variable, including the median (line inside the box), interquartile range (box), and outliers (points beyond the whiskers). By comparing the boxplots across different numerical features, we can identify potential relationships or differences in the distributions of these features between churn and non-churn groups, which can provide insights into the predictive power of these features for determining churn.

### Pairplot
![image]()
The pairplot provides a visual overview of the relationships between different numerical features in the dataset. It shows scatter plots for each pair of variables, allowing us to observe potential correlations or patterns. Diagonal elements typically display the distribution of each feature using histograms or kernel density estimates.

### Distribution of Categorical features
![image]()
The aforementioned illustrates how categorical attributes, such as state, international plan, and voice mail plan, differ throughout the evaluation variable.

## Modelling
### Models used
Logistic Regression was the first model that was used. We trained the Logistic Regression model with a random state of 42, made predictions, and printed out the results. The accuracy of the Logistic Regression model on the test set was printed, followed by the classification report and confusion matrix. Next, we trained a Decision Tree Classifier with a random state of 42, made predictions, and printed out the evaluation results. The accuracy of the Decision Tree model on the test set was printed, followed by the classification report and confusion matrix. Finally, we trained a Random Forest Classifier and printed out the outputs.

### Applying SMOTE
To address the class imbalance issue, the Synthetic Minority Over-sampling Technique (SMOTE) was applied. SMOTE generates synthetic samples for the minority class, balancing the distribution of classes in the dataset.

### Hyperparameter Tuning
GridSearchCV was employed to find the optimal hyperparameters for the Random Forest Classifier. This process systematically searches through a grid of hyperparameters, using cross-validation to determine the best combination that maximizes model performance

## Evaluation
The evaluation of the model's performance included various metrics such as Accuracy, Precision, Recall, and F1-Score. These metrics provide insights into different aspects of the model's predictive capabilities. To compare the three models, Logistic Regression, Decision Tree, and Random Forest, we first evaluated their performance metrics on the test set. Logistic Regression achieved an accuracy of 85.5%, demonstrating good precision (87%) for the "False" class but lower recall (19%) and F1-score (28%) for the "True" class. Decision Tree outperformed with an accuracy of 94.3%, showing balanced precision (95%) and recall (98%) for the "False" class and acceptable precision (88%) but lower recall (72%) for the "True" class. Random Forest attained an accuracy of 89.8%, with a strong precision (89%) and perfect recall (100%) for the "False" class, but lower recall (34%) and F1-score (50%) for the "True" class. The confusion matrices also highlight the model performances We improved the random forest model using hyper parameter tuning where it showed a significant increase in its accuracy to predict customer churn. We got an accuracy of 94.3% and a best score of 95.4% which improved our model’s ability to predict the probability of a customer to stop using a company’s product due to various reasons.

![image]()
The confusion matrix shows that our model has high accuracy (94.01%) and precision (98.41%), indicating that it is effective in predicting non-churn customers and is usually correct when it predicts churn. However, the recall (61.39%) is relatively low, meaning that the model misses a significant number of actual churners. The F1-Score, which balances precision and recall, is 75.56%. To improve the model's performance, especially in identifying more actual churners, further tuning of the model parameters or adjusting the decision threshold might be necessary.

![image]()
* The bar graph represents the feature importances from the tuned Random Forest model used for predicting customer churn. Each bar corresponds to a different feature from the dataset, with the height indicating the importance of that feature in making predictions.
* Certain features such as 'total day charge', 'number of voice mail messages', 'total eve charge', 'international plan' and 'state' show higher importance, implying that they have a stronger influence on the model's predictions.
* This insight helps in understanding which factors are most critical in predicting customer churn, thereby guiding strategic decisions to mitigate churn.

![image]()
The ROC curve illustrates the model's performance across various threshold levels, with the true positive rate (sensitivity) plotted against the false positive rate (1-specificity). The Area Under the Curve (AUC) is 0.93, indicating a high level of predictive accuracy. An AUC of 0.93 suggests that the model is very effective at distinguishing between customers who churn and those who do not, as a value of 1 represents a perfect model and 0.5 represents a model with no discrimination ability.

## Recommendations
### Recommendations to the Model
We recommended the use of the tuned Random Forest model since it had the best accuracy in predicting the probability of customers churning. The best model achieved an impressive accuracy of 94.0% in its predictions, indicating strong overall performance. In the classification report, the model demonstrated excellent precision for both the majority class (0) and the minority class (1), with 94% precision for class 0 and 98% precision for class 1.

### Recommendations to Stakeholder
1. Syriatel should monitor high-usage customers for potential churn triggers and offer special plans or discounts to heavy users to encourage retention.
2. Improve customer service quality and responsiveness. Track frequent callers and ensure their issues are resolved promptly and satisfactorily.
3. Introduce attractive evening and off-peak hour plans or discounts to cater to users who predominantly use the service during these times.
4. Analyze and address state-specific customer behaviors and issues. Customize marketing and retention strategies to cater to regional preferences and needs.
5. Promote competitive international plans and ensure customers are aware of these options. Offer personalized discounts for frequent international callers.
