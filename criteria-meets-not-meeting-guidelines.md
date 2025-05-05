
This file contains guideline with Criteria, Meets Expectations and Does Not Meet Expectations to eveluate how this problem statement is attempted and demonstrated:

Criteria: Data Cleaning	
Meets Expectations: 
Missing values are handled correctly
Redundant values within categorical columns are correctly identified and handled (if any)
Redundant columns are dropped correctly
DataTypes have been corrected for columns with incorrect DataTypes

 
Does Not Meet Expectations:
Missing values are handled inadequately
Redundant values within categorical columns are incorrectly identified and handled (if any)
Redundant columns are not dropped correctly
DataTypes have not been corrected or have been incorrectly assigned for columns with incorrect DataTypes

Criteria: Train Validation Split	
Meets Expectations: 
Feature and target variables are defined correctly
Data is split into training and validation sets, maintaining a ratio of 70:30

Does Not Meet Expectations:
Feature and target variables are not defined or are defined incorrectly
Data splitting is implemented incorrectly or not implemented at all

Criteria: EDA on Training Data	
Meets Expectations: 
Performed univariate analysis by visualising the distribution of all numerical columns
Performed correlation analysis by visualising a heatmap of the correlation matrix
Visualised class distribution of the target variable
Performed bivariate analysis by visualising the relationship between categorical columns and the target variable

Does Not Meet Expectations:
Failed to plot the distributions of numerical columns or created incomplete or inaccurate plots without meaningful insights
Incorrectly visualised the heatmap or failed to interpret correlations
Failed to visualise the class distribution of the target variable
Failed to visualise the influence of categorical variables on the target variable. Provided incomplete or unclear visualisations and insights

Criteria: Feature Engineering	
Meets Expectations: 
Performed resampling to handle class imbalance
Created new feature(s) from existing features to enhance the model’s ability to capture patterns
Handled redundant columns that may be redundant or contribute minimal information towards prediction
Identified and combined low-frequency values in categorical columns to reduce sparsity and improve model generalisation
Created dummy variables for independent and dependent columns in both training and validation sets
Applied feature scaling to numerical columns effectively by scaling the features
 

Does Not Meet Expectations:
Failed to perform resampling to handle class imbalance
Failed to create new feature(s) from existing features to enhance the model’s ability to capture patterns
Failed to handle redundant  columns that may be redundant or contribute minimal information towards prediction
Failed to identify and combine low frequency values in categorical columns to reduce sparsity and improve model generalisation
Failed to identify or create appropriate dummy variables for independent and dependent columns in both training and validation sets
Failed to apply or incorrectly performed feature scaling, leading to inconsistent data ranges


Criteria: Model Building	
Meets Expectations: 
Selected the most important features using Recursive Feature Elimination Cross Validation (RFECV)
Built a logistic regression model using the selected features, evaluated multicollinearity with p-values and VIFs, made predictions and assessed model performance
Found the optimal cutoff by plotting the ROC curve, visualising trade-offs between sensitivity and specificity, precision and recall and evaluated the final prediction with the optimal cutoff
Built a random forest model, made predictions and assessed model performance
Tuned the random forest model using appropriate technique, optimised hyperparameters, made predictions and assessed performance with relevant metrics
 

Does Not Meet Expectations:
Failed to select appropriate features using Recursive Feature Elimination Cross Validation (RFECV)
Failed to correctly build the logistic regression model using selected features and evaluated or interpreted the performance metrics incorrectly
Failed to identify the optimal cutoff or omitted key evaluations such as plotting curves and calculated necessary performance metrics
Failed to correctly build the random forest model, did not apply appropriate evaluation metrics or misinterpreted the results
Failed to tune the random forest model effectively, did not optimise hyperparameters or misinterpreted the performance evaluation


Criteria: Prediction and Model Evaluation	
Meets Expectations: 
Predictions were made on validation data using the selected relevant features in the logistic regression model
Predictions were made on validation data using the random forest model
Evaluated the performance of both the logistic regression and random forest models using the given evaluation metrics
 

Does Not Meet Expectations:
Failed to select relevant features or made incorrect predictions on validation data in the logistic regression model
Made incorrect predictions on validation data using the random forest model
Failed to evaluate the performance of both the logistic regression and random forest models using the correct evaluation metrics or interpreted the results inaccurately

Criteria: Report 	
Meets Expectations: 
The report has a clear structure, is not too long and explains the most important results concisely in simple language
The recommendations to solve the problem are realistic, actionable and coherent with the analysis
The report includes visualisations and insights derived from them
If any assumptions are made, they are stated clearly

 
Does Not Meet Expectations: 
The report lacks structure, is too long or does not emphasise the important observations. The language used is complicated for business people to understand
The recommendations to solve the problem are either unrealistic, non-actionable or incoherent with the analysis
The report is missing visualisations or fails to provide meaningful insights
Assumptions made, if any, are not stated clearly

Criteria: PPT 	
Meets Expectations: 
Provided a concise overview of the assignment, covering key objectives, methodology and findings
Effectively answered the questions asked in the assignment using appropriate visualisations
Used clear, well-structured slides with relevant charts, graphs and summaries to enhance understanding
 

Does Not Meet Expectations:
Failed to provide a clear and concise overview of the assignment
Did not effectively answer the questions asked in the assignment or used inappropriate/unclear visualisations
PPT lacked structure, clarity or  relevant visual elements, making it difficult to understand key insights
 

Criteria: Conciseness and Readability of the Code
Meets Expectations: 
The code is concise and syntactically correct. Wherever appropriate, built-in functions and standard libraries are used instead of writing long code (if-else statements, loops, etc.)
Custom functions are used to perform repetitive tasks
The code is readable with appropriately named variables, and detailed comments are        written wherever necessary
 
Does Not Meet Expectations:
Long and complex code is used instead of shorter built-in functions
Custom functions are not used to perform repetitive tasks, resulting in the same piece of code being repeated multiple times
Code readability is poor because of vaguely named variables or a lack of comments wherever necessary
 
