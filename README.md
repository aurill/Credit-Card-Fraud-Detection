# Credit-Card-Fraud-Detection
Leveraged Machine Learning for a fraud detection project: executed extensive data preprocessing, compared multiple ML algorithms (including Logistic Regression, SVM, KNN, Random Forest, and MLP), and applied feature selection techniques for optimal model performance. I used k-fold cross-validation to prevent overfitting, fine-tuned hyperparameters via Grid Search and Random Search, and evaluated models based on precision, recall, F1-score, and AUC-ROC. Results were communicated through comprehensive data visualizations and reports.

Step 1: Data Extraction
The first step I took entailed extracting the provided CSV files' transaction data out. I used Python's Pandas package, a potent tool for data processing and analysis, and Google Colab, a machine learning framework developed by Google Research. Using these tools, I could successfully upload and read the data into a Data Frame for further processing.

Step 2: Data Analysis and Preprocessing
It was crucial to understand the patterns within the data so that I could know how to prepare the data. I investigated the unique values of potential fraudulent transactions and plotted histograms to visualize patterns better. Interestingly, I recognized a large number of transactions at repeat retailers were fraudulent. Additionally, I did my checks for null values, and I performed feature engineering to prepare the raw data for machine learning algorithms. This process involved normalizing features using StandardScaler from sklearn.preprocessing while splitting the dataset into a training set and a test set.

Step 3: Model Creation and Selection
After I had preprocessed the data, I used several machine learning models to find the one that performs best for the task at hand. I used Logistic Regression, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Random Forest, and Multi-Layer Perceptron (MLP) which is all offered under the python sklearn library. Among all the models, the Random Forest Classifier performed the best yielding a score of 1.0 or 100%- which means a perfect accuracy in how the model could correctly predict whether each transaction in the dataset was fraudulent or not.

Step 4: Model Evaluation
After training all the AI models, I made my evaluation on the scores. Further, I used the Receiver Operating Characteristic Area Under Curve (ROC AUC) score to evaluate the model's performance in a binary classification task with an imbalance between the classes. The Random Forest model delivered a ROC AUC score of 1.0 - a near-perfect/ perfect score that shows our model has excellent predictive power.

Step 5: Making Predictions
The final step was to use the chosen model to make predictions on the testing data and create a new DataFrame with the predicted labels. I then saved this DataFrame as a CSV file.

Summary
In closing, this project showed the power and potential of AI in providing practical solutions to pressing issues in digital finance, such as credit card fraud. By using different machine learning algorithms and data analysis techniques, I was able to develop a highly effective model for predicting fraudulent transactions.
