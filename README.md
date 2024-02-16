# Diabetes-Health-Indictor
We made preprocessing, feature extraction and selection, model training, model evaluation and testing the models with new data to predict if the user has  prediabetes/diabetes or no diabetes from the given features. This project was made on diabetes  _ binary _ health _ indicators _ BRFSS2015.csv dataset and the language used is python.


Overview:
•	Diabetes mellitus is one of the most serious chronic illnesses in the world. Diabetes is a serious chronic disease in which individuals lose the ability to effectively regulate levels of glucose in the blood and can lead to reduced quality of life and life expectancy.
•	Therefore, in this project we will be predicting whether the user has prediabetes/diabetes or no diabetes based on the given features to our machine learning model, and for that, we will be using Diabetes Health Indicators dataset.
Project steps in detail:
1.	Importing libraries:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/12f06374-b33b-46d1-836c-dc8e188d7f2c)

2.	 Reading from the file and displaying a tuple with the shape of DataFrame:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/215a3101-f72e-4a57-97d1-23c97f4e6e03)

•	Output:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/f0306704-25f8-4013-9e6d-a91ec814b4f7)


3.	Preprocessing:
•	Data cleaning: checking whether there’s nulls:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/a1b32fbd-4b03-450c-82ab-0a2c46948a52)

•	Output:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/bdda93d0-feac-4b14-9e81-ad4600f9891f)


4.	Representing data as X (data) and y (target) and displaying their shape:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/1f69c045-d395-4ec4-a952-79ecbc227836)

•	Output: 
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/c6ff5aa9-3651-4bbb-875b-362f1135e305)
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/f39c6c8d-847b-4f74-a02a-ca465f17fdee)

 

5.	The dataset is not balanced so we’ve tried oversampling, and undersampling to balance our data. We have found that oversampling is giving us better results, because we keep all the information in the training dataset. On the other hand, undersampling drops a lot of information.
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/221d34bd-f161-4448-96e1-469609dd3423)

•	Output:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/3c9de8f0-bd3e-4dc4-86da-209e0ca7ec09)

6.	Feature Selection:
•	Feature selection by percentile:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/4d8d4817-6c05-4b41-9c99-c3b852f3f748)

7.	Visualizing correlation:
•	With numbers:
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/3543334b-04f6-4802-ac29-0a3bc6d4b089)
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/714e733d-23f9-4b1a-8ed2-a55d1c5275ba)

 

•	Without numbers:
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/38e29b64-4d1b-4a7c-9aa7-2d0eec1fe511)
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/2164a037-df2f-46ea-8a93-dc065108c146)


 

8.	Data scaling: by FunctionTransformer:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/a8213478-3c31-470c-bf06-f427c4aa850d)

•	We’ve tried more than a way in data scaling, but we found FunctionTransformer the best one, and here are the other ways:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/881e9252-ff19-46c2-aafe-4cee1b30f86b)

9.	Normalization:
•	We’ve tried normalization, but it wasn’t the best results:
  ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/7ad2fbb0-201d-4c16-ad2a-b80358307f1d)

10.	Data splitting; x -> (x_train, x_test), y - > (y_train, y_test):
•	Now we will split the data into training and testing data using the train_test_split method:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/198d2233-b78c-4f5b-afdd-3fc0dd30ea53)

11.	Before building our models, we have made a model evaluation function which we’re going to use throughout our 3 different models:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/bfa7be1c-6acb-4d2a-b8c4-484ffa801fe8)

•	This function calculates the classification report (accuracy, precision, f1 score, classification, and confusion matrix) of our testing data.









12.	Model Building: we have made the 3 required models in    the project pdf:
•	Through our runtime process, we’ve found that the running time was so long especially in the svm model, so we’ve added a load and save functions to be able to speed up the process:
a)	Logistic Regression model:
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/4a2a80a4-129e-452d-a14d-04e0d9de2c41)



•	With save and load:
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/35e4b327-3011-45e7-ad05-714cdc9a9117)








•	Output:
 
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/34cf1b11-14bd-4ca5-bbad-9dd397d9bad8)
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/d0a8f22c-97bc-4f24-8ba1-e2c4e13754a2)

 








b)	Decision Tree model:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/3270d820-9643-45f9-97c9-a9a0a4cf98d8)

•	With save and load:
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/c3740750-778b-4e54-8ff1-dc5911e11d56)

•	Output:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/2833c3c3-975a-407d-90d2-af6b1a8fbeab)
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/e8b8de22-ed2a-4195-ae56-adb97d9e9a52)

 
c)	SVM model:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/ad2b8731-dd52-46b4-b95e-b4b0a8e7bf28)

•	With save and load:
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/745ff98f-c9b3-4607-8203-ca2bc59ea3f5)

•	Output:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/1601827d-f000-4a37-ac5b-a06b48346182)
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/93cb3052-5d69-49aa-8234-901f6add5360)

 
13.	Testing the methods with new data:
•	Our last step is testing our models with a new dataset, we have extracted random rows as a new dataset from our already existing dataset (new dataset -> 300 rows only).
•	We’ve changed the file name we’re reading from at the beginning, and we’ve removed the data splitting, so we’ve changed y_test and y_train to y, and X_train and X_test to X . 
•	New dataset and checking for nulls:
  ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/8d47ef83-2b8e-46c0-9f94-a19a8254460b)


•	X and y:  
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/e6f4ce38-d0bf-42f7-8077-75b02f4c069f)

•	X and y before and after over sampling:
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/5de5f83d-74de-4dea-9cbc-de97cc842fc4)

 
•	Logistic Regression Model:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/f91e051e-e77e-4cb7-9006-4049976bda0f)

•	SVM Model:
 ![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/1eeacc10-acee-43b4-93fa-4b3404e45e37)


•	Decision Tree Model:
![image](https://github.com/Nouran936/Diabetes-Health-Indictor/assets/112628931/e69474e4-1c51-4465-a118-ef96984a0dce)
