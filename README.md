# Recommender-System-using-Neural-Networks
The objective of this project is to develop a recommender system using collaborative filtering techniques enhanced with neural networks. 
## What is Recommender System, How is it achieved?
Recommender Systems are sophisticated algorithms designed to provide product-revelant-suggestions to users. These systems play a paramount role in enhancing user experience on various online platforms, including streaming services, e-commerce websites, social media

They achieve this goal by analyzing past user interactions such as ratings, click, purchases. These algorithms can be collaborative (based on user interaction) or content-based (item features).

## Objective:
The objective of this project is to develop a recommender system using collaborative filtering techniques enhanced with neural networks. The system aims to provide personalized recommendations to users based on their historical interactions with items, such as ratings or clicks. By leveraging neural network architectures, the recommender system seeks to capture complex patterns and relationships in the user-item interaction data, leading to more accurate and effective recommendations.

The process includes

Model Development: Design and implement a collaborative filtering model using neural networks to learn latent representations of users and items from interaction data

Personalization: Provide personalized recommendations to users by leveraging the learned user and item embeddings to predict user preferences for unseen items

Scalability: Ensure the scalability and efficiency of the recommender system to handle large-scale datasets and serve real-time recommendations to users

Evaluation: Evaluate the performance of the recommender system using relevant metrics such as Hit Rate and ranking metrics like NDCG (Normalized Discounted Cumulative Gain), to assess the quality of recommendations.

## Data:
### Name: MovieLens dataset (contains users and movies with user ratings) Samples: 100836 Number of Unique Users: 610 Number of Unique Movies: 9724 Number of Unique Genres: 951

## Training Setup:
•	Install Dependencies mentioned in notebook
•	run training.ipynb notebook on Jupyter or Visual Studio
•	Set up CUDA for faster training, Tesla 37C(Single Machine) is used in our setup
•	Trained model is saved on model folder with .pth extension

training.ipynb : Contains code to prepare data and train model EDA.ipynb : Contains Exploratory data analysis on the entire data train.csv, evaluation.csv : Contains 80:20 split of the entire dataset

Epoch: 10

Hidden Layers stack: [128,,64,32,16,8]

Total params: 12,441,185

batch_size = 32

Top k picks = 10

Loss function: Cross entroy loss function

# Why Cross Entropy Loss Function:

We use the cross-entropy loss function in classification tasks, including recommendation systems, because it measures the difference between predicted probabilities and actual class labels, effectively penalizing incorrect predictions by a larger margin compared to correct predictions. Additionally, it encourages the model to output confident probabilities for correct classes, aiding in better optimization and model convergence.

Our custom sequntial model with a stack of Linear layers and ReLU activation functions are trained for 10,12 epochs in different experimental setup. The best model parameters are given above.

Architecture:

![image](https://github.com/user-attachments/assets/a327cdf7-35c9-40b9-9b98-c1d93dc32c3d)


## Results 
We use the following metrics to evaluate the performance

NDCG : Normalized discounted cumulative gain HR : Hit Rate

The following are the various experiemntal setup and their respective results
![image](https://github.com/user-attachments/assets/7bb6377f-5838-4b0c-a434-9eeacd08d0bb)


Best performance has been recorded with the following setup

![image](https://github.com/user-attachments/assets/a32d23db-e83f-4429-8563-37682823d593)

### Re-Ranking:

To further improve Hit Rate and NDCG scores, re-ranking techniques has been used to get best user specific recommendations in the top-k results. In this approach the recommended movies from the model are then reranked based on users interaction on the respective recommended image. This will highly help us in giving better recommendation in top-k results

Eg:
Getting recommendations for userID : 453

Models HR and NDCG scores are

HR: 0.8 NDCG : 0.55
![image](https://github.com/user-attachments/assets/3809aa14-a3a0-4cc7-9afc-7ac51e8e1c2b)

But, after re-ranking scores are

HR : 1.0 NDCG: 1.0

![image](https://github.com/user-attachments/assets/f48b3f95-6a69-41c2-bf60-fd092ec5c0d7)

GPU-Utilization:

![image](https://github.com/user-attachments/assets/a0d1225e-5fba-48a0-a979-430aa7ccc2d9)


