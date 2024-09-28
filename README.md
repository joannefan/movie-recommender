# movie-recommender

by Joanne and Jane

This project trains a movie recommendation system using the MovieLens 100k dataset.

We use matrix factorization (specifically Singular Value Decomposition) to decompose the user-item interaction matrix into lower-dimensional latent factors that represent users' and movies' underlying preferences. By projecting both users and movies into this latent space, we can predict unseen ratings
by training several types of machine learning models and searching over many hyperparameters. We evaluate the performance using balanced accuracy and AUROC, selecting a Random Forest classifier as the best performing model, which achieved the highest accuracy and AUROC among 48 teams.