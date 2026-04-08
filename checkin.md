# Intermediate Check-In
Benjamin Liu and William Zhang

## Get to Know Your Data
Our dataset is a join of Billboard Hot 100 tracks (1958 to present) and a Kaggle Spotify audio features dataset, both with 100k+ tracks each. After joining and deduplicating, there are 2102 matched tracks. Using top-10 on Billboard as "hit" classification, 96.4% are non-hits and 3.6% are hits. Thus, we plan to use F1 and AUC to measure the performance of our model. Our original plan was to use Spotify preview URLs for librosa extraction, but since the API has been deprecated since November 2024 for new apps, we pivoted to pre-computed audio features for now, until we can figure out an alternative.

## EDA
Our first EDA figure is a bar chart that shows the class imbalance. Our second one consists of distribution plots of each audio feature split by label. Our third EDA consists of a correlation heatmap, which shows that features like energy and loudness are highly correlated.

## Modeling
We implemented simple Logistic Regression as a baseline and found an AUC of 0.776 and F1 score of 0.799. We applied class weights to prevent the model from defaulting to non-hit when it is not sure. Since Logistic Regression uses a linear decision boundary, this may not work perfectly as audio features don't separate linearly. Thus, in the future, we plan to implement Random Forests, which can handle non-linear relationships and be robust to correlated features, and Gradient Boosted Trees, which are stronger on imbalanced data.

## Project Management
April 9: Complete intermediate check-in with David Zhang.
April 16: Train and tune Random Forest and Gradient Boosted Tree.
April 23: Dashboard integration completed.
April 30: Finalize code, dashboard polish, and presentation prep. Submit.