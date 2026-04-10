# Intermediate Check-In
Benjamin Liu and William Zhang

## Get to Know Your Data
Our dataset is a join of Billboard Hot 100 tracks (1958 to present) and a Kaggle Spotify audio features dataset, both with 100k+ tracks each. After joining and deduplicating, there are 2102 matched tracks. Using top-10 on Billboard as "hit" classification, 96.4% are non-hits and 3.6% are hits. Thus, we plan to use F1 and AUC to measure the performance of our model. Our original plan was to use Spotify preview URLs for librosa extraction, but since the API has been deprecated since November 2024 for new apps, we pivoted to pre-computed audio features for now, until we can figure out an alternative.

## EDA & Analysis
Our first EDA figure is a bar chart that shows the class imbalance. Our second one consists of distribution plots of each audio feature split by label. Our third EDA consists of a correlation heatmap, which shows that features like energy and loudness are highly correlated.

### PCA (Principal Component Analysis)
We performed PCA dimensionality reduction on the 9 audio features:
- **PC1** explains 26.3% of variance (driven by energy & loudness)
- **PC2** explains 16.8% of variance (driven by danceability & valence)
- **PC3** explains 11.8% of variance (driven by liveness & speechiness)
- **First 3 PCs capture ~54.9% of total variance** - substantial reduction from 9 features

#### Key PCA Insights:
- Energy and loudness load heavily on PC1: features are strongly correlated as suspected
- Danceability and valence load on PC2: both are "vibe" features (liveness)
- The biplot shows modest separation between hits (blue) and non-hits (red), suggesting some linearly separable patterns exist
- Feature loading arrows reveal which audio qualities contribute to each principal component

### t-SNE (Non-linear Dimensionality Reduction)
t-SNE visualization of the data reveals:
- **Non-hits (red) and hits (blue) cluster separately in some regions**: suggests non-linear decision boundaries exist
- **Hits don't form a tight cluster**: hits are heterogeneous (upbeat songs, slow songs, acoustic songs all can be hits)
- **t-SNE vs PCA difference**: t-SNE shows more distinct clustering than PCA, indicating non-linear patterns that Logistic Regression may miss
- **Imbalance visible**: vast red (non-hit) region with sparse blue (hit) dots confirms extreme class imbalance

### Feature Importance (Cohen's d Effect Sizes)
Features ranked by discriminative power (hits vs non-hits):
speechiness         0.416
loudness            0.398
danceability        0.394
liveness            0.171
acousticness        0.155
tempo               0.111
energy             -0.099
instrumentalness   -0.271
valence            -0.392

## Modeling
We implemented simple Logistic Regression as a baseline and found an AUC of 0.776 and F1 score of 0.799. 

## Project Management
April 9: Complete intermediate check-in with David Zhang.
April 16: Train and tune Random Forest and Gradient Boosted Tree.
April 23: Dashboard integration completed.
April 30: Finalize code, dashboard polish, and presentation prep. Submit.