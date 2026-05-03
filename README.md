#Spotify MPD Recommendation Pipeline
Big Data and Analytics — Group Project
Master in Management | Database Management 2026
Supervision: Prof. Carlos J. Costa, PhD | Saeed Angorani, DBA

Authors: Diessner, Maximilian (65344) | Fiedler, Felix (65386) | Ryciuk, Anita (66743) | Leonetti Luparini, Filippo Maria (65233) | De Tuddo, Chiara (65254)

Live Dashboard: https://big-data-gdrchdvtkjcmditzfdx9vx.streamlit.app

Project Overview
This project implements an end-to-end Big Data recommendation pipeline using the Spotify Million Playlist Dataset (MPD). The pipeline processes 66 million track-level observations across 1 million playlists in a distributed Databricks environment, trains an ALS Collaborative Filtering model and applies a slot-based long-tail re-ranking step to increase exposure to less dominant artists.

Research Question: How can large-scale Spotify playlist data be used to build a scalable track recommendation system that balances predictive accuracy with long-tail artist exposure?

Dataset
Metric	Value
Playlists	1,000,000
Track-level rows	66,346,428
Unique tracks	2,262,292
Unique artists	295,860
Missing values	0
Duplicate rows	0
Source: Kaggle — himanshuwagh/spotify-million
Original paper: Chen et al. (2018), RecSys Challenge 2018

Repository Structure
ML_RecommendationSystem.ipynb   # Main pipeline: ingestion, ETL, modeling, evaluation
dashboard.py                    # Streamlit dashboard (2 dashboards, 4 charts each)
export_for_dashboard.py         # Export script for dashboard data
requirements.txt                # Python dependencies for Streamlit Cloud
README.md                       # This file
Technology Stack
Layer	Technology
Ingestion	Kaggle API (Python) → Unity Catalog Volume
Storage	Databricks Unity Catalog Volumes (Parquet)
Processing	Apache Spark 4.1, PySpark, SparkSQL
Modeling	Spark MLlib ALS (implicit feedback)
Scoring	NumPy driver-side matrix multiply
Dashboard	Streamlit + Plotly
Environment	Databricks Serverless
Setup Instructions
Prerequisites
Databricks Community Edition: community.cloud.databricks.com
Kaggle account with API token: kaggle.com → Settings → API → Create New Token
Step 1 — Create a Databricks Notebook
Log in to Databricks Community Edition
Click New → Notebook → set language to Python
Make sure Serverless Compute is selected
Step 2 — Configure Credentials
Open the notebook and fill in credentials at the top:

KAGGLE_USERNAME = "your_kaggle_username"
KAGGLE_KEY      = "your_kaggle_key"
Never commit credentials to a public repository.

Step 3 — Run the Pipeline
Copy the contents of ML_RecommendationSystem.ipynb into Databricks and run. The pipeline will:

Download MPD from Kaggle → copy to Unity Catalog Volume
Flatten and clean JSON data → save Parquet checkpoint
Compute track popularity via SparkSQL
Build ALS modeling table with integer IDs
Train ALS model
Evaluate with Precision@10, Recall@10, NDCG@10
Apply slot-based long-tail re-ranking
Save all outputs
Step 4 — Resume After Timeout
Checkpoints are saved automatically. The pipeline resumes from the last checkpoint:

mpd_clean_flat/     # after flatten + clean
track_popularity/   # after popularity computation
als_model/          # after ALS training
Model Details
Algorithm: ALS (Alternating Least Squares) Collaborative Filtering
Feedback type: Implicit — playlist membership as behavioral signal

Parameter	Value	Reason
Rank	20	Keeps model under 256 MB Serverless memory limit
MaxIter	5	Sufficient convergence for implicit feedback ALS
RegParam	0.05	Regularization to prevent overfitting
Alpha	60.0	Confidence scaling for implicit feedback
Popularity Tiers
Tier	Definition	Role
High	Top 10% by playlist appearances	7 recommendation slots
Medium	70th–90th percentile	1 recommendation slot
Low	Below 70th percentile (long-tail)	2 guaranteed slots
The Spotify Web API was deliberately excluded. Spotify deprecated the popularity score in 2024 — deriving popularity from playlist_count is more authentic and reproducible.

Evaluation Results
Metric	ALS only	After Re-ranking
Precision@10	0.0300	0.0415
Recall@10	0.0135	0.0280
NDCG@10	0.0301	0.0480
Long-tail Coverage	0%	22%
RMSE: 6.4787 · Popularity Baseline Precision@10: 0.0621

Pipeline Outputs
All outputs saved as Parquet in /Volumes/spotify_project/mpd/outputs/:

mpd_clean_flat/              # 66M rows, cleaned flat table
track_popularity/            # 2.2M tracks with playlist_count and popularity_tier
mpd_modeling/                # Deduplicated interaction table (ALS input)
playlist_index/              # playlist_id → playlist_idx mapping
track_index/                 # track_id → track_idx mapping
als_model/                   # Trained ALS model (Rank 20)
recommendations_reranked/    # Final re-ranked top-10 recommendations
#SparkSQL Integration
SparkSQL is used at four key points in the pipeline:

Track popularity: COUNT(DISTINCT playlist_id) per track
Top-10 artists: ranked by playlist appearances
Top-20 tracks: ranked by playlist count
Re-ranking analysis: tier distribution with SQL window functions
References
Chen, C. W., Lamere, P., Schedl, M., & Zamani, H. (2018). Recsys Challenge 2018: Automatic music playlist continuation. RecSys '18. ACM. https://doi.org/10.1145/3240323.3240342
Costa, C. J., & Aparicio, J. T. (2020). POST-DS: A methodology to boost data science. CISTI 2020. IEEE.
Spotify Research. (2020). The Spotify Million Playlist Dataset — Remastered. https://research.atspotify.com/2020/09/the-million-playlist-dataset-remastered
