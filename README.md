# Scalable Music Recommendation Pipeline for Spotify Playlist Data
#Big Data and Analytics — Group Project
Master in Management | Database Management 2026
Supervision: Prof. Carlos J. Costa, PhD | Saeed Angorani, DBA
Authors:
Diessner, Maximilian (65344) | Fiedler, Felix (65386) | Ryciuk, Anita (66743)
Filippo Maria Leonetti Luparini (65233) | Chiara De Tuddo (65254)

Project Overview
This project implements an end-to-end Big Data recommendation pipeline using the Spotify Million Playlist Dataset (MPD). The pipeline processes 66 million track-level observations across 1 million playlists in a distributed Databricks environment, trains an ALS Collaborative Filtering model and applies a long-tail re-ranking step to increase exposure to less dominant artists.

Research Question: How can large-scale Spotify playlist data be used to build a scalable track recommendation system that balances predictive accuracy with long-tail artist exposure?

Repository Structure
spotify_pipeline_final.py   # Main pipeline: ingestion, ETL, modeling, evaluation
README.md                   # This file

Technology Stack
LayerTechnologyIngestionKaggle API (Python) → Unity Catalog VolumeStorageDatabricks Unity Catalog Volumes (Parquet)ProcessingApache Spark 4.1, PySpark, SparkSQLModelingSpark MLlib ALS (implicit feedback)DeploymentStreamlit dashboardEnvironmentDatabricks Community Edition (Serverless)

Dataset
Spotify Million Playlist Dataset (MPD)
Source: Kaggle — himanshuwagh/spotify-million
Original paper: Chen et al. (2018), RecSys Challenge 2018
MetricValuePlaylists 1,000,000 Track-level rows 66,346,428, Unique track 2,262,292, Unique artists 295,860, Missing values 0, Duplicate rows 0

Setup Instructions
Prerequisites

Databricks Community Edition account: community.cloud.databricks.com
Kaggle account with API token: kaggle.com → Settings → API → Create New Token

Step 1 — Create a Databricks Notebook

Log in to Databricks Community Edition
Click New → Notebook
Set language to Python
Make sure Serverless Compute is selected

Step 2 — Configure Credentials
Open spotify_pipeline_final.py and fill in the credentials at the top of the file:

pythonKAGGLE_USERNAME = "your_kaggle_username"
KAGGLE_KEY      = "your_kaggle_key"

==> Never commit credentials to a public repository. Use environment variables or Databricks Secrets in production.

Step 3 — Run the Pipeline
Copy the entire contents of spotify_pipeline_final.py into a single Python cell in Databricks and run it.

The pipeline will:

Download the MPD from Kaggle → copy to Unity Catalog Volume
Flatten and clean the JSON data → save Parquet checkpoint
Compute track popularity via SparkSQL
Build the ALS modeling table with integer IDs
Train the ALS model
Evaluate with Precision@10, Recall@10, NDCG@10
Apply long-tail re-ranking
Save all outputs

Step 4 — Resume After Timeout
If the session times out, simply run the pipeline again. Checkpoints are saved at:

mpd_clean_flat/ — after flatten + clean
track_popularity/ — after popularity computation
als_model/ — after ALS training

The pipeline detects existing checkpoints and resumes automatically.

Pipeline Outputs
All outputs are saved as Parquet in:
/Volumes/spotify_project/mpd/outputs/
mpd_clean_flat/            # 66M rows, cleaned flat table
track_popularity/          # 2.2M tracks with playlist_count and popularity_tier

mpd_modeling/              # Deduplicated interaction table (ALS input) playlist_index/            # playlist_id → playlist_idx mapping
track_index/               # track_id → track_idx mapping
als_model/                 # Trained ALS model (Rank 20)
recommendations_reranked/  # Final re-ranked recommendations

Model Details
Algorithm: ALS (Alternating Least Squares) Collaborative Filtering
Feedback type: Implicit: playlist membership as behavioral signal
Confidence weight: log1p(playlist_count); tracks in more playlists have higher confidence
ParameterValueReasonRank20: Keeps model under 256 MB Serverless memory limitMaxIter5Sufficient: convergence for implicit feedback ALSRegParam0.1 Standard regularization to prevent overfittingAlpha40.0Confidence scaling for implicit feedback

Popularity Tiers
Track popularity is derived directly from the dataset ==> no external API required.
TierDefinitionRole in pipelineHighTop 10% by playlist appearancesBaseline benchmarkMedium70th–90th percentileBoosted by 10% in re-rankingLowBelow 70th percentile (long-tail)Boosted by 30% in re-ranking

The Spotify Web API was deliberately excluded. Spotify deprecated the popularity score in 2024, and deriving popularity from playlist_count is more authentic and reproducible.


Evaluation Results
Model is evaluated on a held-out 20% test split.
MetricValueRMSETBDPrecision@10TBDRecall@10TBDNDCG@10TBDLong-tail CoverageTBDBaseline Prec@10TBD

Results will be filled in after the final pipeline run completes.


SparkSQL Integration
SparkSQL is used at four points in the pipeline:

Track popularity: COUNT(DISTINCT playlist_id) per track
Top-10 artists: ranked by playlist appearances
Top-20 tracks: ranked by playlist count
Re-ranking analysis: tier distribution with SQL window function


References

Chen, C. W., Lamere, P., Schedl, M., & Zamani, H. (2018). Recsys Challenge 2018: Automatic music playlist continuation. RecSys '18. ACM. https://doi.org/10.1145/3240323.3240342
Costa, C. J., & Aparicio, J. T. (2020). POST-DS: A methodology to boost data science. CISTI 2020. IEEE.
Spotify Research. (2020). The Spotify Million Playlist Dataset — Remastered. https://research.atspotify.com/2020/09/the-million-playlist-dataset-remastered
