import pandas as pd
from bertopic import BERTopic
import os
from sklearn.feature_extraction.text import CountVectorizer

# 1. Load your local CSV (No internet needed!)
print(" Loading your 56k articles...")
df = pd.read_csv("filtered_grievance_data_en.csv")

# 2. Data Cleaning for the Model
# Ensure we have dates and text
df['date_publish'] = pd.to_datetime(df['date_publish'], errors='coerce')
df = df.dropna(subset=['maintext', 'date_publish'])

# 3. Subsampling for the Demo
# To keep this fast (under 10 mins), let's use 4,000 articles
df_sample = df.sample(n=min(4000, len(df)), random_state=42).sort_values('date_publish')
docs = df_sample['maintext'].tolist()
timestamps = df_sample['date_publish'].tolist()

print(f" Training BERTopic on {len(docs)} articles...")


vectorizer = CountVectorizer(stop_words="english")
# 4. Run BERTopic
# This uses 'all-MiniLM-L6-v2' by default - small, fast, and accurate
topic_model = BERTopic(
    language="english", 
    min_topic_size=20, # Smaller size for smaller sample
    verbose=True,
    vectorizer_model = vectorizer
)

topics, probs = topic_model.fit_transform(docs)

df_sample = df_sample.copy()
df_sample["topic"] = topics #attach resulting index back to the texts
doc_info = topic_model.get_document_info(docs) #for the centrality score of articles
df_sample["probability"]  =doc_info["Probability"]

# 5. Get the Narrative Summary
topic_info = topic_model.get_topic_info()
print("\n---  TOP 5 NARRATIVES IDENTIFIED ---")
print(topic_info[['Topic', 'Count', 'Name']].head(6))

# 6. Save Narrative Over Time (For your VAR model)
print("\n Mapping narratives to your timeline...")
topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
topics_over_time.to_csv("narrative_pulse_data.csv")

print("\n SUCCESS! 'narrative_pulse_data.csv' is ready for your VAR model.")

#NEWLY ADDED

topic_dfs= {}

for topic_id in df_sample["topic"].unique():
    if topic_id == -1: #ignore the articles that couldnt be classified
        continue
    topic_dfs[topic_id] = df_sample[df_sample["topic"] == topic_id] #split dataset into mini datasets of 
                                                                    #articles based on topics

for topic_id, tdf in topic_dfs.items():##loop through each cluster
    print(f"topic {topic_id}")
    for i, row in tdf.head(5).iterrows(): #get first 5 article per topic
        print(f"article {i}")
        print(row["maintext"][:300])  #get the first 300 characters of first 5 artciles



for topic_id in df_sample["topic"].unique():
    if topic_id == -1: #ignore the articles that couldnt be classified
        continue
    print(f"topic {topic_id}")
    #selects the highest centrality articles per topic
    top_articles = (
    df_sample[df_sample["topic"] == topic_id]
    .sort_values("probability", ascending=False)
    .head(5)
)                                    
    for i, row in top_articles.iterrows():
        print("for n: \n")
        print(row["maintext"][:300])