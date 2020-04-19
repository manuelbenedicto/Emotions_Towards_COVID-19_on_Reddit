# Let's import all the necessary libraries to retrieve and preprocess the data
import requests
import numpy as np
import pandas as pd
import json
from datetime import datetime
import textblob
from textblob.classifiers import NaiveBayesClassifier

terms = ['coronavirus', 'corona virus', 'COVID-19', 'COVID19', 'covid-19', 'covid19']

def pushshift(base_url, **kwargs):
    base_url = 'https://api.pushshift.io/reddit/search/comment/?'
    params = kwargs
    request = requests.get(base_url, params=kwargs)
    return request.json()

def get_pushshift_data(terms):
    df_final = pd.DataFrame()
    for term in terms:
        start_date = 1577750400 # 1st January 2020
        now = datetime.now()
        now_epoch = now.timestamp()
        end_date = int(now_epoch)
        while start_date <= end_date:
            data = pushshift(base_url = 'https://api.pushshift.io/reddit/search/comment/?',
                              q = term,
                              after=str(start_date),
                              before=str(start_date+43200), #3600 seconds == 1 hour; 43200s = 12h
                              size=500,
                              sort_type='score',
                              sort='desc').get('data')
            if data == None:
                start_date = start_date + 43200
            else:
                df = pd.DataFrame.from_records(data)
                df_final= df_final.append(df)
                start_date = start_date + 43200  
    return df_final
                
df_final = get_pushshift_data(terms)
    
# Let's keep only some columns for our final database
df_final = df_final[['created_utc', 'author', 'subreddit', 'body', 'score', 'permalink']]

# Let's create a new column with all the epoch dates converted
df_final['date'] = pd.to_datetime(df_final['created_utc'], unit='s')

import textblob

# create a column with sentiment polarity
df_final["sentiment_polarity"] = df_final.apply(lambda row: textblob.TextBlob(row["body"]).sentiment.polarity, axis=1)

# create a column with sentiment subjectivity
df_final["sentiment_subjectivity"] = df_final.apply(lambda row: textblob.TextBlob(row["body"]).sentiment.subjectivity, axis=1)

# create a column with 'positive' or 'negative' depending on sentiment_polarity
df_final["sentiment"] = df_final.apply(lambda row: "positive" if row["sentiment_polarity"] >= 0 else "negative", axis=1)

# create a column with a text preview that shows the first 50 characters
df_final["preview"] = df_final["body"].str[0:50]

df_final.to_csv(path_or_buf='covid19_comments.csv')
