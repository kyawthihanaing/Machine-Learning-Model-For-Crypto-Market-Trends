#--------------------------Data Collection--------------------------------------------------------#
import twint
import nest_asyncio
from langdetect import detect
nest_asyncio.apply()

twtw = twint.Config()
twtw.Search = "bitcoin"
twtw.Store_csv = True
twtw.Limit = 1000
twtw.Lang = 'en'
twtw.Until = "2022-08-10"
twtw.Since = "2022-08-09"
twtw.Pandas = True
twint.run.Search(twtw)

def savedtoPd(col):
    return twint.output.panda.Tweets_df[col]

scrapeddata = savedtoPd(["id", "date", "tweet"])
print(scrapeddata)
scrapeddata.to_csv("data2.csv", index=False)

#--------------------------Pattern Evaluation--------------------------------------------------------#
import pandas as pd
import csv
from textblob import TextBlob
df_new = pd.read_csv('processeddata.csv')
df_new['polarity'] = df_new.apply(lambda x: TextBlob(x['Tweets']).sentiment.polarity, axis=1)
df_new['subjectivity'] = df_new.apply(lambda x: TextBlob(x['Tweets']).sentiment.subjectivity, axis=1)
print (df_new)

## create function to compute positive, negative and neutral analysis
def getAnalysis(score):
    if score<0:
        return 'Negative'
    elif score==0:
        return 'Neutral'
    else:
        return 'Positive'
df_new['Sentiment'] = df_new['polarity'].apply(getAnalysis)
df_new.to_csv('Mar24.csv', index=False)
print(df_new.to_csv)