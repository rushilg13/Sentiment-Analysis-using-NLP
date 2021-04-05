# Import Modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import nltk
import warnings
warnings.filterwarnings('ignore')

# Import dataset
df = pd.read_csv('train_tweets.csv')
# print(df.head())

# Preprocessing Dataset

def remove_pattern(input_text, pattern):
    r = re.findall(pattern, input_text)
    for word in r:
        input_text = re.sub(word, "", input_text)
    return input_text

# Remove @user (twitter handles)
df['Clean Tweets'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
# print(df.head())

# Remove special characters
df['Clean Tweets'] = df["Clean Tweets"].str.replace("[^a-zA-Z#]", " ")
# print(df.head())

# Remove Short words
df['Clean Tweets'] = df['Clean Tweets'].apply(lambda x: " ". join(w for w in x.split() if len(w) > 3))
# print(df.head())

# Individual words considered as tokens
tokenized_tweets = df["Clean Tweets"].apply(lambda x : x.split())

# Stem the words
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
tokenized_tweets = tokenized_tweets.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
# print(tokenized_tweets.head())

# Combine words into a single sentence
for i in range(len(tokenized_tweets)):
    tokenized_tweets[i] = " ".join(tokenized_tweets[i])
df["Clean Tweets"] = tokenized_tweets
# print(df.head())

# Visualise frequent words
all_words = " ".join([sentence for sentence in df["Clean Tweets"]])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)
# Plot Wordcloud
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
# plt.show()

# Extract hashtags
def hashtag_extract(tweets):
    hashtags = []
    # Loop words in text
    for tweet in tweets:
        ht = re.findall( r"#(\w+)", tweet) 
        hashtags.append(ht)
    return hashtags

# Extract hashtags from non-racist comments
ht_positive = hashtag_extract(df['Clean Tweets'][df['label']==0])
ht_positive = sum(ht_positive, [])
# print(ht_positive)

ht_negative = hashtag_extract(df['Clean Tweets'][df['label']==1])
ht_negative = sum(ht_negative, [])
# print(ht_negative)

freq = nltk.FreqDist(ht_positive)
d = pd.DataFrame({'Hashtags': list(freq.keys()), 'Count': list(freq.values())})
d.head()

# Select top 10 positive hashtags
d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(15,9))
sns.barplot(data=d, x='Hashtags', y='Count')
# plt.show()

# Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['Clean Tweets'])

# Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bow, df['label'], random_state=42, test_size=0.25)

model = LogisticRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print("f1 Score:", f1_score(y_test, pred))
print("Accuracy Score:", accuracy_score(y_test, pred))
