import pandas as pd
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from nltk.probability import FreqDist
from collections import Counter
import string
from scipy.stats import pearsonr

# Loading the json file containing review data
with open("projectnew/Industrial_and_Scientific.json", "r") as file:
    data = [json.loads(line) for line in file]
# selecting a product from the dataset by extracting asin
Product_Id = ["0176496920"]#first product
product_data = [n for n in data if n.get("asin", "") in Product_Id]

# Convert the list to a DataFrame
df = pd.DataFrame(product_data)
# Handling missing values in the column reviewText
df.dropna(subset=['reviewText'], inplace=True)
# dropping the column vote as it contain many NaN values
df.drop('vote', axis=1, inplace=True)
print(df)


# Normalize text function
def normalize_text(text):
    # Removing the full stop,hyphen,punctuation,comma
    
        custom_punctuation = string.punctuation.replace('-', '') 
        translation_table = str.maketrans('', '', custom_punctuation)
        cleaned_text = text.translate(translation_table)
       
    
   
    # Tokenization
        tokens = word_tokenize(cleaned_text.lower())
    # Stop word removal
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
    #Lemmatization
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        processed_text = ' '.join(lemmatized_tokens)
        return processed_text
# Apply the  above function to the 'reviewText' column
df['processed_review'] = df['reviewText'].apply(normalize_text)

# Initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# function for calculating polarity score
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 2  # Positive sentiment
    elif scores['compound'] <=-0.05:
        return 0  # Negative sentiment
    else:
        return 1  # Neutral sentiment


# Adding sentiment column to dataframe by applying get_sentiment function to reviewText
df['sentiment'] = df['reviewText'].apply(get_sentiment)
#printing dataframe information and  first five rows.
print(df.info())
print(df.head())
# saving the dataframe into a csv file
df.to_csv('output_file1.csv')

# Plot word clouds

pos = df[df['sentiment'] == 2]
neutral = df[df['sentiment'] == 1]
neg = df[df['sentiment'] == 0]
def plot_cloud(wordcloud, title):
    
    plt.figure(figsize=(15, 5), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.suptitle(title, fontsize=20)
    plt.tight_layout(pad=0)
    plt.show()
pos_wordcloud = WordCloud(width=500, height=300).generate(' '.join(pos['processed_review']))
neutral_wordcloud = WordCloud(width=500, height=300).generate(' '.join(neutral['processed_review']))
neg_wordcloud = WordCloud(width=500, height=300).generate(' '.join(neg['processed_review']))

plot_cloud(pos_wordcloud, 'Positive Sentiment Word Cloud')
plot_cloud(neutral_wordcloud, 'Neutral Sentiment Word Cloud')
plot_cloud(neg_wordcloud, 'Negative Sentiment Word Cloud')



# finding the most frequent 10 words from each sentiment category
negative_rev = df[df['sentiment'] == 0]['processed_review']
neutral_rev = df[df['sentiment'] == 1]['processed_review']
positive_rev = df[df['sentiment'] == 2]['processed_review']

# Tokenize and count the frequency of words in each sentiment category
negative_word_number = Counter(word_tokenize(' '.join(negative_rev)))
neutral_word_number = Counter(word_tokenize(' '.join(neutral_rev)))
positive_word_number = Counter(word_tokenize(' '.join(positive_rev)))

# finding the 10 most frequenct words for each sentiment category
most_common_neg = negative_word_number.most_common(10)
most_common_neu = neutral_word_number.most_common(10)
most_common_pos = positive_word_number.most_common(10)

# Print the most frequent words for each sentiment category
print("Most Frequently Used Terms in Negative Reviews:")
print(most_common_neg)

print("\nMost Frequently Used Terms in Neutral Reviews:")
print(most_common_neu)

print("\nMost Frequently Used Terms in Positive Reviews:")
print(most_common_pos)

#Extracting year from reviewTime 
df['reviewTime'] = pd.to_datetime(df['reviewTime'])
df['Year'] = df['reviewTime'].dt.year


#//Plotting the count of positive ,negative and neutral reviews of product over the years 

Sentiment_Year = df.groupby(['Year', 'sentiment']).size().reset_index(name='Count')
Sentiment_Year=Sentiment_Year.rename(columns={'asin': 'Count'})
positive_year = Sentiment_Year[Sentiment_Year['sentiment'] == 2]
negative_year = Sentiment_Year[Sentiment_Year['sentiment'] == 0]
neutral_year = Sentiment_Year[Sentiment_Year['sentiment'] == 1]
Sentiment_Total_Year=Sentiment_Year.groupby('Year')['Count'].sum().reset_index()
Sentiment_Total_Year=Sentiment_Total_Year.rename(columns={'Count': 'Total_Count'})
result_Positive_Year=pd.merge(positive_year, Sentiment_Total_Year, on='Year', how='inner')
result_Negative_Year=pd.merge(negative_year, Sentiment_Total_Year, on='Year', how='inner')
result_Neutral_Year=pd.merge(neutral_year, Sentiment_Total_Year, on='Year', how='inner')
result_Positive_Year['Percentage']=(result_Positive_Year['Count']/result_Positive_Year['Total_Count'])*100
result_Negative_Year['Percentage']=(result_Negative_Year['Count']/result_Negative_Year['Total_Count'])*100
result_Neutral_Year['Percentage']=(result_Neutral_Year['Count']/result_Neutral_Year['Total_Count'])*100
positive_year.plot(x="Year", y="Count", kind="bar", title="Positive Reviews over the Years based on Sentiments")
plt.show()
negative_year.plot(x="Year", y="Count", kind="bar", title="Negative Reviews over the Years based on Sentiments")
plt.show()
neutral_year.plot(x="Year", y="Count", kind="bar", title="Neutral Reviews over the Years based on Sentiments")
plt.show()

#PLOTTING TOTAL NO OF reviews OVER YEARS
yearly=df.groupby(['Year'])['asin'].count().reset_index()
yearly=yearly.rename(columns={'asin': 'Number_Of_Reviews'})
yearly.head()
yearly.plot(x="Year",y="Number_Of_Reviews",kind="line",title="TOTAL NUMBER OF REVIEWS OVER THE YEARS")
plt.show()


# plotting a histogram for sentiment polarity distribution using matplotlib
sentiment_df = pd.DataFrame({'sentiment': df['sentiment']})
plt.figure(figsize=(9, 4))
plt.hist(sentiment_df['sentiment'], bins=30, color='red', edgecolor='black')
plt.xlabel('Polarity score')
plt.ylabel('Count')
plt.title('Sentiment Polarity Distribution')
plt.show()

# Create a DataFrame for overall ratings
overall_df = pd.DataFrame({'overall': df['overall']})

# plotting a histogram for overall ratings using matplotlib
plt.figure(figsize=(9, 4))
plt.hist(overall_df['overall'], bins=30, color='green', edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Review Rating Distribution')
plt.show()
#plotting average rating
df_Rating=df[['reviewerID','overall']]
df_Rating1=df_Rating.groupby(['reviewerID'])['overall'].mean().reset_index()
df_Rating2=df_Rating1.groupby('overall')['reviewerID'].count().reset_index()
df_Rating2=df_Rating2.rename(columns={'reviewerID':'number'})
df_Rating2.head()
df_Rating2.plot(x="overall",y="number",kind="scatter",title="Average Rating Written By User")
plt.show()

# Calculating Pearson correlation between sentiment and overall rating
correlation, p_value = pearsonr(df['sentiment'], df['overall'])
print(f"Pearson Correlation: {correlation}")
print(f"P-Value: {p_value}")

import seaborn as sns  # Import seaborn for enhanced plotting capabilities

# Scatter plot with regression line
sns.regplot(x='sentiment', y='overall', data=df, scatter_kws={'alpha':0.5})
plt.title('Sentiment vs Overall Ratings')
plt.xlabel('Sentiment')
plt.ylabel('Overall Ratings')

# Display the Pearson correlation in the plot
plt.text(df['sentiment'].min(), df['overall'].max(), f'Pearson Correlation: {correlation:.2f}', va='bottom', ha='left')
plt.show()

# Finding recommendations to manufacturers
# sentiment score '0' represents negative sentiment
neg_commts = df[df['sentiment'] == 0]

# Extract important words from negative comments
important_words = []
for comment in neg_commts['processed_review']:

    if 'quality' in comment:
       important_words.append('Product Quality')
    if 'customer service' in comment:
        important_words.append('Customer Service')
    if 'shipping' in comment:
        important_words.append('Shipping')

# Provide recommendations based on important_words
recommendations = []
for key in set(important_words):
    if   key== 'Product Quality':
        recommendations.append('Improve product quality standards and testing procedures.')
    elif key == 'Customer Service':
        recommendations.append('Enhance customer service responsiveness and effectiveness.')
    elif key == 'Shipping':
        recommendations.append('Optimize shipping processes to prevent delays.')

# Print recommendations
print("\nRecommendations to the manufacturer:")
for recommendation in recommendations:
    print("- " + recommendation)