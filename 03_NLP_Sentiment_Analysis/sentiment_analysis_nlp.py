
!python -m spacy download en_core_web_sm

!pip install negspacy

import pandas as pd
import re
import string
from collections import Counter

# NLTK imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# SpaCy import for NER
import spacy

# WordCloud imports
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ==============================================================================
## Task 1: Text Preprocessing
# ==============================================================================
import pandas as pd
import re
import unicodedata
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
import spacy


# Download necessary NLTK data (if not already downloaded)
try:
    nltk.download('wordnet')
except LookupError:
    print("WordNet download failed. Please check your internet connection.")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
# removed punkt_tab download as it's not typically needed for word tokenization


# Load spaCy model for POS tagging and efficient processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Add 'sentencizer' to the spaCy pipeline to set sentence boundaries
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer", first=True)


# Initialize NLTK Lemmatizer (though spaCy lemmatization is often preferred and used here)
# lemmatizer = WordNetLemmatizer()

# --- Custom Resources for Preprocessing ---

# 1. Standard NLTK Stop Words (remove domain-specific words)
standard_stopwords = set(nltk_stopwords.words('english'))


# 2. Emoticon to Sentiment Word Mapping (using more generic sentiment words)
EMOTICON_MAP = {
    ":)": "happy", ":-)": "happy", ":D": "happy", ":-D": "happy",
    ":(": "sad", ":-(": "sad", ":'(": "sad", ";(": "sad",
    ":/": "mixed", ":-/": "mixed", ":|": "unclear", ":-|": "unclear", # Changed neutral to mixed/unclear
    "<3": "love", ":*": "kiss", ":-P": "sarcasm", ":P": "sarcasm",
    "xD": "laugh", "=D": "laugh", ":O": "surprise", ":-O": "surprise",
    "-_-": "annoyed", "-.-": "annoyed", "^_^": "happy",
    "😒": "annoyed", "😠": "angry", "😡": "angry", "😔": "sad",
    "😭": "crying", "😢": "crying", "😂": "laugh", "😊": "happy",
    "😄": "happy", "😁": "happy", "😍": "love", "👍": "good", # Changed positive to good
    "👎": "bad", # Changed negative to bad
    "🙏": "thankful", "💯": "excellent", "❤️": "love"
}


# Regex to find common emoticons (basic, might need refinement for comprehensive coverage)
# Keep the regex as it is for now
EMOTICON_REGEX = r"(:[-]?[\)\(\]\[oOpP\/\\|*@$]|[<>]?[:;=8][\-o\*\']?[D\)\]\(\/\@P]|[xX][Dd]|[=][D]|[><]:[DP])|\ud83d[\ude00-\ude4f]|\ud83c[\udf00-\udfff]|\ud83d[\udd00-\udfff]|\ud83e[\udd00-\uddff]"

# --- Main Preprocessing Function ---

def preprocess_text_for_ml_refined(text):
    if not isinstance(text, str):
        return "" # Handle non-string inputs, e.g., NaNs

    # 1. Lowercasing
    text = text.lower()

    # 2. Handle Emojis/Emoticons
    # Apply emoticon mapping before removing punctuation or tokenizing
    for emo, sentiment_word in EMOTICON_MAP.items():
        text = text.replace(emo, f" {sentiment_word} ")
    # Handle emoticons not in the map using the regex, replace with a placeholder or remove
    text = re.sub(EMOTICON_REGEX, lambda match: f" {EMOTICON_MAP.get(match.group(0), 'emoticon')} ", text) # Replace unmatched with 'emoticon'
    text = re.sub(r'\s+', ' ', text).strip() # Clean up spaces after replacement


    # 3. Remove URLs, Mentions, Hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)

    # Process with spaCy for tokenization, POS tagging, and lemmatization
    doc = nlp(text)

    # 4. Remove Punctuation (keep underscore for now)
    # This is done after spaCy processing to preserve token boundaries initially
    tokens = [token.text for token in doc]
    # Rejoin to apply punctuation removal and then re-tokenize
    text = ' '.join(tokens)
    text = re.sub(r'[^\w\s_]', '', text) # Remove punctuation except underscore
    tokens = text.split() # Re-tokenize after punctuation removal


    # 5. Rule-based Negation Handling (Replacing negspacy due to issues)
    # More sophisticated rule: negate terms until a punctuation or conjunction/preposition
    negation_terms = set(['not', 'no', 'never', 'none', 'nobody', 'nothing', 'nohow', 'nowhere', 'hardly', 'scarcely', 'barely', 'rarely'])
    stop_negation = set(['.', ',', ';', ':', '!', '?', 'but', 'however', 'although', 'whereas', 'while', 'and', 'or', 'nor', 'if', 'because', 'since', 'as', 'unless', 'until', 'before', 'after', 'when', 'where', 'why', 'how', 'than', 'rather', 'whether', 'whereas', 'according', 'across', 'after', 'against', 'along', 'amid', 'among', 'around', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by', 'concerning', 'considering', 'despite', 'down', 'during', 'except', 'excepting', 'excluding', 'following', 'for', 'from', 'in', 'inside', 'into', 'like', 'near', 'of', 'off', 'on', 'onto', 'opposite', 'out', 'outside', 'over', 'past', 'regarding', 'respecting', 'round', 'since', 'through', 'throughout', 'to', 'toward', 'towards', 'under', 'underneath', 'unlike', 'until', 'unto', 'up', 'upon', 'versus', 'via', 'with', 'within', 'without'])


    processed_tokens = []
    negate_next = False
    for token in tokens:
        lower_token = token.lower()
        if lower_token in negation_terms:
            negate_next = True
        elif lower_token in stop_negation:
            negate_next = False
            processed_tokens.append(token)
        elif negate_next:
            processed_tokens.append(f"not_{token}")
        else:
            processed_tokens.append(token)

    # Rejoin tokens to apply lemmatization on the negated phrases
    text = ' '.join(processed_tokens)

    # Process with spaCy again for lemmatization after negation handling
    doc = nlp(text)

    lemmatized_tokens = []
    for token in doc:
        # Apply Lemmatization only to Nouns and Verbs as requested
        if token.pos_ in ['NOUN', 'VERB']:
             # Handle negated terms during lemmatization
             if token.text.startswith('not_'):
                  # Split 'not_' and lemma, then join back
                  negated_part = 'not_'
                  original_word = token.text[len(negated_part):]
                  # Process the original word with spaCy to get its lemma
                  original_word_doc = nlp(original_word)
                  if original_word_doc and original_word_doc[0].pos_ in ['NOUN', 'VERB']:
                       lemmatized_tokens.append(f"{negated_part}{original_word_doc[0].lemma_}")
                  else:
                       # If the original word isn't a noun/verb or spaCy fails, keep as is
                       lemmatized_tokens.append(token.text)
             elif token.lemma_ == '-PRON-':
                  lemmatized_tokens.append(token.text)
             else:
                  lemmatized_tokens.append(token.lemma_)
        else:
            # Keep other POS tags (including negated adjectives/adverbs) un-lemmatized
            lemmatized_tokens.append(token.text)

    text = ' '.join(lemmatized_tokens)


    # 6. Remove Standard NLTK Stop Words
    tokens = text.split() # Re-tokenize after lemmatization
    filtered_tokens = [word for word in tokens if word not in standard_stopwords]
    text = ' '.join(filtered_tokens)


    # 7. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Load the dataset (assuming Tweets.csv is available)
try:
    df = pd.read_csv('Tweets.csv')
except FileNotFoundError:
    print("Error: 'Tweets.csv' not found. Please upload the file.")
    # Exit or handle the error appropriately if the file is not found


# Apply the refined preprocessing function to the 'text' column
df['processed_text_refined'] = df['text'].apply(preprocess_text_for_ml_refined)

# Display a few examples
print("Original Text vs. Refined Processed Text:")
for i in range(min(10, len(df))): # Display first 10 examples or fewer if dataset is small
    print(f"\n--- Example {i+1} ---")
    print(f"Original: '{df['text'].iloc[i]}'")
    print(f"Processed: '{df['processed_text_refined'].iloc[i]}'")

# Save the updated DataFrame to a new CSV file
df.to_csv('Tweets_processed_refined.csv', index=False)
print("\nDataFrame with refined processed text saved to 'Tweets_processed_refined.csv'")

# ==============================================================================
## Task 2: Word Frequency Count
# ==============================================================================
print("\n--- Task 2: Word Frequency Count ---")

# Collect all processed tokens from the 'processed_text_sophisticated' column
all_processed_tokens = df['processed_text_refined'].str.split().sum()

# Create a word frequency dictionary
word_freq = Counter(all_processed_tokens)

# Print the 5 most common words
print("\nTop 5 Most Common Words:")
for word, freq in word_freq.most_common(5):
    print(f"'{word}': {freq}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Preparation ---
# This dictionary holds the performance metrics you provided for each model.
# We'll use this to create the plots.
data = {
    'Model': [
        'TF-IDF, SMOTE, XGBoost',
        'Random Forest',
        'Sparse Voting Ensemble (LR + XGBoost)',
        'Tuned Sparse Voting Ensemble',
        'Tuned Stacking Ensemble',
        'Efficient Voting Ensemble (LR, NB, XGBOOST)',
        'SBERT + XGBoost',
        'Optimized SBERT + XGBoost',
        'SBERT + XGBoost (Oversampling, mpnet)'
    ],
    'Accuracy': [0.7661, 0.7691, 0.7794, 0.7879, 0.7821, 0.7787, 0.7763, 0.7865, 0.7958],
    # Using the weighted average F1-score for comparison
    'F1-Score': [0.77, 0.76, 0.78, 0.79, 0.78, 0.76, 0.77, 0.79, 0.79]
}

# Convert the dictionary to a pandas DataFrame for easier data manipulation.
df = pd.DataFrame(data)

# Sort the dataframes by the metrics to ensure the charts are ordered
# from best to worst performing, which makes them easier to interpret.
df_sorted_accuracy = df.sort_values('Accuracy', ascending=False)
df_sorted_f1 = df.sort_values('F1-Score', ascending=False)


# --- Generate Accuracy Comparison Bar Graph ---
# Set the size of the plot for better readability
plt.figure(figsize=(12, 8))

# Create the bar plot using seaborn for a clean and modern look.
# The 'viridis' palette provides a nice color scheme.
sns.barplot(x='Accuracy', y='Model', data=df_sorted_accuracy, palette='viridis', hue='Model', dodge=False, legend=False)

# Add labels and a title to the plot
plt.xlabel('Accuracy Score', fontsize=12)
plt.ylabel('Model', fontsize=12)
plt.title('Sentiment Analysis Model Accuracy Comparison', fontsize=16)

# Set the x-axis limits to focus on the range of your scores.
plt.xlim(0.75, 0.81)

# Add the exact accuracy value on each bar for precise comparison.
for index, value in enumerate(df_sorted_accuracy['Accuracy']):
    plt.text(value, index, f' {value:.4f}', va='center')

# Ensure all plot elements fit nicely within the figure.
plt.tight_layout()

# Display the plot. In a script, you might use plt.savefig('accuracy_comparison.png')
plt.show()


# --- Generate F1-Score Comparison Bar Graph ---
# Set the figure size
plt.figure(figsize=(12, 8))

# Create the F1-score bar plot using the 'plasma' color palette.
sns.barplot(x='F1-Score', y='Model', data=df_sorted_f1, palette='plasma', hue='Model', dodge=False, legend=False)

# Add labels and a title
plt.xlabel('F1-Score (Weighted Avg)', fontsize=12)
plt.ylabel('Model', fontsize=12)
plt.title('Sentiment Analysis Model F1-Score Comparison', fontsize=16)

# Set the x-axis limits
plt.xlim(0.75, 0.81)

# Add the exact F1-score value on each bar.
for index, value in enumerate(df_sorted_f1['F1-Score']):
    plt.text(value, index, f' {value:.2f}', va='center')

# Ensure a tight layout
plt.tight_layout()

# Display the plot. In a script, you might use plt.savefig('f1_score_comparison.png')
plt.show()

# Import necessary scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder
# Import resampling techniques
from imblearn.over_sampling import SMOTE # Import SMOTE
# Import XGBoost
import xgboost as xgb
import pandas as pd # Import pandas to use Series

# ==============================================================================
## Task 3: By Multi-class Sentiment Analysis (TF-IDF, SMOTE, XGBoost)
# ==============================================================================
print("\n--- Task X: Multi-class Sentiment Analysis (TF-IDF, SMOTE, XGBoost) ---")

if 'processed_text_refined' not in df.columns:
    print("Error: 'processed_text_refined' column not found. Please run the preprocessing cell (Task 1) first.")
else:
    # Define features (X) and target (y)
    X = df['processed_text_refined']
    y = df['airline_sentiment']

    # Encode the target variable into numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data into training and testing sets (using the encoded target)
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    # Print distribution of original labels for clarity
    print(f"Sentiment distribution in training set:\n{label_encoder.inverse_transform(y_train_encoded).tolist().count('negative')/len(y_train_encoded):.4f} negative, {label_encoder.inverse_transform(y_train_encoded).tolist().count('neutral')/len(y_train_encoded):.4f} neutral, {label_encoder.inverse_transform(y_train_encoded).tolist().count('positive')/len(y_train_encoded):.4f} positive")
    print(f"Sentiment distribution in test set:\n{label_encoder.inverse_transform(y_test_encoded).tolist().count('negative')/len(y_test_encoded):.4f} negative, {label_encoder.inverse_transform(y_test_encoded).tolist().count('neutral')/len(y_test_encoded):.4f} neutral, {label_encoder.inverse_transform(y_test_encoded).tolist().count('positive')/len(y_test_encoded):.4f} positive")


    # Feature Extraction: Convert text data into numerical features using TF-IDF
    print("\nApplying TF-IDF Vectorization...")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5, sublinear_tf=True)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print("TF-IDF Vectorization complete.")
    print(f"Number of features (TF-IDF terms): {X_train_tfidf.shape[1]}")


    # Apply SMOTE to the training data to handle class imbalance (using the encoded target)
    print("\nApplying SMOTE resampling to training data...")
    smote = SMOTE(sampling_strategy='not majority', random_state=42) # Use SMOTE with sampling_strategy='not majority'
    X_train_resampled, y_train_resampled_encoded = smote.fit_resample(X_train_tfidf, y_train_encoded)
    print("SMOTE resampling complete.")
    print(f"Training samples after resampling: {X_train_resampled.shape[0]}")
    # Print distribution of original labels in resampled set for clarity
    print(f"Sentiment distribution in resampled training set:")
    # Calculate and print the distribution using value_counts on the decoded labels for clarity
    resampled_labels_decoded = label_encoder.inverse_transform(y_train_resampled_encoded)
    resampled_distribution = pd.Series(resampled_labels_decoded).value_counts(normalize=True)
    print(resampled_distribution)


    # Train XGBoost Classifier on the resampled training data (using the encoded target)
    print("\nTraining XGBoost Classifier on resampled data...")
    xgb_classifier = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    xgb_classifier.fit(X_train_resampled, y_train_resampled_encoded)
    print("XGBoost Classifier training complete.")


    # Predict on the test data (using the trained vectorizer and classifier)
    print("\nPredicting on the test data...")
    y_pred_encoded = xgb_classifier.predict(X_test_tfidf)
    # Decode the predictions back to original labels
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    print("Prediction complete.")

    # Evaluate the model (using the original test labels and decoded predictions)
    print("\nEvaluating model performance on the test set:")
    accuracy = accuracy_score(label_encoder.inverse_transform(y_test_encoded), y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    # Use zero_division=0 to avoid warnings for classes with no predicted samples
    print(classification_report(label_encoder.inverse_transform(y_test_encoded), y_pred, zero_division=0))

    print("\n--- Using the Trained Model for Prediction ---")

    # Example sentences to predict sentiment on (using the trained vectorizer and classifier)
    new_sentences_for_prediction = [
        "This airline is fantastic, loved the service!",
        "My flight was delayed and the experience was terrible.",
        "The journey was okay, nothing special.",
        "I had a really bad day, everything went wrong.",
        "Virgin America, you are the best airline!",
        "AmericanAir my flight was Cancelled Flightled, so frustrating."
    ]

    # Preprocess the new sentences (assuming preprocess_text_for_ml_refined is defined in the environment)
    # If not, you would need to include or import the function here.
    # For this example, we assume it's available from the preprocessing cell.
    processed_new_sentences_for_prediction = [preprocess_text_for_ml_refined(s) for s in new_sentences_for_prediction]

    # Transform new sentences using the *same* TF-IDF vectorizer fitted on training data
    new_sentences_tfidf_for_prediction = tfidf_vectorizer.transform(processed_new_sentences_for_prediction)

    # Predict sentiment using the trained classifier (predictions will be encoded)
    predicted_sentiments_encoded = xgb_classifier.predict(new_sentences_tfidf_for_prediction)
    predicted_probabilities = xgb_classifier.predict_proba(new_sentences_tfidf_for_prediction) # Get probabilities

    # Decode the predictions back to original labels
    predicted_sentiments = label_encoder.inverse_transform(predicted_sentiments_encoded)

    print("\nSentiment Predictions for New Sentences:")
    # Get class labels from the LabelEncoder
    class_labels = label_encoder.classes_
    for i, sentence in enumerate(new_sentences_for_prediction):
        sentiment = predicted_sentiments[i]
        # Optionally, show probabilities for each class
        prob_dict = {class_labels[j]: prob for j, prob in enumerate(predicted_probabilities[i])}
        print(f"\nSentence: '{sentence}'")
        print(f"Predicted Sentiment: {sentiment}")
        print(f"Probabilities: {prob_dict}")

# ==============================================================================
## Task 3: Sentiment Analysis using Random Forest
# ==============================================================================
from sklearn.ensemble import RandomForestClassifier
# Define features (X) and target (y)
X = df['processed_text_refined']
y = df['airline_sentiment']

# Split the data into training and testing sets
# Using a similar test_size as your previous output (2928 / (11712 + 2928) = ~0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print("\nSentiment distribution in training set:")
print(y_train.value_counts(normalize=True))
print("\nSentiment distribution in test set:")
print(y_test.value_counts(normalize=True))


# TF-IDF Vectorization
# Use the same number of features (2000) as in your previous example
tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2)) # Added ngram_range for potentially better features

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"\nTF-IDF vectorization complete.")
print(f"Number of features (TF-IDF terms): {X_train_tfidf.shape[1]}")

# Train Random Forest model
print(f"\nTraining Random Forest model with class_weight='balanced'...")
# n_estimators: Number of trees in the forest. More trees generally improve performance
#               but increase computation time.
# class_weight='balanced': Addresses class imbalance, similar to your Logistic Regression
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
random_forest_model.fit(X_train_tfidf, y_train)

print(f"Model training complete.")

# Evaluate model performance
print(f"\nEvaluating model performance:")
y_pred = random_forest_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\n--- Using the Trained Model for Prediction ---")
new_texts = [
 "This airline is fantastic, loved the service!",
    "My flight was delayed and the experience was terrible.",
    "The journey was okay, nothing special.",
    "I had a really bad day, everything went wrong.",
    "Virgin America, you are the best airline!",
    "AmericanAir my flight was Cancelled Flightled, so frustrating."]

new_texts_tfidf = tfidf_vectorizer.transform(new_texts)
predictions = random_forest_model.predict(new_texts_tfidf)
print(f"\nPredictions for new texts: {predictions}")

# ==============================================================================
## Task 3:using Sparse Voting Ensemble (LR + XGBoost)
# ==============================================================================
print("\n--- Task 3: Using Sparse Voting Ensemble (LR + XGBoost) Sentiment Analysis ---")

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assume df is already loaded with processed text and airline_sentiment

# Define features (X) and target (y)
# Use the processed text column, checking for refined first
processed_text_column = 'processed_text_refined' if 'processed_text_refined' in df.columns else 'processed_text_sophisticated'

if processed_text_column not in df.columns or 'airline_sentiment' not in df.columns:
    print("Error: Required columns ('processed_text_refined'/'processed_text_sophisticated' or 'airline_sentiment') not found in DataFrame.")
else:
    X = df[processed_text_column]
    y = df['airline_sentiment']

    # Split data into training and testing sets (80/20 split with stratification)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Sentiment distribution in training set:\n{y_train.value_counts(normalize=True)}")
    print(f"Sentiment distribution in test set:\n{y_test.value_counts(normalize=True)}")


    # Feature Extraction: Convert text data into numerical features using TF-IDF
    # Keep output as sparse matrix, use parameters similar to previous tasks
    print("\nApplying TF-IDF Vectorization (keeping sparse)...")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5, sublinear_tf=True)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print("TF-IDF Vectorization complete.")
    print(f"Number of features (TF-IDF terms): {X_train_tfidf.shape[1]}")


    # Apply SMOTE only to the training data (after sparse TF-IDF)
    # SMOTE is compatible with sparse matrices
    print("\nApplying SMOTE resampling to sparse training data...")
    smote = SMOTE(sampling_strategy='not majority', random_state=42)
    X_train_resampled_sparse, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)
    print("SMOTE resampling complete.")
    print(f"Training samples after resampling: {X_train_resampled_sparse.shape[0]}")
    print(f"Sentiment distribution in resampled training set:\n{y_train_resampled.value_counts(normalize=True)}")


    # Define individual classifiers compatible with sparse input
    print("\nDefining individual classifiers (Logistic Regression and XGBoost)...")
    # Logistic Regression is sparse-compatible
    clf1 = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
    # XGBoost is sparse-compatible
    # use_label_encoder=False to avoid warning, eval_metric='mlogloss' suitable for multi-class
    clf2 = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    print("Individual classifiers defined.")


    # Create the Voting Ensemble Classifier (LR + XGBoost)
    # voting='soft' requires predict_proba from base estimators
    print("\nCreating Voting Ensemble Classifier (Logistic Regression and XGBoost)...")
    ensemble_clf_sparse = VotingClassifier(estimators=[('lr', clf1), ('xgb', clf2)], voting='soft', n_jobs=-1)
    print("Voting Ensemble Classifier created.")


    # Train the ensemble on the SMOTE-resampled sparse training data
    print("\nTraining the Sparse Voting Ensemble on resampled data...")
    ensemble_clf_sparse.fit(X_train_resampled_sparse, y_train_resampled)
    print("Voting Ensemble training complete.")


    # Predict on the original sparse test data
    print("\nPredicting on the original sparse test data...")
    y_pred_sparse_ensemble = ensemble_clf_sparse.predict(X_test_tfidf)
    print("Prediction complete.")


    # Evaluate the sparse ensemble model
    print("\n--- Evaluating Sparse Voting Ensemble Model ---")

    # Accuracy
    accuracy_sparse_ensemble = accuracy_score(y_test, y_pred_sparse_ensemble)
    print(f"Accuracy: {accuracy_sparse_ensemble:.4f}")

    # Classification Report (includes macro and weighted F1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_sparse_ensemble, zero_division=0))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm_sparse_ensemble = confusion_matrix(y_test, y_pred_sparse_ensemble)

    # Display confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_sparse_ensemble, annot=True, fmt='d', cmap='Blues', xticklabels=ensemble_clf_sparse.classes_, yticklabels=ensemble_clf_sparse.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Sparse Ensemble)')
    plt.show()

    print("\n--- Sparse Voting Ensemble Task Complete ---")

# ==============================================================================
## Task 3: By Hyperparameter Tuning and Tuned Sparse Voting Ensemble
# ==============================================================================
print("\n--- Task 3: Hyperparameter Tuning and Tuned Sparse Voting Ensemble ---")

# Import necessary libraries for tuning
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder


# Assume df is already loaded with processed text and airline_sentiment

# Define features (X) and target (y)
# Use the processed text column, checking for refined first
processed_text_column = 'processed_text_refined' if 'processed_text_refined' in df.columns else 'processed_text_sophisticated'

if processed_text_column not in df.columns or 'airline_sentiment' not in df.columns:
    print("Error: Required columns ('processed_text_refined'/'processed_text_sophisticated' or 'airline_sentiment') not found in DataFrame.")
else:
    X = df[processed_text_column]
    y = df['airline_sentiment']

    # Encode the target variable into numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data into training and testing sets (using the encoded target)
    # We will use the original string labels for SMOTE later, but encoded for model fitting
    X_train, X_test, y_train_original, y_test_original = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_train_encoded = label_encoder.transform(y_train_original)
    y_test_encoded = label_encoder.transform(y_test_original)


    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Sentiment distribution in original training set:\n{y_train_original.value_counts(normalize=True)}")
    print(f"Sentiment distribution in test set:\n{y_test_original.value_counts(normalize=True)}")


    # Feature Extraction: Convert text data into numerical features using TF-IDF (keeping sparse)
    # Using parameters similar to Task X/Task 7 for consistency
    print("\nApplying TF-IDF Vectorization (keeping sparse)...")
    # Re-fit_transform is necessary here if we want to ensure the vectorizer is fitted on THIS split
    # However, to use the SAME vectorizer as Task X, we should use the one fitted there.
    # Assuming tfidf_vectorizer is available globally from Task X
    if 'tfidf_vectorizer' in globals():
         X_train_tfidf = tfidf_vectorizer.transform(X_train) # Use transform if already fitted
         X_test_tfidf = tfidf_vectorizer.transform(X_test)
         print("Using existing TF-IDF Vectorizer from Task X.")
    else:
         # If not available, fit a new one (using parameters from Task X description)
         print("TF-IDF Vectorizer not found from Task X, fitting a new one.")
         tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5, sublinear_tf=True)
         X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
         X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print("TF-IDF Vectorization complete.")
    print(f"Number of features (TF-IDF terms): {X_train_tfidf.shape[1]}")

    # Apply SMOTE only to the training data (after sparse TF-IDF) for the final ensemble training
    # SMOTE works with encoded target variables
    print("\nApplying SMOTE resampling to sparse training data for final ensemble training...")
    smote = SMOTE(sampling_strategy='not majority', random_state=42)
    X_train_resampled_sparse, y_train_resampled_encoded = smote.fit_resample(X_train_tfidf, y_train_encoded)
    print("SMOTE resampling complete for final ensemble training.")
    print(f"Training samples after resampling: {X_train_resampled_sparse.shape[0]}")
    # Print distribution of original labels in resampled set for clarity
    print(f"Sentiment distribution in resampled training set:")
    # Calculate and print the distribution using value_counts on the decoded labels for clarity
    resampled_labels_decoded = label_encoder.inverse_transform(y_train_resampled_encoded)
    resampled_distribution = pd.Series(resampled_labels_decoded).value_counts(normalize=True)
    print(resampled_distribution)


    # --- Hyperparameter Tuning ---

    print("\n--- Performing Hyperparameter Tuning ---")

    # Define cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # 1. Tune Logistic Regression (using encoded target)
    print("\nTuning Logistic Regression...")
    lr_param_dist = {
        'C': np.logspace(-3, 3, 7), # Example: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        'penalty': ['l1', 'l2'] # Compatible with solver='liblinear'
    }
    lr = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000) # Specify solver and max_iter
    lr_random_search = RandomizedSearchCV(
        lr,
        param_distributions=lr_param_dist,
        n_iter=10, # Number of parameter settings that are sampled. Increase for better results.
        scoring='accuracy',
        cv=cv_strategy,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    # Fit on the original sparse training data (X_train_tfidf) with encoded labels
    lr_random_search.fit(X_train_tfidf, y_train_encoded)
    best_lr = lr_random_search.best_estimator_

    print(f"\nBest parameters for Logistic Regression: {lr_random_search.best_params_}")
    print(f"Best cross-validation accuracy for Logistic Regression: {lr_random_search.best_score_:.4f}")


    # 2. Tune XGBoost Classifier (using encoded target)
    print("\nTuning XGBoost Classifier...")
    xgb_param_dist = {
        'n_estimators': [100, 200, 300], # Number of boosting rounds
        'max_depth': [3, 5, 7, 9], # Maximum depth of a tree
        'learning_rate': [0.01, 0.05, 0.1, 0.2], # Boosting learning rate
        'subsample': [0.6, 0.8, 1.0], # Subsample ratio of the training instances
        'colsample_bytree': [0.6, 0.8, 1.0], # Subsample ratio of columns when constructing each tree
        'gamma': [0, 0.1, 0.2, 0.3] # Minimum loss reduction required to make a further partition on a leaf node
    }
    xgb_base = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    xgb_random_search = RandomizedSearchCV(
        xgb_base, # Use the base classifier for tuning
        param_distributions=xgb_param_dist,
        n_iter=10, # Number of parameter settings that are sampled. Increase for better results.
        scoring='accuracy',
        cv=cv_strategy,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    # Fit on the original sparse training data (X_train_tfidf) with encoded labels
    xgb_random_search.fit(X_train_tfidf, y_train_encoded)
    best_xgb = xgb_random_search.best_estimator_ # This will be the tuned XGBoost classifier

    print(f"\nBest parameters for XGBoost: {xgb_random_search.best_params_}")
    print(f"Best cross-validation accuracy for XGBoost: {xgb_random_search.best_score_:.4f}")

    print("\n--- Hyperparameter Tuning Complete ---")


    # --- Re-create and Evaluate Voting Ensemble with Tuned Models ---

    print("\n--- Training and Evaluating Tuned Sparse Voting Ensemble ---")

    # Define individual tuned classifiers
    print("\nDefining individual tuned classifiers...")
    # Use the best estimators found during random search
    tuned_lr = best_lr
    tuned_xgb = best_xgb
    print("Individual tuned classifiers defined.")


    # Create the Voting Ensemble Classifier (LR + XGBoost) with tuned models
    # voting='soft' requires predict_proba from base estimators
    print("\nCreating Voting Ensemble Classifier (Tuned Logistic Regression and Tuned XGBoost)...")
    # The individual estimators within VotingClassifier need to be compatible with the data used for *ensemble training*
    # Since the ensemble is trained on SMOTE-resampled sparse data, the estimators must be sparse-compatible.
    # LR and XGBoost are sparse-compatible.
    ensemble_clf_tuned_sparse = VotingClassifier(estimators=[('lr', tuned_lr), ('xgb', tuned_xgb)], voting='soft', n_jobs=-1)
    print("Voting Ensemble Classifier created.")


    # Train the tuned ensemble on the SMOTE-resampled sparse training data (using encoded labels)
    print("\nTraining the Tuned Sparse Voting Ensemble on resampled data...")
    # Use the X_train_resampled_sparse and y_train_resampled_encoded obtained after SMOTE
    ensemble_clf_tuned_sparse.fit(X_train_resampled_sparse, y_train_resampled_encoded)
    print("Voting Ensemble training complete.")


    # Predict on the original sparse test data
    print("\nPredicting on the original sparse test data...")
    # Predictions will be encoded numerical labels
    y_pred_tuned_sparse_ensemble_encoded = ensemble_clf_tuned_sparse.predict(X_test_tfidf)

    # Decode the predictions back to original labels for evaluation
    y_pred_tuned_sparse_ensemble = label_encoder.inverse_transform(y_pred_tuned_sparse_ensemble_encoded)

    print("Prediction complete.")


    # Evaluate the tuned sparse ensemble model (using original test labels and decoded predictions)
    print("\n--- Evaluating Tuned Sparse Voting Ensemble Model ---")

    # Accuracy (compare original test labels with decoded predictions)
    accuracy_tuned_sparse_ensemble = accuracy_score(y_test_original, y_pred_tuned_sparse_ensemble)
    print(f"Accuracy: {accuracy_tuned_sparse_ensemble:.4f}")

    # Classification Report (includes macro and weighted F1)
    print("\nClassification Report:")
    # Use original test labels and decoded predictions
    print(classification_report(y_test_original, y_pred_tuned_sparse_ensemble, zero_division=0))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    # Use original test labels and decoded predictions
    cm_tuned_sparse_ensemble = confusion_matrix(y_test_original, y_pred_tuned_sparse_ensemble)

    # Display confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_tuned_sparse_ensemble, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Tuned Sparse Ensemble)')
    plt.show()

    print("\n--- Tuned Sparse Voting Ensemble Task Complete ---")

# ==============================================================================
## Task 3: By Hyperparameter Tuning and Tuned Stacking Ensemble
# ==============================================================================
print("\n--- Task 3: Hyperparameter Tuning and Tuned Stacking Ensemble ---")

# Import necessary libraries for tuning and stacking
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier # Import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder


# Assume df is already loaded with processed text and airline_sentiment

# Define features (X) and target (y)
# Use the processed text column, checking for refined first
processed_text_column = 'processed_text_refined' if 'processed_text_refined' in df.columns else 'processed_text_sophisticated'

if processed_text_column not in df.columns or 'airline_sentiment' not in df.columns:
    print("Error: Required columns ('processed_text_refined'/'processed_text_sophisticated' or 'airline_sentiment') not found in DataFrame.")
else:
    X = df[processed_text_column]
    y = df['airline_sentiment']

    # Encode the target variable into numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data into training and testing sets (using the encoded target)
    # We will use the original string labels for SMOTE later, but encoded for model fitting
    X_train, X_test, y_train_original, y_test_original = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_train_encoded = label_encoder.transform(y_train_original)
    y_test_encoded = label_encoder.transform(y_test_original)


    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Sentiment distribution in original training set:\n{y_train_original.value_counts(normalize=True)}")
    print(f"Sentiment distribution in test set:\n{y_test_original.value_counts(normalize=True)}")


    # Feature Extraction: Convert text data into numerical features using TF-IDF (keeping sparse)
    # Using parameters similar to Task X/Task 7 for consistency
    print("\nApplying TF-IDF Vectorization (keeping sparse)...")
    # Re-fit_transform is necessary here if we want to ensure the vectorizer is fitted on THIS split
    # However, to use the SAME vectorizer as Task X, we should use the one fitted there.
    # Assuming tfidf_vectorizer is available globally from Task X
    if 'tfidf_vectorizer' in globals():
         X_train_tfidf = tfidf_vectorizer.transform(X_train) # Use transform if already fitted
         X_test_tfidf = tfidf_vectorizer.transform(X_test)
         print("Using existing TF-IDF Vectorizer from Task X.")
    else:
         # If not available, fit a new one (using parameters from Task X description)
         print("TF-IDF Vectorizer not found from Task X, fitting a new one.")
         tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5, sublinear_tf=True)
         X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
         X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print("TF-IDF Vectorization complete.")
    print(f"Number of features (TF-IDF terms): {X_train_tfidf.shape[1]}")

    # Apply SMOTE only to the training data (after sparse TF-IDF) for the final ensemble training
    # SMOTE works with encoded target variables
    print("\nApplying SMOTE resampling to sparse training data for final ensemble training...")
    smote = SMOTE(sampling_strategy='not majority', random_state=42)
    X_train_resampled_sparse, y_train_resampled_encoded = smote.fit_resample(X_train_tfidf, y_train_encoded)
    print("SMOTE resampling complete for final ensemble training.")
    print(f"Training samples after resampling: {X_train_resampled_sparse.shape[0]}")
    # Print distribution of original labels in resampled set for clarity
    print(f"Sentiment distribution in resampled training set:")
    # Calculate and print the distribution using value_counts on the decoded labels for clarity
    resampled_labels_decoded = label_encoder.inverse_transform(y_train_resampled_encoded)
    resampled_distribution = pd.Series(resampled_labels_decoded).value_counts(normalize=True)
    print(resampled_distribution)


    # --- Hyperparameter Tuning ---
    # Assuming tuning was done and best_lr, best_xgb are available from previous execution
    # If not, uncomment and run the tuning block below:

    print("\n--- Using Tuned Models from Previous Steps (Assuming Tuning Was Run) ---")
    # Define individual base classifiers with parameters from successful tuning (Task 8 output)
    # These are placeholders based on typical results, replace with actual best params if needed
    # For this execution, we assume best_lr and best_xgb are globally available
    if 'best_lr' not in globals() or 'best_xgb' not in globals():
         print("Warning: Tuned models not found. Running tuning now...")

         print("\n--- Performing Hyperparameter Tuning ---")

         # Define cross-validation strategy
         cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

         # 1. Tune Logistic Regression (using encoded target)
         print("\nTuning Logistic Regression...")
         lr_param_dist = {
             'C': np.logspace(-3, 3, 7), # Example: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
             'penalty': ['l1', 'l2'] # Compatible with solver='liblinear'
         }
         lr = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000) # Specify solver and max_iter
         lr_random_search = RandomizedSearchCV(
             lr,
             param_distributions=lr_param_dist,
             n_iter=10, # Number of parameter settings that are sampled. Increase for better results.
             scoring='accuracy',
             cv=cv_strategy,
             random_state=42,
             n_jobs=-1,
             verbose=1
         )
         # Fit on the original sparse training data (X_train_tfidf) with encoded labels
         lr_random_search.fit(X_train_tfidf, y_train_encoded)
         best_lr = lr_random_search.best_estimator_

         print(f"\nBest parameters for Logistic Regression: {lr_random_search.best_params_}")
         print(f"Best cross-validation accuracy for Logistic Regression: {lr_random_search.best_score_:.4f}")


         # 2. Tune XGBoost Classifier (using encoded target)
         print("\nTuning XGBoost Classifier...")
         xgb_param_dist = {
             'n_estimators': [100, 200, 300], # Number of boosting rounds
             'max_depth': [3, 5, 7, 9], # Maximum depth of a tree
             'learning_rate': [0.01, 0.05, 0.1, 0.2], # Boosting learning rate
             'subsample': [0.6, 0.8, 1.0], # Subsample ratio of the training instances
             'colsample_bytree': [0.6, 0.8, 1.0], # Subsample ratio of columns when constructing each tree
             'gamma': [0, 0.1, 0.2, 0.3] # Minimum loss reduction required to make a further partition on a leaf node
         }
         xgb_base = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, random_state=42)
         xgb_random_search = RandomizedSearchCV(
             xgb_base, # Use the base classifier for tuning
             param_distributions=xgb_param_dist,
             n_iter=10, # Number of parameter settings that are sampled. Increase for better results.
             scoring='accuracy',
             cv=cv_strategy,
             random_state=42,
             n_jobs=-1,
             verbose=1
         )
         # Fit on the original sparse training data (X_train_tfidf) with encoded labels
         xgb_random_search.fit(X_train_tfidf, y_train_encoded)
         best_xgb = xgb_random_search.best_estimator_ # This will be the tuned XGBoost classifier

         print(f"\nBest parameters for XGBoost: {xgb_random_search.best_params_}")
         print(f"Best cross-validation accuracy for XGBoost: {xgb_random_search.best_score_:.4f}")

         print("\n--- Hyperparameter Tuning Complete ---")

    tuned_lr = best_lr
    tuned_xgb = best_xgb
    print("Individual tuned classifiers defined.")


    # --- Create and Evaluate Tuned Stacking Ensemble ---

    print("\n--- Training and Evaluating Tuned Sparse Stacking Ensemble ---")

    # Define base estimators
    base_estimators = [
        ('lr', tuned_lr),
        ('xgb', tuned_xgb)
    ]

    # Define final estimator (meta-model)
    # Ensure compatibility with sparse input
    final_estimator = LogisticRegression(C=1.0, solver='liblinear', random_state=42, max_iter=1000)


    # Create the Stacking Classifier
    # Use cv=3 internally for stacking
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=3,
        n_jobs=-1,
        passthrough=False # Set to False to only use predictions of base estimators
    )
    print("Stacking Ensemble Classifier created.")


    # Train the Stacking Ensemble on the SMOTE-resampled sparse training data
    print("\nTraining the Tuned Sparse Stacking Ensemble on resampled data...")
    # Use the X_train_resampled_sparse and y_train_resampled_encoded obtained after SMOTE
    stacking_clf.fit(X_train_resampled_sparse, y_train_resampled_encoded)
    print("Stacking Ensemble training complete.")


    # Predict on the original sparse test data
    print("\nPredicting on the original sparse test data...")
    # Predictions will be encoded numerical labels
    y_pred_tuned_sparse_ensemble_encoded = stacking_clf.predict(X_test_tfidf)

    # Decode the predictions back to original labels for evaluation
    y_pred_tuned_sparse_ensemble = label_encoder.inverse_transform(y_pred_tuned_sparse_ensemble_encoded)

    print("Prediction complete.")


    # Evaluate the tuned sparse ensemble model (using original test labels and decoded predictions)
    print("\n--- Evaluating Tuned Sparse Stacking Ensemble Model ---")

    # Accuracy (compare original test labels with decoded predictions)
    accuracy_tuned_sparse_ensemble = accuracy_score(y_test_original, y_pred_tuned_sparse_ensemble)
    print(f"Accuracy: {accuracy_tuned_sparse_ensemble:.4f}")

    # Classification Report (includes macro and weighted F1)
    print("\nClassification Report:")
    # Use original test labels and decoded predictions
    print(classification_report(y_test_original, y_pred_tuned_sparse_ensemble, zero_division=0))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    # Use original test labels and decoded predictions
    cm_tuned_sparse_ensemble = confusion_matrix(y_test_original, y_pred_tuned_sparse_ensemble)

    # Display confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_tuned_sparse_ensemble, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Tuned Sparse Stacking Ensemble)')
    plt.show()

    print("\n--- Tuned Sparse Stacking Ensemble Task Complete ---")

    # --- Using the Trained Stacking Ensemble for Prediction on Sample Sentences ---

    print("\n--- Using the Trained Stacking Ensemble for Prediction ---")

    # Example sentences to predict sentiment on (using the trained vectorizer and stacking classifier)
    new_sentences_for_prediction = [
        "This airline is fantastic, loved the service!",
        "My flight was delayed and the experience was terrible.",
        "The journey was okay, nothing special.",
        "I had a really bad day, everything went wrong.",
        "Virgin America, you are the best airline!",
        "AmericanAir my flight was Cancelled Flightled, so frustrating."
    ]

    # Preprocess the new sentences (assuming preprocess_text_for_ml_refined is defined in the environment)
    # If not, you would need to include or import the function here.
    # For this example, we assume it's available from the preprocessing cell.
    processed_new_sentences_for_prediction = [preprocess_text_for_ml_refined(s) for s in new_sentences_for_prediction]

    # Transform new sentences using the *same* TF-IDF vectorizer fitted on training data
    new_sentences_tfidf_for_prediction = tfidf_vectorizer.transform(processed_new_sentences_for_prediction)

    # Predict sentiment using the trained stacking classifier (predictions will be encoded)
    predicted_sentiments_encoded = stacking_clf.predict(new_sentences_tfidf_for_prediction)
    # StackingClassifier does not have predict_proba on the final output,
    # but base estimators do. If probabilities are needed, predict on base estimators.
    # For simplicity, we'll just provide the final prediction.
    # predicted_probabilities = stacking_clf.predict_proba(new_sentences_tfidf_for_prediction) # This will raise an error

    # Decode the predictions back to original labels
    predicted_sentiments = label_encoder.inverse_transform(predicted_sentiments_encoded)

    print("\nSentiment Predictions for New Sentences:")
    # Get class labels from the LabelEncoder
    class_labels = label_encoder.classes_
    for i, sentence in enumerate(new_sentences_for_prediction):
        sentiment = predicted_sentiments[i]
        # We cannot easily get probabilities from the StackingClassifier's final prediction
        # print(f"Probabilities: {prob_dict}") # Removed probability print

        print(f"\nSentence: '{sentence}'")
        print(f"Predicted Sentiment: {sentiment}")

    print("\n--- Stacking Ensemble Prediction on Samples Complete ---")

"""i also tried using  Voting Ensemble Sentiment Analysis (SMOTE,LR,SVC,XGBOOST) but that crashed after running for 2 hours and it consumed a huge ammount of computing unita as well, so I think it is not effeicient to perform this task using that"""

# ==============================================================================
## Task 3: Efficient Voting Ensemble Sentiment Analysis (LR, NB, XGBOOST)
# ==============================================================================
print("\n--- Task 3: Efficient Voting Ensemble Sentiment Analysis ---")

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define features (X) and target (y)
processed_text_column = 'processed_text_refined' if 'processed_text_refined' in df.columns else 'processed_text_sophisticated'

if processed_text_column not in df.columns or 'airline_sentiment' not in df.columns:
    print("Error: Required columns not found in DataFrame.")
else:
    X = df[processed_text_column]
    y = df['airline_sentiment']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Sentiment distribution in training set:\n{y_train.value_counts(normalize=True)}")
    print(f"Sentiment distribution in test set:\n{y_test.value_counts(normalize=True)}")

    # TF-IDF Vectorization
    print("\nApplying TF-IDF Vectorization...")
    if 'tfidf_vectorizer' in globals():
        X_train_tfidf = tfidf_vectorizer.transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        print("Using existing TF-IDF Vectorizer.")
    else:
        print("Fitting new TF-IDF Vectorizer.")
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5, sublinear_tf=True, stop_words='english')
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print("TF-IDF Vectorization complete.")
    print(f"Number of features (TF-IDF terms): {X_train_tfidf.shape[1]}")

    # Define individual classifiers
    clf1 = LogisticRegression(class_weight='balanced', random_state=42, solver='liblinear', max_iter=1000)
    clf2 = MultinomialNB()
    clf3 = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, random_state=42)

    # Create Voting Ensemble
    ensemble_clf = VotingClassifier(estimators=[('lr', clf1), ('nb', clf2), ('xgb', clf3)], voting='soft', n_jobs=-1)

    # Train
    print("\nTraining Voting Ensemble...")
    ensemble_clf.fit(X_train_tfidf, y_train)
    print("Training complete.")

    # Predict
    print("\nPredicting on test data...")
    y_pred = ensemble_clf.predict(X_test_tfidf)
    print("Prediction complete.")

    # Evaluate
    print("\n--- Evaluation ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ensemble_clf.classes_, yticklabels=ensemble_clf.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    print("\n--- Task Complete ---")

# ==============================================================================
## Task 3 (SBERT + XGBoost): Sentiment Analysis with SBERT Embeddings
# ==============================================================================
print("\n--- Task 3 (SBERT + XGBoost): Sentiment Analysis with SBERT Embeddings ---")

# Install necessary libraries if not already installed
#!pip install sentence-transformers xgboost scikit-learn pandas matplotlib seaborn

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assume df is already loaded with processed text and airline_sentiment

# Define features (X) and target (y)
# Use the processed text column, checking for refined first
processed_text_column = 'processed_text_refined' if 'processed_text_refined' in df.columns else 'processed_text_sophisticated'

if processed_text_column not in df.columns or 'airline_sentiment' not in df.columns:
    print("Error: Required columns ('processed_text_refined'/'processed_text_sophisticated' or 'airline_sentiment') not found in DataFrame.")
else:
    X = df[processed_text_column]
    y = df['airline_sentiment']

    # Split data into training and testing sets (80/20 split with stratification)
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train_original, y_test_original = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Sentiment distribution in original training set:\n{y_train_original.value_counts(normalize=True)}")
    print(f"Sentiment distribution in test set:\n{y_test_original.value_counts(normalize=True)}")

    # Encode the target variable into numerical labels for XGBoost
    print("\nEncoding target variable...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_original)
    y_test_encoded = label_encoder.transform(y_test_original)
    print("Target variable encoding complete.")


    # Generate SBERT Embeddings
    print("\nGenerating SBERT embeddings using 'paraphrase-MiniLM-L6-v2'...")
    # Load the SBERT model
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Generate embeddings for training and testing data
    X_train_embeddings = sbert_model.encode(X_train.tolist(), show_progress_bar=True)
    X_test_embeddings = sbert_model.encode(X_test.tolist(), show_progress_bar=True)

    print("SBERT embedding generation complete.")
    print(f"Training embeddings shape: {X_train_embeddings.shape}")
    print(f"Testing embeddings shape: {X_test_embeddings.shape}")


    # Compute sample weights for handling class imbalance
    print("\nComputing sample weights for class imbalance...")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_original)
    print("Sample weight computation complete.")


    # Define and Train XGBoost Classifier
    print("\nDefining and training XGBoost Classifier...")
    xgb_classifier_sbert = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=3, # Specify num_class for multi-class
        learning_rate=0.1,
        max_depth=5,
        n_estimators=300,
        use_label_encoder=False, # To avoid warning
        random_state=42
    )

    # Train the XGBoost classifier on SBERT embeddings with sample weights
    xgb_classifier_sbert.fit(X_train_embeddings, y_train_encoded, sample_weight=sample_weights)
    print("XGBoost Classifier training complete.")


    # Predict on the test data
    print("\nPredicting on the test data...")
    y_pred_encoded = xgb_classifier_sbert.predict(X_test_embeddings)

    # Decode the predictions back to original labels
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    print("Prediction complete.")


    # Evaluate the model
    print("\n--- Evaluating SBERT + XGBoost Model ---")

    # Accuracy
    accuracy = accuracy_score(y_test_original, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Classification Report (includes macro and weighted F1)
    print("\nClassification Report:")
    print(classification_report(y_test_original, y_pred, zero_division=0))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_original, y_pred)

    # Display confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (SBERT + XGBoost)')
    plt.show()

    print("\n--- SBERT + XGBoost Sentiment Analysis Task Complete ---")

# ==============================================================================
## Task 3 (Optimized SBERT + XGBoost): Sentiment Analysis with Dimensionality Reduction & Tuning
# ==============================================================================
print("\n--- Task 3 (Optimized SBERT + XGBoost) ---")

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define features (X) and target (y)
processed_text_column = 'processed_text_refined' if 'processed_text_refined' in df.columns else 'processed_text_sophisticated'

if processed_text_column not in df.columns or 'airline_sentiment' not in df.columns:
    print("Error: Required columns not found in DataFrame.")
else:
    X = df[processed_text_column]
    y = df['airline_sentiment']

    # Train-test split
    print("\nSplitting data...")
    X_train, X_test, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Encode target
    print("\nEncoding target variable...")
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_orig)
    y_test = le.transform(y_test_orig)

    # SBERT Embeddings
    print("\nGenerating SBERT embeddings...")
    sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    X_train_embed = sbert.encode(X_train.tolist(), show_progress_bar=True)
    X_test_embed = sbert.encode(X_test.tolist(), show_progress_bar=True)

    # Dimensionality Reduction
    print("\nApplying TruncatedSVD to reduce embedding dimensionality...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_train_reduced = svd.fit_transform(X_train_embed)
    X_test_reduced = svd.transform(X_test_embed)

    # Compute sample weights
    print("\nComputing sample weights...")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_orig)

    # Define and train XGBoost
    print("\nTraining XGBoost classifier...")
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=3,
        learning_rate=0.05,
        max_depth=6,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=1,
        use_label_encoder=False,
        random_state=42
    )

    xgb_clf.fit(X_train_reduced, y_train, sample_weight=sample_weights)

    # Predict and decode
    print("\nMaking predictions...")
    y_pred_enc = xgb_clf.predict(X_test_reduced)
    y_pred = le.inverse_transform(y_pred_enc)

    # Evaluate
    print("\n--- Evaluation ---")
    acc = accuracy_score(y_test_orig, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test_orig, y_pred, zero_division=0))

    cm = confusion_matrix(y_test_orig, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (SBERT + XGBoost with SVD)')
    plt.show()

    print("\n--- Task Complete ---")

# ==============================================================================
## Task 3 (SBERT + XGBoost with Oversampling + mpnet Embeddings)
# ==============================================================================
print("\n--- Task 3 (SBERT + XGBoost with Oversampling + mpnet Embeddings) ---")

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define features (X) and target (y)
processed_text_column = 'processed_text_refined' if 'processed_text_refined' in df.columns else 'processed_text_sophisticated'

if processed_text_column not in df.columns or 'airline_sentiment' not in df.columns:
    print("Error: Required columns not found in DataFrame.")
else:
    X = df[processed_text_column]
    y = df['airline_sentiment']

    # Split original data
    print("\nSplitting data...")
    X_train_orig, X_test, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Combine into dataframe for oversampling
    print("\nApplying oversampling on neutral and positive classes...")
    train_df = pd.DataFrame({'text': X_train_orig, 'label': y_train_orig})
    majority = train_df[train_df.label == 'negative']
    neutral = train_df[train_df.label == 'neutral']
    positive = train_df[train_df.label == 'positive']

    neutral_upsampled = resample(neutral, replace=True, n_samples=len(majority), random_state=42)
    positive_upsampled = resample(positive, replace=True, n_samples=len(majority), random_state=42)
    train_balanced = pd.concat([majority, neutral_upsampled, positive_upsampled])

    X_train = train_balanced.text.tolist()
    y_train_orig = train_balanced.label.tolist()

    # Encode target
    print("\nEncoding target variable...")
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_orig)
    y_test = le.transform(y_test_orig)

    # SBERT Embeddings (mpnet)
    print("\nGenerating SBERT embeddings with mpnet model...")
    sbert = SentenceTransformer('paraphrase-mpnet-base-v2')
    X_train_embed = sbert.encode(X_train, show_progress_bar=True)
    X_test_embed = sbert.encode(X_test.tolist(), show_progress_bar=True)

    # Dimensionality Reduction
    print("\nApplying TruncatedSVD to reduce embedding dimensionality...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_train_reduced = svd.fit_transform(X_train_embed)
    X_test_reduced = svd.transform(X_test_embed)

    # Define and train XGBoost
    print("\nTraining XGBoost classifier...")
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=3,
        learning_rate=0.05,
        max_depth=6,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=1,
        use_label_encoder=False,
        random_state=42
    )

    xgb_clf.fit(X_train_reduced, y_train)

    # Predict and decode
    print("\nMaking predictions...")
    y_pred_enc = xgb_clf.predict(X_test_reduced)
    y_pred = le.inverse_transform(y_pred_enc)

    # Evaluate
    print("\n--- Evaluation ---")
    acc = accuracy_score(y_test_orig, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test_orig, y_pred, zero_division=0))

    cm = confusion_matrix(y_test_orig, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (SBERT mpnet + XGBoost + Oversampling)')
    plt.show()

    print("\n--- Task Complete ---")

# ==============================================================================
## Task 4: Named Entity Recognition (NER)
# ==============================================================================

from transformers import pipeline
from datasets import Dataset # You'll need to install the 'datasets' library if you haven't: pip install datasets
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd # Import pandas to check for non-string types

print("\n--- Task 4: Named Entity Recognition (NER) with Hugging Face Transformers ---")

# 1. Initialize the NER pipeline
# You can specify the model, but 'dslim/bert-base-NER' is a good default for general NER
# Make sure to set device to 0 for GPU, or -1 for CPU if no GPU is available
try:
    # Try to use GPU if available
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", device=0, aggregation_strategy="simple")
    print("NER pipeline initialized on GPU (cuda:0).")
except RuntimeError:
    # Fallback to CPU if GPU is not available
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", device=-1, aggregation_strategy="simple")
    print("NER pipeline initialized on CPU.")


# 2. Prepare your entire 'processed_text_refined' column as a Dataset
# Filter out non-string and empty values
valid_processed_texts = [text for text in df['processed_text_refined'] if isinstance(text, str) and text.strip()]


# Create a Hugging Face Dataset from your list of texts
# We'll put them in a dictionary format that Dataset expects
hf_dataset = Dataset.from_dict({'text': valid_processed_texts})

print(f"\nApplying NER to all {len(hf_dataset)} valid tweets in the dataset...")

# 3. Use KeyDataset for efficient batch processing with the pipeline
# This will iterate over the 'text' column of your dataset in batches,
# which is much more efficient on a GPU.
all_ner_results = []
# Adjust batch_size if needed based on available memory (especially on GPU)
batch_size_ner = 16
for out in ner_pipeline(KeyDataset(hf_dataset, "text"), batch_size=batch_size_ner):
    all_ner_results.append(out)

print("NER application complete for all valid tweets.")

# 4. Display results (you might want to store these in a new column or a separate structure)
print("\nNamed Entities Found in Tweets (Hugging Face Transformers):")

# Create a list to store results for easy analysis (e.g., converting to DataFrame)
extracted_entities_list = []

# Iterate through the results and the corresponding valid texts
for i, entities_in_tweet in enumerate(all_ner_results):
    # Get the original valid tweet text (matching the order in hf_dataset)
    original_tweet = valid_processed_texts[i]
    if entities_in_tweet: # Only process if entities were found
        # print(f"\nTweet: '{original_tweet}'") # Optional: print the tweet text
        for entity in entities_in_tweet:
            entity_info = {
                'tweet_text': original_tweet, # Store the tweet text
                'entity_text': entity['word'],
                'entity_label': entity['entity_group'], # Corrected back to 'entity_group'
                'score': entity['score'],
                'start_char': entity['start'],
                'end_char': entity['end']
            }
            extracted_entities_list.append(entity_info)
            # print(f"- Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.4f}") # Optional: print entities

# Convert the list of extracted entities into a DataFrame for easier analysis
if extracted_entities_list:
    entities_df = pd.DataFrame(extracted_entities_list)
    print(f"\nExtracted {len(entities_df)} entities from the dataset.")
    print("\nSample of Extracted Entities DataFrame:")
    display(entities_df.head())

    # Optional: Further analysis on extracted entities
    print("\nTop 10 Most Frequent Entity Labels:")
    display(entities_df['entity_label'].value_counts().head(10))

    print("\nTop 10 Most Frequent Entities (Words):")
    display(entities_df['entity_text'].value_counts().head(10))

else:
    print("\nNo entities were extracted from the valid tweets.")

# ==============================================================================
## Task 5: Create a Word Cloud
# ==============================================================================
print("\n--- Task 5: Creating a Word Cloud ---")

# Join all processed text from the 'processed_text_refined' column into a single string
wordcloud_text = " ".join(df['processed_text_refined'].astype(str).tolist())


# Create the word cloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off') # Hide the axes
plt.show()
