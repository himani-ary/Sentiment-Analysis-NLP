import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Step 1: Data Collection
# Assuming we have collected customer reviews from an e-commerce website and stored them in a variable called 'reviews'

# Step 2: Data Preprocessing
# Download stopwords corpus (needed only once)
nltk.download('stopwords')

# Define a function to perform text preprocessing
def preprocess_text(text):
    # Remove special characters
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Apply preprocessing to each review in the dataset
preprocessed_reviews = [preprocess_text(review) for review in reviews]

# Step 3: Feature Extraction
# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on preprocessed reviews and transform the reviews into numerical representations
X = vectorizer.fit_transform(preprocessed_reviews)

# Step 4: Training a Sentiment Analysis Model
# Assuming we have the corresponding sentiment labels for the reviews stored in a variable called 'labels'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Create a LinearSVC classifier
classifier = LinearSVC()

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Step 5: Evaluation
# Make predictions on the testing data
y_pred = classifier.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 6: Predicting Sentiment
# Assuming an unseen review stored in a variable called 'new_review'

# Preprocess the new review
preprocessed_new_review = preprocess_text(new_review)

# Transform the preprocessed new review into a numerical representation using the same vectorizer
new_review_vectorized = vectorizer.transform([preprocessed_new_review])

# Make a prediction on the new review
predicted_sentiment = classifier.predict(new_review_vectorized)

# Print the predicted sentiment
print("Predicted Sentiment:", predicted_sentiment)
