# Fake News Detection using Python and Machine Learning 

## Overview
This tutorial demonstrates building a **fake news classifier** using Natural Language Processing (NLP) and Machine Learning. The project identifies whether news articles are reliable or unreliable (fake) and deploys the model as a web application using Streamlit.

## Problem Statement
**Goal:** Classify news articles as reliable or unreliable (fake news)

**Application:**
- Combat misinformation spread
- Help readers verify news authenticity
- Automated content moderation
- Media literacy tool

**Data Source:** Kaggle Fake News Detection Dataset

---

## DATASET

### Source
- **Platform:** Kaggle
- **Dataset Name:** Fake News Detection
- **Files:** train.csv, test.csv

### Dataset Structure

**Columns (5 features):**
1. **id** - Unique identifier for each article
2. **title** - Article headline
3. **author** - Article author name
4. **text** - Article content (main feature)
5. **label** - Target variable
   - **0** = Reliable (real news)
   - **1** = Unreliable (fake news)

**Records:** 20,800 articles

### Columns Used for Analysis
- **text** - Article content (primary feature)
- **label** - Classification target

**Why only these two?**
- Text contains all necessary information for classification
- Title, author, and ID are not predictive features
- Reduces complexity and overfitting risk

---

## COMPLETE CODE WALKTHROUGH

### Step 1: Load Dataset

```python
import pandas as pd

# Load training data
df = pd.read_csv('train.csv')

# View first 5 rows
df.head()
```

**Output:**
```
   id     title          author                text  label
0   0  House Dem  Darrell Lucus  House Dem Aide...      1
1   1  FLYNN: ...  Daniel J. Flynn  Ever get the...   0
2   2  Why the...  Consortiumnews.com  Why the...   1
```

---

### Step 2: Data Exploration

**Check data types:**
```python
df.info()
```

**Output:**
```
RangeIndex: 20800 entries
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype 
 0   id      20800 non-null  int64 
 1   title   20242 non-null  object
 2   author  18843 non-null  object
 3   text    20761 non-null  object
 4   label   20800 non-null  int64
```

**Statistical description:**
```python
df.describe()
```

**Result:** Dataset is balanced (50% fake, 50% real)

---

### Step 3: Handle Missing Values

**Check for null values:**
```python
df.isnull().sum()
```

**Output:**
```
id         0
title    558
author  1957
text      39
label      0
```

**Solution: Fill with empty strings**
```python
df = df.fillna('')

# Verify
df.isnull().sum()  # All zeros now
```

---

### Step 4: Feature Selection

```python
# Check columns
df.columns

# Drop unnecessary columns
df = df.drop(['id', 'title', 'author'], axis=1)

# Verify - only text and label remain
df.head()
```

---

### Step 5: Text Preprocessing Function

```python
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

port_stem = PorterStemmer()

def stemming(content):
    # Remove special characters (keep only letters)
    con = re.sub('[^a-zA-Z]', ' ', content)
    
    # Convert to lowercase
    con = con.lower()
    
    # Split into words
    con = con.split()
    
    # Remove stopwords and stem
    con = [
        port_stem.stem(word) 
        for word in con 
        if word not in stopwords.words('english')
    ]
    
    # Join back into string
    con = ' '.join(con)
    
    return con
```

**Example transformation:**
```
Input:  "Hi this is Chandu!"
Output: "hi chandu"
```

---

### Step 6: Apply Preprocessing

```python
# Apply to all text
df['text'] = df['text'].apply(stemming)

# Processing time: 10-20 minutes for 20,800 articles
```

---

### Step 7: Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Features and target
X = df['text']
y = df['label']

# Split 80-20
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print(f"Training samples: {len(X_train)}")  # 16,640
print(f"Testing samples: {len(X_test)}")    # 4,160
```

---

### Step 8: TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create and fit vectorizer
vector = TfidfVectorizer()

# Fit on training data
X_train = vector.fit_transform(X_train)

# Transform test data (don't fit!)
X_test = vector.transform(X_test)

print(X_train.shape)  # (16640, vocabulary_size)
```

**What is TF-IDF?**
```
TF-IDF = TF × IDF

TF = Word count in document / Total words in document
IDF = log(Total documents / Documents containing word)
```

---

### Step 9: Train Model

```python
from sklearn.tree import DecisionTreeClassifier

# Create model
model = DecisionTreeClassifier(random_state=0)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

---

### Step 10: Evaluate Performance

```python
# Calculate accuracy
accuracy = model.score(X_test, y_test)

print(f"Accuracy: {accuracy:.2%}")
# Output: Accuracy: 88%
```

---

### Step 11: Save Models

```python
import pickle

# Save vectorizer
pickle.dump(vector, open('vector.pkl', 'wb'))

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
```

---

### Step 12: Load and Predict

```python
# Load saved models
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def fake_news(news):
    # Preprocess
    news = stemming(news)
    
    # Convert to array
    input_data = [news]
    
    # Vectorize
    vector_form1 = vector_form.transform(input_data)
    
    # Predict
    prediction = load_model.predict(vector_form1)
    
    return prediction[0]

# Test
result = fake_news("Your news article here")

if result == 0:
    print("Reliable News")
else:
    print("Unreliable News")
```

---

## STREAMLIT WEB APPLICATION

### Complete Code (app.py)

```python
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize
port_stem = PorterStemmer()

# Load models
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

# Preprocessing function
def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if word not in stopwords.words('english')]
    con = ' '.join(con)
    return con

# Prediction function
def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction[0]

# Main app
def main():
    st.title("Fake News Classification App")
    st.subheader("Input the news content below")
    
    # Text input
    sentence = st.text_area("Enter your news content here", height=200)
    
    # Predict button
    if st.button("Predict"):
        prediction = fake_news(sentence)
        
        if prediction == 0:
            st.success("✅ Reliable News")
        else:
            st.error("🚨 Unreliable News")

if __name__ == '__main__':
    main()
```

---

### Run Application

**Command:**
```bash
streamlit run app.py
```

**NOT this:**
```bash
python app.py  # ❌ Wrong!
```

**Access:** http://localhost:8501

---

## KEY CONCEPTS

### Text Preprocessing Steps

```
Raw Text
    ↓
Remove special characters ([^a-zA-Z])
    ↓
Convert to lowercase
    ↓
Split into words
    ↓
Remove stopwords (the, is, a, an, etc.)
    ↓
Stemming (running → run, beautiful → beauti)
    ↓
Join back to string
    ↓
Clean Text
```

**Example:**
```
Input:  "House Dem Aide: We Didn't Even See Comey's Letter"
Output: "hous dem aid didn even see comey letter"
```

### Stemming Explained

**What it does:** Reduces words to root form

**Examples:**
- running → run
- runs → run
- beautiful → beauti
- beauty → beauti

**Why use it?**
- Combines similar words
- Reduces vocabulary size
- Saves memory
- Improves generalization

### TF-IDF vs Count Vectorizer

**Count Vectorizer:**
```
"the cat sat on the mat"
Vector: [the:2, cat:1, sat:1, on:1, mat:1]
```
- Just counts words
- Common words dominate

**TF-IDF:**
```
"the cat sat on the mat"
Vector: [the:0.1, cat:0.8, sat:0.6, on:0.1, mat:0.7]
```
- Weights by importance
- Rare words valued higher
- Better for classification

### Model Performance

**Accuracy: 88%**

**What this means:**
- 88 out of 100 predictions correct
- Better than random (50%)
- Good for text classification
- Room for improvement

**Confusion Matrix (typical):**
```
                Predicted
              Reliable  Fake
Actual  
Reliable      1800      200
Fake          300       1860
```

---

## IMPROVEMENTS

### Try Different Algorithms

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Compare models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"{name}: {accuracy:.2%}")
```

### Deep Learning Approach

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(vocab_size, 128),
    LSTM(64, dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)
```

### Add Confidence Scores

```python
def fake_news_with_confidence(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    
    # Get probability
    probability = load_model.predict_proba(vector_form1)
    prediction = load_model.predict(vector_form1)[0]
    confidence = max(probability[0])
    
    return prediction, confidence

# In Streamlit
if st.button("Predict"):
    pred, conf = fake_news_with_confidence(sentence)
    
    if pred == 0:
        st.success(f"✅ Reliable (Confidence: {conf:.1%})")
    else:
        st.error(f"🚨 Unreliable (Confidence: {conf:.1%})")
```

---

## COMMON ISSUES

### Issue 1: NLTK Data Missing

**Error:** `LookupError: Resource stopwords not found`

**Solution:**
```python
import nltk
nltk.download('stopwords')
```

### Issue 2: Model File Not Found

**Error:** `FileNotFoundError: model.pkl`

**Solution:**
- Ensure model.pkl and vector.pkl are in same directory as app.py
- Check spelling exactly matches

### Issue 3: Wrong Command

**Error using:** `python app.py`

**Correct:** `streamlit run app.py`

---

## BEST PRACTICES

1. **Always save both vectorizer and model**
2. **Use same preprocessing for training and prediction**
3. **Test with multiple algorithms**
4. **Add error handling in production**
5. **Validate with cross-validation**

---

## REAL-WORLD APPLICATIONS

### Use Cases

**Social Media:**
- Flag suspicious posts
- Warn before sharing
- Reduce misinformation

**News Aggregators:**
- Quality filtering
- Source verification
- Trust indicators

**Fact-Checkers:**
- Initial screening
- Prioritize reviews
- Scale operations

**Education:**
- Media literacy tool
- Critical thinking
- Source evaluation

---

## CONCLUSION

### Accomplished

✅ Loaded 20,800 news articles
✅ Preprocessed text (stemming, stopword removal)
✅ TF-IDF vectorization
✅ Trained Decision Tree (88% accuracy)
✅ Deployed Streamlit web app

### Key Learnings

1. Text preprocessing is critical
2. TF-IDF better than simple counts
3. Must save both vectorizer AND model
4. Streamlit makes deployment simple
5. 88% accuracy is good baseline

### Next Steps

**Beginners:**
- Run the code
- Test different articles
- Modify UI
- Try different parameters

**Advanced:**
- Implement LSTM/BERT
- Add confidence scores
- Multi-class classification
- Deploy to cloud
- Create API

---

*This project demonstrates practical NLP for combating misinformation - a critical tool for modern media literacy.*

**Note:** This tutorial uses Python/NLP/Machine Learning only - no Excel formulas are involved.