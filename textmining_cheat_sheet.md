# **Text Mining Cheat Sheet**

*A Comprehensive Guide for Data Science Enthusiasts*  

---

## **1. Text Preprocessing**

### **1.1. Text Cleaning**

- **Lowercasing**: Convert all text to lowercase for uniformity.  
  
  ```python
  text = text.lower()
  ```

- **Remove Punctuation**:  
  
  ```python
  import re
  text = re.sub(r'[^\w\s]', '', text)
  ```

- **Remove Numbers**:  
  
  ```python
  text = re.sub(r'\d+', '', text)
  ```

- **Remove Extra Whitespaces**:  
  
  ```python
  text = ' '.join(text.split())
  ```

### **1.2. Tokenization**

- **Word Tokenization**: Splitting text into words.  
  
  ```python
  from nltk.tokenize import word_tokenize
  tokens = word_tokenize(text)
  ```

- **Sentence Tokenization**: Splitting text into sentences.  
  
  ```python
  from nltk.tokenize import sent_tokenize
  sentences = sent_tokenize(text)
  ```

### **1.3. Stopword Removal**

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]
```

### **1.4. Stemming & Lemmatization**

- **Stemming (crude reduction to root form)**:  
  
  ```python
  from nltk.stem import PorterStemmer
  stemmer = PorterStemmer()
  stemmed_words = [stemmer.stem(word) for word in tokens]
  ```

- **Lemmatization (proper dictionary-based reduction)**:  
  
  ```python
  from nltk.stem import WordNetLemmatizer
  lemmatizer = WordNetLemmatizer()
  lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
  ```

---

## **2. Feature Extraction**

### **2.1. Bag of Words (BoW)**

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
```

### **2.2. TF-IDF (Term Frequency-Inverse Document Frequency)**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)
```

### **2.3. Word Embeddings**

- **Word2Vec (Google's Pretrained Model)**:  
  
  ```python
  from gensim.models import Word2Vec
  model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
  ```

- **GloVe (Global Vectors)**:  
  
  ```python
  from gensim.scripts.glove2word2vec import glove2word2vec
  from gensim.models import KeyedVectors
  glove_model = KeyedVectors.load_word2vec_format('glove.6B.100d.txt')
  ```

- **FastText (by Facebook)**:  
  
  ```python
  from gensim.models import FastText
  model = FastText(sentences, vector_size=100, window=5, min_count=1)
  ```

---

## **3. Text Classification & NLP Models**

### **3.1. Traditional ML Models**

- **Naive Bayes**:  
  
  ```python
  from sklearn.naive_bayes import MultinomialNB
  model = MultinomialNB()
  model.fit(X_train, y_train)
  ```

- **SVM (Support Vector Machines)**:  
  
  ```python
  from sklearn.svm import SVC
  model = SVC(kernel='linear')
  model.fit(X_train, y_train)
  ```

### **3.2. Deep Learning Models**

- **RNN/LSTM**:  
  
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Embedding, LSTM, Dense
  model = Sequential([
      Embedding(input_dim=vocab_size, output_dim=100),
      LSTM(128),
      Dense(1, activation='sigmoid')
  ])
  ```

- **Transformer Models (BERT, GPT, etc.)**  
  
  ```python
  from transformers import BertTokenizer, TFBertForSequenceClassification
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
  ```

---

## **4. Topic Modeling**

### **4.1. Latent Dirichlet Allocation (LDA)**

```python
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=5)
lda.fit(X)  # X is a BoW/TF-IDF matrix
```

### **4.2. Non-Negative Matrix Factorization (NMF)**

```python
from sklearn.decomposition import NMF
nmf = NMF(n_components=5)
nmf.fit(X)
```

---

## **5. Text Similarity & Clustering**

### **5.1. Cosine Similarity**

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vec1, vec2)
```

### **5.2. K-Means Clustering**

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
```

---

## **6. Evaluation Metrics**

### **6.1. Classification Metrics**

- **Accuracy, Precision, Recall, F1-Score**:  
  
  ```python
  from sklearn.metrics import classification_report
  print(classification_report(y_true, y_pred))
  ```

- **Confusion Matrix**:  
  
  ```python
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(y_true, y_pred)
  ```

### **6.2. Topic Modeling Evaluation**

- **Coherence Score**:  
  
  ```python
  from gensim.models import CoherenceModel
  coherence = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary)
  coherence_score = coherence.get_coherence()
  ```

---

## **7. Useful Libraries**

| **Library**                     | **Purpose**                                 |
| ------------------------------- | ------------------------------------------- |
| **NLTK**                        | NLP tasks (tokenization, POS tagging, etc.) |
| **spaCy**                       | Industrial-strength NLP                     |
| **Gensim**                      | Topic modeling & word embeddings            |
| **Scikit-learn**                | ML models & feature extraction              |
| **Transformers (Hugging Face)** | BERT, GPT, etc.                             |
| **TensorFlow/PyTorch**          | Deep learning models                        |

---

### **Final Tips**

✅ **Always preprocess text before modeling.**  
✅ **Experiment with different feature extraction techniques.**  
✅ **Fine-tune transformer models for better performance.**  
✅ **Visualize topics & word clouds for better insights.**  

---

# **Text Mining: Key Concepts Explained**

Text mining (or text analytics) is the process of extracting meaningful information from unstructured text data. Below is a breakdown of the **core concepts** in text mining, along with examples and applications.

---

## **1. Text Preprocessing**

*Before analyzing text, we clean and structure it for better results.*  

### **1.1. Tokenization**

- **What?** Breaking text into smaller units (words, sentences).  
- **Example**:  
  - *Input*: "I love NLP!"  
  - *Output*: `["I", "love", "NLP", "!"]` (word tokens)  

### **1.2. Stopword Removal**

- **What?** Removing common words (e.g., "the", "is") that add little meaning.  
- **Example**:  
  - *Input*: "This is a sample sentence."  
  - *Output*: `["sample", "sentence"]`  

### **1.3. Stemming & Lemmatization**

- **Stemming**: Crudely chops words to root form (e.g., "running" → "run").  
- **Lemmatization**: Uses dictionary to convert words properly (e.g., "better" → "good").  

### **1.4. Normalization**

- Lowercasing, removing punctuation, correcting typos.  

---

## **2. Feature Extraction**

*Converting text into numerical form for machine learning.*  

### **2.1. Bag of Words (BoW)**

- **What?** Counts word frequencies in a document.  
- **Example**:  
  - *Text*: "Cats love dogs. Dogs love cats."  
  - *BoW*: `{"cats":2, "love":2, "dogs":2}`  

### **2.2. TF-IDF (Term Frequency-Inverse Document Frequency)**

- **What?** Weighs words by importance (rare words get higher scores).  

- **Formula**:  
  
  ```
  TF-IDF = (Term Frequency) × (Inverse Document Frequency)
  ```

### **2.3. Word Embeddings**

- **What?** Represents words as dense vectors (captures meaning).  
- **Examples**:  
  - **Word2Vec**: Learns word associations from large text.  
  - **GloVe**: Uses global statistics (co-occurrence matrix).  
  - **FastText**: Handles subword information (good for rare words).  

---

## **3. Text Classification**

*Categorizing text into predefined classes.*  

### **3.1. Common Algorithms**

| **Algorithm**                 | **Use Case**                            |
| ----------------------------- | --------------------------------------- |
| Naive Bayes                   | Spam detection                          |
| SVM (Support Vector Machines) | Sentiment analysis                      |
| LSTM/RNN                      | Sequential data (e.g., text generation) |
| BERT/Transformer Models       | Advanced NLP tasks                      |

### **3.2. Example: Sentiment Analysis**

- **Input**: "This movie was amazing!"  
- **Output**: `Positive`  

---

## **4. Topic Modeling**

*Discovering hidden themes in a collection of documents.*  

### **4.1. Latent Dirichlet Allocation (LDA)**

- **What?** Assigns topics to documents based on word distributions.  
- **Example**:  
  - *Document*: "Data science involves statistics and machine learning."  
  - *Topics*: `["Data Science", "Statistics", "ML"]`  

### **4.2. Non-Negative Matrix Factorization (NMF)**

- **What?** Decomposes a term-document matrix into topics.  

---

## **5. Text Similarity & Clustering**

### **5.1. Cosine Similarity**

- Measures angle between two vectors (1 = identical, 0 = unrelated).  
- **Use Case**: Plagiarism detection, recommendation systems.  

### **5.2. K-Means Clustering**

- Groups similar documents together.  
- **Example**:  
  - *Cluster 1*: Sports articles  
  - *Cluster 2*: Politics articles  

---

## **6. Evaluation Metrics**

### **6.1. Classification Metrics**

| **Metric** | **Purpose**                                   |
| ---------- | --------------------------------------------- |
| Accuracy   | % of correct predictions                      |
| Precision  | % of true positives among predicted positives |
| Recall     | % of actual positives correctly predicted     |
| F1-Score   | Harmonic mean of precision & recall           |

### **6.2. Topic Modeling Evaluation**

- **Coherence Score**: Measures how interpretable topics are.  

---

## **7. Applications of Text Mining**

✔ **Sentiment Analysis** (e.g., Twitter, reviews)  
✔ **Chatbots & Virtual Assistants** (NLP + ML)  
✔ **Document Summarization** (e.g., news articles)  
✔ **Fraud Detection** (e.g., phishing emails)  
✔ **Healthcare** (e.g., extracting info from medical reports)  

---

### **Key Takeaways**

🔹 **Preprocessing is crucial** (clean, tokenize, normalize).  
🔹 **Feature extraction** (BoW, TF-IDF, embeddings) converts text to numbers.  
🔹 **Models range from simple (Naive Bayes) to advanced (BERT).**  
🔹 **Evaluation ensures model reliability.**  
