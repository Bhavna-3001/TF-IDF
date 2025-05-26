# TF-IDF
TF (Term Frequency) and IDF (Inverse Document Frequency) are fundamental concepts in information retrieval and text mining. Together, they are used to evaluate how important a word is to a document in a collection or corpus, forming the TF-IDF metric. This is commonly used in search engines, document similarity.
# TF (Term frequency ) -> 
this is  measure how often a word appear in a specific documents.

# IDF(Inverse Document  Frequency) ->
this measures how rare a word is across the entire corpus of documents.
A word that appears in many documents will have a lower IDF score while a word that appears in few documents will have a higher IDF score.


# Formula:

# TF( Term Frequency):
Number of time terms appears in a documents/ total numbers of term in  document.

# IDF(Inverse Document Frequency): 

log(Total number of document)/ (numbers of documents containing terms)

# TF-IDF (Term, document)
TF(term,document)* IDF(terms)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Your mini document database
documents = [
    "The food was delicious and the service was excellent",
    "The service was poor and the food was cold",
    "Excellent food and wonderful service",
    "The food was not good and the waiter was rude",
    "Fantastic service and very tasty food"
]

# Step 2: Create the TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Step 3: User query
query = input("Enter your search query: ")
query_vec = vectorizer.transform([query])

# Step 4: Calculate cosine similarity
similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

# Step 5: Rank results
top_indices = np.argsort(similarity_scores)[::-1][:3]  # top 3 results

print("\nTop search results:")
for idx in top_indices:
    print(f"Score: {similarity_scores[idx]:.4f} | Document: {documents[idx]}")
