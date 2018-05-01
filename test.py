from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

sentences = ["Hôm nay thời tiết như thế nào ?","Ngày mai thời tiết như thế nào ?"]

vectorizer.fit(sentences)
bag_of_words = vectorizer.transform(sentences)
print(vectorizer.vocabulary_.get("thời"))



