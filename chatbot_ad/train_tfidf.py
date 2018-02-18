
import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append('..')

def main():
    x_data, _, _ = pickle.load(
        open('chatbot.pkl', 'rb'))

    vectorizer = TfidfVectorizer(
        min_df=10,
        analyzer='char', ngram_range=(1, 2))

    vectorizer.fit([
        ''.join(x)
        for x in x_data
    ])

    t = vectorizer.transform([
        ''.join(x)
        for x in x_data[:1]
    ])

    print(len(vectorizer.vocabulary_))

    print(np.sum(t))

    pickle.dump(vectorizer, open('tfidf.pkl', 'wb'))




if __name__ == '__main__':
    main()
