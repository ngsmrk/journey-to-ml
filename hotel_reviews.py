import pickle
import textutils
import pandas as pd

hotel_data = pd.read_csv('./data/hotel-reviews.csv', index_col=0)
# print(hotel_data.columns)

test_data = hotel_data.head(5)
input = test_data['text']

model = pickle.load(open('hotel_reviews.pkl', 'rb'))
vectorizer = pickle.load(open('hotel_reviews_vectorizer.pkl', 'rb'))

print(model.classes_)

print('Input')
print(input)
print('Expected')
print(test_data['label'])
print('Prediction')
print(model.predict_proba(vectorizer.transform(input)))