# api/extract.py

import pickle

def predict(text):

    emoji_dict={"Joy":["😂","🤣","😃","😄","😊"],"Disgust":["😷","😖","🤢","😩","😪"],"Sad":["😭","😞","😢","🙁","😔"],"Anger":["😠","😤","👊","😡","😒"],"Scared":["😨","😓","😖","😣","😞"],"Surprise":["🤔","😲","😳","🙊","🙈"],"Love":["😘","💋","💘","❤️","💕"] }

    loaded_model = pickle.load(open('api/finalized_model.sav', 'rb'))

    text=[text]

    vectorizer = pickle.load(open('api/vectorizer_model.sav', 'rb'))

    text_vector = vectorizer.transform(text)

    predicted_rf = loaded_model.predict(text_vector)

    emo = predicted_rf[0]

    emoji=emoji_dict[emo]

    return emoji