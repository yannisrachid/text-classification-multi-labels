import streamlit as st
import joblib
from nltk.corpus import stopwords
import pickle

french_stopwords = set(stopwords.words('french'))
tfidf = pickle.load(open("../output/tfidf.pkl", 'rb'))
id_to_category = {0: 'vn', 1: 'location', 2: 'vo', 3: 'apv'}

st.write('# Classification de conversation')
message_text = st.text_area("Entrez une conversation à prédire")

model = joblib.load('../output/model_svc.joblib')

def classify_message(model, message: str):
    """
    input: the model and the message
    output: dict (label predicted and probability)
    predicts the class of a message with the trained model
    """
    message = tfidf.transform([message]).toarray()
    prob = model._predict_proba_lr(message)[0].tolist()
    print(prob)
    p = max(prob)
    max_index = prob.index(p)
    label = id_to_category[max_index]
    print(label)
    return {'label': label, 'probabilité': p}


if message_text != '':
    result = classify_message(model, message_text)
    st.write(result)
