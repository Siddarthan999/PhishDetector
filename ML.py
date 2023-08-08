import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

phishing_urls = [
    "https://phishing-site1.com/login",
    "http://phishing-site2.ngrok.io",
    "https://phishing-site3.com/credentials",
    "https://379f-106-208-41-125.ngrok.io",
    "https://gmc-genetic-manage-currency.trycloudflare.com",
    "https://j5ibasixxx.loclx.io",
    "https://5006cc23690fb2.lhr.life",
    "https://phishing-site7.com",
    "https://hacker-targeted-site.com/login",
    "http://example-phishing-url.net/",
    "https://secure-login-phishing.com/credentials",
    "https://unsecured-target-url.xyz",    
    "https://secure-login-page-try-cloudflare.com",
]

legitimate_urls = [
    "https://example.com/login",
    "http://example2.com/",
    "https://example3.com/home",
    "https://www.example4.edu/",
    "https://en.wikipedia.org/wiki/.org",
    "https://www.coca-cola.com/in/en",
    "https://facebook.com",
    "https://www.example5.in/",
    "https://youtube.com",
    "https://instagram.com", 
    "https://example6.org",  
    "https://example7.co.in/"
]

all_urls = phishing_urls + legitimate_urls

labels = [1] * len(phishing_urls) + [0] * len(legitimate_urls)

# Use TfidfVectorizer to extract features from URLs
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_urls)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Model evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) 
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save the model to a pickle file
pickle.dump(clf, open('model.pkl', 'wb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    prediction = is_phishing_url(url)
    if prediction:
        result = f"{url} is a Potential Phishing Link."
    else:
        result = f"{url} is a Legitimate Link."
    return render_template('result.html', result=result)

def is_phishing_url(url):
    try:
        response = requests.get(url)
        html_content = response.text
    except:
        return True
    features = vectorizer.transform([html_content])
    prediction = clf.predict(features)
    return prediction[0] == 1

if __name__ == '__main__':
    app.run(debug=True)
