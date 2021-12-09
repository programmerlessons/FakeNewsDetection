from flask import Flask, escape, request, render_template
import pickle
import requests
import json
#from sklearn.feature_extraction.text import TfidfVectorizer


#vector = TfidfVectorizer(stop_words = 'english',max_df = 0.7)
vector = pickle.load(open("vectorizer.pkl",'rb'))
model = pickle.load(open("finalized_model.pkl",'rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction',methods = ['GET','POST'])
def prediction():
    print(request.method)
    if request.method == "POST":
        news = str(request.form['news'])
        print(news)
        

        predict = model.predict(vector.transform([news]))[0]
        print(predict)

        return render_template("prediction.html",prediction_text="News headlines is -> {}".format(predict))
    else:
        return render_template("prediction.html")

@app.route('/predictioncb',methods = ['GET','POST'])
def predictioncb():
    print(request.method)
    if request.method == "POST":
        news = str(request.form['news'])
        print(news)
        

        api_key = "48ae6e66171240029c2682528d36690d"
        input_claim = news

        # Define the endpoint (url), payload (sentence to be scored), api-key (api-key is sent as an extra header)
        api_endpoint = "https://idir.uta.edu/claimbuster/api/v2/score/text/"
        request_headers = {"x-api-key": api_key}
        payload = {"input_text": input_claim}

        # Send the POST request to the API and store the api response
        api_response = requests.post(url=api_endpoint, json=payload, headers=request_headers)
        result = api_response.json()
        y = result ['results']
        x=y[0]

        final=round(x['score'],2)
        print(final)
        if (final > 0.30):
        
            predict = "real"
        else:
            predict = "fake"
        return render_template("predictioncb.html",prediction_text="News headlines is -> {}".format(predict))
    else:
        return render_template("predictioncb.html")

if __name__ == '__main__':
    app.run()

