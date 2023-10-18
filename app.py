from flask import Flask, render_template, request, jsonify
import preprocessing_data
import pickle

app = Flask(__name__)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('get_input.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        data = request.form.get('user_input')
        pre_data = preprocessing_data.preprocess_text(data)

        vector_input = tfidf.transform(pre_data).toarray()
        emotion = model.predict(vector_input)

        emotion_mapping = {
            0: "Anger",
            1: "Fear",
            2: "Joy",
            3: "Love",
            4: "Sadness",
            5: "Surprise"
        }

        predicted_emotion_index = emotion[0]
        predicted_emotion = emotion_mapping[predicted_emotion_index]

        response = {'emotion': predicted_emotion}
        return render_template('predict.html', prediction=response)


if __name__ == '__main__':
    app.run(debug=True)
