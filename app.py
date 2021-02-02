from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open("model.pkl","rb"))
app = Flask(__name__)

@app.route('/')
def main():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def home():
    user_age = request.form['age']
    arr = np.array([[user_age]],dtype="float64")
    pre = model.predict(arr)
    output = round(pre[0], 2)
    return render_template("home.html", answer = "YOUR PREDICTED SALARY IS : Rs.{}".format(output))

if __name__ == "__main__":
    app.run(debug = True)