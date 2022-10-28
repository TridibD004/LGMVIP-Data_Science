from flask import Flask ,render_template,request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('iris.csv')
pipe=pickle.load(open('iris_flower.pkl','rb'))

@app.route('/')
def index():
    return  render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    s_len = float(request.form.get('sl'))
    s_wid = float(request.form.get('sw'))
    p_len = float(request.form.get('pl'))
    p_wid = float(request.form.get('pw'))
    print (s_len,s_wid,p_len,p_wid )
    input=pd.DataFrame([[s_len,s_wid,p_len,p_wid]],
          columns=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width'])
    ans = pipe.predict(input)[0]
    if ans==0:
        return "Setosa"
    elif ans==1:
        return "Versicolor"
    elif ans==2:
        return "Virginica"



if __name__=="__main__":
    app.run(port=5010)

