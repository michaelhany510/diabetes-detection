from flask import Flask,render_template, request,url_for
import pickle
import numpy as np 
import os


app = Flask(__name__,template_folder='templates')
picFolder = os.path.join('static','assets')
app.config['UPLOAD_FOLDER'] = picFolder

def loadModel(x):
    model = pickle.load(open('finalised_model.pkl','rb'))
    out = model.predict(x)
    return out[0]

@app.route('/',methods = ["GET"])
def index():
    print("hooooooooooome")
    return render_template('index.html',empty = 1,approved = 1)

@app.route('/predict',methods=['GET','POST'])
def predict():
    print('entered predict')
    if request.method == 'POST':
        print("pooooooooooooost")
        Cholesterol    = float( request.form['Cholesterol'])
        Glucose        = float(request.form['Glucose'])
        hdl_chol       = float(request.form['hdl_chol'])
        Chol_hdl_ratio = float(request.form['Chol_hdl_ratio'])
        Age            = float(request.form['Age'])
        Height         = float(request.form['Height'])
        Weight         = float(request.form['Weight'])
        bmi            = float(request.form['bmi'])
        Systolic_bp    = float(request.form['Systolic_bp'])
        Diastolic_bp   = float(request.form['Diastolic_bp'])
        Waist          = float(request.form['Waist'])
        Hip            = float(request.form['Hip'])
        Waist_Hip_ratio= float(request.form['Waist_Hip_ratio'])
        gender = float(request.form['Gender'])
        
        
        x = [Cholesterol,Glucose,hdl_chol,Chol_hdl_ratio,Age,gender,
             Height,Weight,bmi,Systolic_bp,Diastolic_bp,Waist
             ,Hip,Waist_Hip_ratio]
        x = np.array(x).reshape(1,-1)
        out = loadModel(x)
        print (out)
        return render_template('index.html',empty = 0,approved = out)
    print ("geeeeeeeeeeeeeeeeeet")
    return 'predict'
    

if __name__ == '__main__':
    app.run(debug  = True)