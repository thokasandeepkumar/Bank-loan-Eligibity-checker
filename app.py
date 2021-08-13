from flask import Flask, render_template,url_for, request,redirect ,make_response
import pandas as pd
import warnings
import joblib
from flask_cors import cross_origin
from predictionfolder.prediction import LA_predict,Fraud_predict,LR_predict,LE_predict,MA_predict
from pandas_profiling import ProfileReport
import matplotlib
matplotlib.use('Agg')

def warns(*args, **kwargs):
    pass
warnings.warn = warns

ALLOWED_EXTENSIONS = set(['csv','xlsx','data'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# load the model from directory
ss_LA = joblib.load('pickle_files/LA_Std_scaler.pkl')
model_LA=joblib.load('pickle_files/DTModel-1.pkl')
model_Fraud = joblib.load('pickle_files/Fraud_new_model.pkl')
model_LR = joblib.load('pickle_files/loan_risk.pkl')
model_LE = joblib.load('pickle_files/LE-DecTreeModel.pkl')
model_MA = joblib.load('pickle_files/Mortgage_RE.pkl')

LA_instance = LA_predict()
Fraud_instance = Fraud_predict()
LR_instance = LR_predict()
LE_instance = LE_predict()
MA_instance = MA_predict()

app = Flask(__name__)

@app.route('/')
@cross_origin()
def intro():
    return render_template('intro.html')

@app.route('/home',methods=['GET','POST'])
@cross_origin()
def home():
    return render_template('home.html')

@app.route("/Down_Bulk_File",methods=['GET'])
@cross_origin()
def download_file():
    table = bulk_predict()
    response = make_response(table)
    response.headers['Content-Disposition'] = 'attachment; filename=report.csv'
    response.headers['Content-type'] = "text/csv"
    return response

@app.route('/LA',methods=['GET','POST'])
@cross_origin()
def LA():
    if request.method == 'POST':
        return render_template('LA_about_page.html')   #LA_home.html

@app.route('/LA_main_dash',methods=['GET','POST'])
@cross_origin()
def LA_main_dash():
    if request.method == 'POST':
        return render_template('LA_main_dash.html')

@app.route('/FD',methods=['GET','POST'])
@cross_origin()
def FD():
    if request.method == 'POST':
        return render_template('FD_about_page.html')  # FD_new_home.html

@app.route('/FD_main_dash',methods=['GET','POST'])
@cross_origin()
def FD_main_dash():
    if request.method == 'POST':
        return render_template('FD_main_dash.html')

@app.route('/LR',methods=['GET','POST'])
@cross_origin()
def LR():
    if request.method == 'POST':
        return render_template('LR_about_page.html')    # loan_risk.html

@app.route('/LR_main_dash',methods=['GET','POST'])
@cross_origin()
def LR_main_dash():
    if request.method == 'POST':
        return render_template('lrisk_main_dash.html')

@app.route('/LE',methods=['GET','POST'])
@cross_origin()
def LE():
    if request.method == 'POST':
        return render_template('LE_about_page.html')

@app.route('/LE_main_dash',methods=['GET','POST'])
@cross_origin()
def LE_main_dash():
    if request.method == 'POST':
        return render_template('LE_main_dash.html')  # laon_eligibilty.html

@app.route('/MA',methods=['GET','POST'])
@cross_origin()
def MA():
    if request.method == 'POST':
        return render_template('MA_about_page.html')

@app.route('/MA_main_dash',methods=['GET','POST'])
@cross_origin()
def MA_main_dash():
    if request.method == 'POST':
        return render_template('MA_main_dash.html')

@app.route('/LA_single_predict',methods=['GET','POST'])
@cross_origin()
def LA_single_predict():
    if request.method == 'POST':
        return render_template('LA_single_predict.html')

@app.route('/LA_multi_predict',methods=['GET','POST'])
@cross_origin()
def LA_multi_predict():
    if request.method == 'POST':
        return render_template('LA_multi_predict.html')

@app.route('/LA_retrain',methods=['GET','POST'])
@cross_origin()
def LA_retrain():
    if request.method == 'POST':
        return render_template('LA_retrain.html')

@app.route('/FD_single_predict',methods=['GET','POST'])
@cross_origin()
def FD_single_predict():
    if request.method == 'POST':
        return render_template('FD_single_predict.html')

@app.route('/FD_multi_predict',methods=['GET','POST'])
@cross_origin()
def FD_multi_predict():
    if request.method == 'POST':
        return render_template('FD_multi_predict.html')

@app.route('/FD_retrain',methods=['GET','POST'])
@cross_origin()
def FD_retrain():
    if request.method == 'POST':
        return render_template('FD_retrain.html')

@app.route('/lrisk_single_predict',methods=['GET','POST'])
@cross_origin()
def lrisk_single_predict():
    if request.method == 'POST':
        return render_template('lrisk_single_predict.html')

@app.route('/lrisk_multi_predict',methods=['GET','POST'])
@cross_origin()
def lrisk_multi_predict():
    if request.method == 'POST':
        return render_template('lrisk_multi_predict.html')

@app.route('/lrisk_retrain',methods=['GET','POST'])
@cross_origin()
def lrisk_retrain():
    if request.method == 'POST':
        return render_template('lrisk_retrain.html')

@app.route('/LE_single_predict',methods=['GET','POST'])
@cross_origin()
def LE_single_predict():
    if request.method == 'POST':
        return render_template('LE_single_predict.html')

@app.route('/LE_multi_predict',methods=['GET','POST'])
@cross_origin()
def LE_multi_predict():
    if request.method == 'POST':
        return render_template('LE_multi_predict.html')

@app.route('/LE_retrain',methods=['GET','POST'])
@cross_origin()
def LE_retrain():
    if request.method == 'POST':
        return render_template('LE_retrain.html')

@app.route('/MA_single_predict',methods=['GET','POST'])
@cross_origin()
def MA_single_predict():
    if request.method == 'POST':
        return render_template('MA_single_predict.html')

@app.route('/MA_multi_predict',methods=['GET','POST'])
@cross_origin()
def MA_multi_predict():
    if request.method == 'POST':
        return render_template('MA_multi_predict.html')

@app.route('/MA_retrain',methods=['GET','POST'])
@cross_origin()
def MA_retrain():
    if request.method == 'POST':
        return render_template('MA_retrain.html')

@app.route('/bulk_predict',methods=['GET','POST'])
@cross_origin()
def bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            global data
            data = Fraud_instance.predictor(file)
            new_data = data[:8]
            return render_template('result_bulk.html', tables=[new_data.to_html(classes='data')], titles=data.columns.values)
        else:
            return redirect(request.url)
    else:
        return data.to_string()

@app.route('/LA_bulk_predict',methods=['GET','POST'])
@cross_origin()
def LA_bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            global data
            data = LA_instance.predictor(file)
            new_data = data[:8]   # to restrict number of rows in bulk output ..
            return render_template('result_bulk.html', tables=[new_data.to_html(classes='data')], titles=data.columns.values)
        else:
            return redirect(request.url)
    else:
        return data.to_string()

@app.route('/LR_bulk_predict',methods=['GET','POST'])
@cross_origin()
def LR_bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            global data
            data = LR_instance.predictor(file)
            new_data = data[:8]
            return render_template('result_bulk.html', tables=[new_data.to_html(classes='data')], titles=data.columns.values)
        else:
            return redirect(request.url)
    else:
        return data.to_string()

@app.route('/LE_bulk_predict',methods=['GET','POST'])
@cross_origin()
def LE_bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            global data
            data = LE_instance.predictor(file)
            new_data = data[:8]
            return render_template('result_bulk.html', tables=[new_data.to_html(classes='data')], titles=data.columns.values)
        else:
            return redirect(request.url)
    else:
        return data.to_string()

@app.route('/MA_bulk_predict',methods=['GET','POST'])
@cross_origin()
def MA_bulk_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            global data
            data = MA_instance.predictor(file)
            new_data = data[:8]
            return render_template('result_bulk.html', tables=[new_data.to_html(classes='data')], titles=data.columns.values)
        else:
            return redirect(request.url)
    else:
        return data.to_string()

@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict():
    if request.method == 'POST':

        Type = request.form.get("gender", False)
        if (Type == 'Male'):
            Type = 0
        elif (Type == 'Female'):
            Type = 1
        elif (Type == 'Enterprise'):  # random
            Type = 2
        elif (Type == 'Unknown'):
            Type = 3

        amount = float(request.form.get("amount", False))
        merchant = float(request.form.get("merchant", False))
        category = float(request.form.get("category", False))
        step = float(request.form.get("step", False))
        age = float(request.form.get("age", False))

        df = pd.DataFrame(
            {"step": step ,"age": age,'gender': Type,  "merchant": merchant,
                    "category": category,"amount": amount  }, index=[0])

        my_prediction = model_Fraud.predict(df)

        # check my_pred ..............................................
        return render_template('fraud_detect_result.html', prediction=my_prediction)

@app.route('/predict_LA',methods=['GET','POST'])
@cross_origin()
def predict_LA():
    if request.method == 'POST':
        Age = float(request.form.get("Age", False))
        Experience = float(request.form.get("Experience",False))
        Income = float(request.form.get("Income",False))
        Family = float(request.form.get("Family",False))
        CCAvg = float(request.form.get("CCAvg",False))

        Education = request.form.get("Education", False)
        if (Education == 'Undergrad'):
            Education = 0
        elif (Education == "Graduate"):
            Education = 1
        elif (Education == "Professional"):
            Education= 2

        Mortgage = float(request.form.get("Mortgage", False))

        SecuritiesAccount = request.form.get("SecuritiesAccount", False)
        if (SecuritiesAccount == 'Yes'):
            SecuritiesAccount = 1
        elif (SecuritiesAccount == "No"):
            SecuritiesAccount = 0

        CDAccount = request.form.get("CDAccount", False)
        if (CDAccount == 'Yes'):
            CDAccount = 1
        elif (CDAccount == "No"):
            CDAccount = 0

        Online = request.form.get("Online", False)
        if (Online == 'Yes'):
            Online = 1
        elif (Online == "No"):
            Online = 0

        CreditCard = request.form.get("CreditCard", False)
        if (CreditCard == 'Yes'):
            CreditCard = 1
        elif (CreditCard == "No"):
            CreditCard = 0

        LA_prediction = model_LA.predict(ss_LA.transform([[Age,Experience,Income,Family,CCAvg
                                                         ,Education,Mortgage,SecuritiesAccount,CDAccount,Online,CreditCard]]))

        return render_template('LA_result.html',prediction=LA_prediction)
    else:
        return render_template('LA_home.html')

@app.route('/predict_LE',methods=['GET','POST'])
@cross_origin()
def predict_LE():
    if request.method == 'POST':

        CurrentLoanAmount = float(request.form.get("CurrentLoanAmount",False))
        CreditScore = float(request.form.get("CreditScore",False))
        AnnualIncome = float(request.form.get("AnnualIncome",False))
        Yearsincurrentjob = float(request.form.get("Yearsincurrentjob",False))
        MonthlyDebt = float(request.form.get("MonthlyDebt",False))
        YearsofCreditHistory = float(request.form.get("YearsofCreditHistory",False))
        Monthssincelastdelinquent = float(request.form.get("Monthssincelastdelinquent",False))
        NumberofOpenAccounts = float(request.form.get("NumberofOpenAccounts",False))
        NumberofCreditProblems = float(request.form.get("NumberofCreditProblems",False))
        CurrentCreditBalance = float(request.form.get("CurrentCreditBalance",False))
        MaximumOpenCredit = float(request.form.get("MaximumOpenCredit",False))
        Term_LongTerm = float(request.form.get("Term_LongTerm",False))

        df = pd.DataFrame({
            "Monthssincelastdelinquent": Monthssincelastdelinquent, "MonthlyDebt": MonthlyDebt,
            "AnnualIncome": AnnualIncome, "CurrentCreditBalance": CurrentCreditBalance,
            "MaximumOpenCredit": MaximumOpenCredit, "CreditScore": CreditScore, "Yearsincurrentjob": Yearsincurrentjob,
            "Term_LongTerm": Term_LongTerm, "YearsofCreditHistory": YearsofCreditHistory,
            "NumberofOpenAccounts": NumberofOpenAccounts,
            "NumberofCreditProblems": NumberofCreditProblems, "CurrentLoanAmount": CurrentLoanAmount}, index=[0])

        LE_prediction = model_LE.predict(df)
        return render_template('LE_result.html', prediction=LE_prediction)
    else:
        return render_template('LE_home.html') # ==============================================

@app.route('/predict_MA',methods=['GET','POST'])
@cross_origin()
def predict_MA():
    if request.method == 'POST':
        PostCode = float(request.form.get("PostCode", False))
        Qtr = float(request.form.get("Qtr", False))
        Unit = float(request.form.get("Unit", False))

        df = pd.DataFrame({ "PostCode":PostCode, "Qtr":Qtr, "Unit":Unit},index=[0])
        MA_prediction = model_MA.predict(df)
        return render_template('MA_result.html',prediction=MA_prediction)
    else:
        return render_template('MA_main_dash.html')


@app.route('/predict_LR',methods=['GET','POST'])
@cross_origin()
def predict_LR():
    if request.method == 'POST':
        Loan_Amount = float(request.form.get("LoanAmount",False))
        Term = float(request.form.get("Loan_Amount_Term",False))
        Interest_Rate = float(request.form.get("Interest_Rate",False))
        Employment_Years = float(request.form.get("Employment_Years",False))
        Annual_Income = float(request.form.get("Annual_Income",False))
        Debt_to_Income = float(request.form.get("Debt_to_Income",False))
        Delinquent_2yr = float(request.form.get("Delinquent_2yr",False))
        Revolving_Cr_Util = float(request.form.get("Revolving_Cr_Util",False))
        Total_Accounts = float(request.form.get("Total_Accounts",False))
        Longest_Credit_Length = float(request.form.get("Longest_Credit_Length",False))
        
        Home_Ownership = request.form.get("Home_Ownership",False)
        if Home_Ownership == 'RENT':
            Home_Ownership = 5
        elif Home_Ownership == 'OWN':
            Home_Ownership = 4
        elif Home_Ownership == 'MORTGAGE':
            Home_Ownership = 1
        elif Home_Ownership == 'OTHER':
            Home_Ownership = 3
        elif Home_Ownership == 'NONE':
            Home_Ownership = 2
        elif Home_Ownership == 'ANY':
            Home_Ownership = 0

        Verification_Status = request.form.get("Verification_Status",False)
        if Verification_Status == 'VERIFIED - income':
            Verification_Status = 1
        elif Verification_Status == 'VER' \
                                    'IFIED - income source':
            Verification_Status = 2
        elif Verification_Status == 'not verified':
            Verification_Status = 0
              
        Loan_Purpose = request.form.get("Loan_Purpose",False)
        if Loan_Purpose == 'credit_card':
            Loan_Purpose = 1
        elif Loan_Purpose =='car':
            Loan_Purpose = 0
        elif Loan_Purpose == 'small_business':
            Loan_Purpose = 11
        elif Loan_Purpose == 'other':
            Loan_Purpose = 9
        elif Loan_Purpose == 'wedding':
            Loan_Purpose = 13
        elif Loan_Purpose == 'debt_consolidation':
            Loan_Purpose = 2
        elif Loan_Purpose == 'home_improvement':
            Loan_Purpose = 4
        elif Loan_Purpose == 'major_purchase':
            Loan_Purpose = 6
        elif Loan_Purpose == 'medical':
            Loan_Purpose = 7
        elif Loan_Purpose == 'moving':
            Loan_Purpose = 8
        elif Loan_Purpose == 'renewable_energy':
            Loan_Purpose = 10
        elif Loan_Purpose == 'vacation':
            Loan_Purpose = 12
        elif Loan_Purpose == 'house':
            Loan_Purpose = 5
        elif Loan_Purpose == 'educational':
            Loan_Purpose = 3

        State = request.form.get("State",False)
        if State == 'AK':
            State = 0
        elif State == 'AL':
            State = 1
        elif State == 'AR':
            State = 2
        elif State == 'AZ':
            State = 3
        elif State == 'CA':
            State = 4
        elif State == 'CO':
            State = 5
        elif State == 'CT':
            State = 6
        elif State == 'DC':
            State = 7
        elif State == 'DE':
            State = 8
        elif State == 'FL':
            State = 9
        elif State == 'GA':
            State = 10
        elif State == 'HI':
            State = 11
        elif State == 'IA':
            State = 12
        elif State == 'ID':
            State = 13
        elif State == 'IL':
            State = 14
        elif State == 'IN':
            State = 15
        elif State == 'KS':
            State = 16
        elif State == 'KY':
            State = 17
        elif State == 'LA':
            State = 18
        elif State == 'MA':
            State = 19
        elif State == 'MD':
            State = 20
        elif State == 'ME':
            State = 21
        elif State == 'MI':
            State = 22
        elif State == 'MN':
            State = 23
        elif State == 'MO':
            State = 24
        elif State == 'MS':
            State = 25
        elif State == 'MT':
            State = 26
        elif State == 'NC':
            State = 27
        elif State == 'NE':
            State = 28
        elif State == 'NH':
            State = 29
        elif State == 'NJ':
            State = 30
        elif State == 'NM':
            State = 31
        elif State == 'NV':
            State = 32
        elif State == 'NY':
            State = 33
        elif State == 'OH':
            State = 34
        elif State == 'OK':
            State = 3
        elif State == 'OR':
            State = 36
        elif State == 'PA':
            State = 37
        elif State == 'RI':
            State = 38
        elif State == 'SC':
            State = 39
        elif State == 'SD':
            State = 40
        elif State == 'TN':
            State = 41
        elif State == 'TX':
            State = 42
        elif State == 'UT':
            State = 43
        elif State == 'VA':
            State = 44
        elif State == 'VT':
            State = 45
        elif State == 'WA':
            State = 46
        elif State == 'WI':
            State = 47
        elif State == 'WV':
            State = 48
        elif State == 'WY':
            State = 49

        LR_prediction = model_LR.predict([[Loan_Amount, Term, Interest_Rate, Employment_Years,
        Home_Ownership, Annual_Income, Verification_Status,
        Loan_Purpose, State, Debt_to_Income, Delinquent_2yr,
        Revolving_Cr_Util, Total_Accounts, Longest_Credit_Length]])

        return render_template('loan_risk_result.html', prediction=LR_prediction)
    else:
        return render_template('loan_risk.html')

@app.route('/FD_data_graph',methods = ['GET','POST'])
@cross_origin()
def FD_graph():
    if request.method == 'POST':
        return render_template('FD_data_graph.html')

@app.route('/LA_data_graph',methods = ['GET','POST'])
@cross_origin()
def LA_graph():
    if request.method == 'POST':
        return render_template('LA_data_graph.html')

@app.route('/Lrisk_data_graph',methods = ['GET','POST'])
@cross_origin()
def Lrisk_graph():
    if request.method == 'POST':
        return render_template('Lrisk_data_graph.html')

@app.route('/LE_data_graph',methods = ['GET','POST'])
@cross_origin()
def LE_graph():
    if request.method == 'POST':
        return render_template('LE_data_graph.html')

@app.route('/MA_data_graph',methods = ['GET','POST'])
@cross_origin()
def MA_graph():
    if request.method == 'POST':
        return render_template('MA_data_graph.html')

@app.route('/retrain',methods = ['GET','POST'])
@cross_origin()
def retrain():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            retrain.retrainer(file)
        else:
            return redirect(request.url)

@app.route('/home1',methods = ['GET','POST'])
@cross_origin()
def home1():
    if request.method=='POST':
        return redirect(url_for('home'))

@app.route('/show_graph',methods=['GET','POST'])
@cross_origin()
def show_graph():
    try:
        if request.method=='POST':
            graph_data =pd.read_csv(r'graph_input_files\graph_data.csv')
            prof = ProfileReport(graph_data)
            prof.to_file(output_file=r'templates\bulk_graph_output.html')
            return render_template('bulk_graph_output.html')
    except Exception as e:
        raise e
'''
@app.route('/drug_retrain',methods=['GET','POST'])
@cross_origin()
def retrain():
    try:
        if request.method == "POST":
            file = request.files['retrain_file']
            if file:
                file.save(secure_filename(file.filename))
                a=trainModel()
                a.trainingModel(file.filename,file_object)
                os.remove(file.filename)
                file_object.close()
                return render_template('home.html',text=".... Model Retrained Successfully ....")
    except Exception as e:
        file_object.close()
        return 'Something went wrong , check your file extension .(should be .csv )'
'''

if __name__ == '__main__':
    # To run on web ..
    #app.run(host='0.0.0.0',port=8080)
    # To run locally ..
    app.run(debug=True)
