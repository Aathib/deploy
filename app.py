from flask import Flask, flash, request, redirect, url_for, render_template
import pickle, os
import pandas as pd
from werkzeug.utils import secure_filename
import sys, time
sys.path.append(r'C:\\Documents\Full_data')

from preprocess import preprocs

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

UPLOAD_FOLDER = r'C:/Documents/Full_data/Upload'
ALLOWED_EXTENSIONS = set(['csv'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class FileNameLatest:   
     def __init__(self,name):
         self.name=name
     def setName(self,value):
         self.name=value
        
fileNameSample = FileNameLatest('')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
           
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fileNameSample.setName(filename)
            return redirect(url_for('upload_file',
                                    filename=filename))
          
    return render_template('index1.html')


@app.route('/predict',methods=['POST', 'GET'])
def predict(filename = None):
    '''
    For rendering results on HTML GUI
    '''
    start = time.process_time()
    data = pd.read_csv(os.path.join(UPLOAD_FOLDER, fileNameSample.name), engine = 'python')
    
    data = data[pd.notnull(data['Heal Code'])]
    data = data[pd.notnull(data['Pat DOB'])]
    data = data[data['Dollars'] >= 0]
    data.drop_duplicates(inplace = True)
    data.reset_index(drop= True, inplace = True)
        
    percent = request.form.get('Percentage to be reviewed')
    percent = int(percent)/100
    data1 = preprocs(data)   # Preprocess the data
    final_features = data1.drop(['Ins ID', 'Dr. ID'], axis =1)
    pred_prob = model.predict_proba(final_features)
    data2 = data1[['Pat_SO_ID']]
    

    ##for severity
    data['Pat_SO_ID'] = ''
    data['Pat_SO_ID'] = data['Patient ID'] + "-" + data['SO ID']
    data  = data.merge(data2.drop_duplicates(subset=['Patient_SO_ID']),how = 'left')
                
    insur_feat = pd.read_excel(r'C:\\Feature_rank.xlsx', sheet_name = 'Insurance Name')
    insur_feat['Insurance Name'] = insur_feat['Insurance Name'].str.upper()
    dxcode_feat = pd.read_excel(r'C:Feature_rank.xlsx', sheet_name = 'DX code')
    clinic_feat = pd.read_excel(r'C:Feature_rank.xlsx', sheet_name = 'Clinic ID')
    claim_feat = pd.read_excel(r'C:Feature_rank.xlsx', sheet_name = 'Claim Type')
    patstat_feat = pd.read_excel(r'C:Feature_rank.xlsx', sheet_name = 'Patient State')
    mod_feat = pd.read_excel(r'C:Feature_rank.xlsx', sheet_name = 'Modifier')
    hcpc_feat = pd.read_excel(r'C:Feature_rank.xlsx', sheet_name = 'HCPC code')
    
    
    data['Insurance Name'] = data['Insurance Name'].str.upper() 
    data = data.merge(insur_feat.drop_duplicates(subset=['Insurance Name']), how='left')
    data = data.merge(dxcode_feat.drop_duplicates(subset=['Dx Code']), how='left')
    data = data.merge(clinic_feat.drop_duplicates(subset=['Clinic ID']), how='left')
    data = data.merge(claim_feat.drop_duplicates(subset=['Claim Type']), how='left')
    data = data.merge(patstat_feat.drop_duplicates(subset=['Patient State']), how='left')
    data = data.merge(mod_feat.drop_duplicates(subset=['Modifier_dummy']), how='left')
    data = data.merge(hcpc_feat.drop_duplicates(subset=['HCPC_dummy']), how='left')
    data.drop(['HCPC_dummy','Modifier_dummy'], axis = 1, inplace = True)
        
    df = pd.DataFrame()
    df['Result']= [p[1] for p in pred_prob]
    data.drop_duplicates(inplace = True)
    sub = pd.concat([data, df], axis = 1) 
    sub.sort_values('Result', ascending = False, inplace = True)
    sub = sub.head(round(sub.shape[0]*percent))
    
    sub.to_csv(r'C:/Users//Output/' + fileNameSample.name[:-4] + '_Output.csv')
    print(time.process_time() - start)
    return render_template('index1.html', prediction_text='Predicted Successfully !')


if __name__ == "__main__":
    app.run(debug=True)
