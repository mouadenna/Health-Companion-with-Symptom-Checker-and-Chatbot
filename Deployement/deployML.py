import joblib
import sklearn
import pandas as pd
import numpy as np
loaded_rf = joblib.load("model_joblib")

Description=pd.read_csv("symptom_Description.csv")
severity=pd.read_csv("Symptom-severity.csv")
severity['Symptom'] = severity['Symptom'].str.replace('_',' ')
precaution = pd.read_csv("symptom_precaution.csv")

def predd(x,psymptoms):
    #print(psymptoms)
    psymptoms.extend([0] * (17-len(psymptoms)))

    a = np.array(severity["Symptom"])
    b = np.array(severity["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j]==a[k]:
                psymptoms[j]=b[k]
    psy = [psymptoms]
    pred2 = x.predict(psy)
    disp= Description[Description['Disease']==pred2[0]]
    disp = disp.values[0][1]
    recomnd = precaution[precaution['Disease']==pred2[0]]
    c=np.where(precaution['Disease']==pred2[0])[0][0]
    precuation_list=[]
    for i in range(1,len(precaution.iloc[c])):
          precuation_list.append(precaution.iloc[c,i])
    combined_info = f"The Disease Name: {pred2[0]}\nThe Disease Description: {disp}\nRecommended Things to do at home:"+''.join([f'\n   -{i}' for i in precuation_list])
    return combined_info