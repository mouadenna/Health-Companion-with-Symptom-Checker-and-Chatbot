#C408
import peft
import bitsandbytes
import datasets
import accelerate
import loralib
import transformers
import sacremoses
import sentencepiece
import gradio as gr

import os

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM,AutoTokenizer, LlamaForCausalLM

import joblib
from deployML import predd

import time

#load the chatbot


model = LlamaForCausalLM.from_pretrained(
    "medalpaca/medalpaca-7b",
    return_dict=True,
    load_in_8bit=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b")


#load the first interface

def fn(*args):
    global symptoms
    all_symptoms = [symptom for symptom_list in args for symptom in symptom_list]
    if len(all_symptoms) > 17:
        raise gr.Error("Please select a maximum of 17 symptoms.")
    elif len(all_symptoms) < 3:
        raise gr.Error("Please select at least 3 symptoms.")
        symptoms = all_symptoms  # Update global symptoms list
    loaded_rf = joblib.load("model_joblib")
    return predd(loaded_rf,symptoms)



symptoms = []



demo = gr.Interface(
    fn, [
        gr.CheckboxGroup(['itching', 'skin rash', 'nodal skin eruptions', 'dischromic patches'], label='Skin Issues'),
        gr.CheckboxGroup(['continuous sneezing', 'shivering', 'chills', 'cough', 'breathlessness', 'phlegm', 'blood in sputum', 'throat irritation', 'runny nose', 'congestion', 'loss of smell', 'sinus pressure'], label='Respiratory Problems'),
        gr.CheckboxGroup(['stomach pain', 'acidity', 'ulcers on tongue', 'vomiting', 'nausea', 'loss of appetite', 'abdominal pain', 'burning micturition', 'spotting urination', 'passage of gases', 'internal itching', 'indigestion', 'muscle wasting', 'patches in throat', 'constipation'], label='Digestive Complaints'),
        gr.CheckboxGroup(['high fever', 'fatigue', 'weight loss', 'restlessness', 'lethargy', 'mild fever'], label='Fever and Fatigue'),
        gr.CheckboxGroup(['blurred and distorted vision', 'red spots over body', 'pain behind the eyes', 'redness of eyes'], label='Vision and Eye Problems'),
        gr.CheckboxGroup(['chest pain', 'fast heart rate', 'swelling of stomach'], label='Cardiovascular Issues'),
        gr.CheckboxGroup(['muscle pain', 'joint pain', 'pain in anal region', 'painful walking', 'movement stiffness'], label='Joint and Muscle Pain'),
        gr.CheckboxGroup(['headache', 'dizziness', 'loss of balance', 'lack of concentration', 'stiff neck', 'depression', 'irritability', 'visual disturbances', 'back pain', 'weakness in limbs', 'neck pain', 'weakness of one body side', 'altered sensorium'], label='Neurological Symptoms'),
        gr.CheckboxGroup(['dark urine', 'sweating', 'mucoid sputum', 'toxic look (typhos)', 'bladder discomfort', 'foul smell of urine', 'continuous feel of urine'], label='Urinary Issues'),
        gr.CheckboxGroup(['skin peeling', 'silver like dusting', 'small dents in nails', 'inflammatory nails', 'blister', 'red sore around nose', 'yellow crust ooze'], label='Skin Abnormalities'),
        gr.CheckboxGroup(['family history', 'headache', 'mood swings', 'anxiety', 'slurred speech', 'palpitations', 'drying and tingling lips'], label='Psychological Symptoms'),
        gr.CheckboxGroup(['knee pain', 'hip joint pain', 'swelling joints'], label='Joint and Bone Issues'),
        gr.CheckboxGroup(['spinning movements', 'unsteadiness'], label='Neurological Movements')
    ],
    outputs="textbox",allow_flagging="never"
)






def predict(message, history):
    prompt = f"""
    Answer the following question:
    {message}/n
    Answer:
    """
    batch = tokenizer(prompt, return_tensors='pt')
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=200)
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True).replace(prompt,"")


loaded_rf = joblib.load("model_joblib")
Fmessage="hello, I'm here to help you!"



chatbot=gr.ChatInterface(predict, chatbot=gr.Chatbot(value=[(None, Fmessage)],),clear_btn=None, retry_btn=None, undo_btn=None)


gr.TabbedInterface(
    [demo, chatbot], ["symptoms checker", "chatbot"]
).queue().launch()