#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:39:13 2022

@author: prekshaparmar
"""

from flask import Flask, request, render_template

app = Flask(__name__)

from textblob import TextBlob
from transformers import pipeline

model = pipeline('sentiment-analysis', "mrm8488/bert-small-finetuned-squadv2")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method =="POST":
        text = request.form.get("text")
        print(text)
        r1 = TextBlob(text).sentiment
        r2 = model(text)
        return(render_template("index.html", result1=r1, result2=r2))
    else:
        return(render_template("index.html", result1="2", result2="2"))   
    
if __name__=="__main__":
    app.run()
