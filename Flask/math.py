# -*- coding: utf-8 -*-
"""
Created on Sun May 30 12:37:26 2021

@author: nagoj
"""
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask,render_template
from tkinter import *
import PIL
from PIL import ImageTk,Image,ImageDraw,ImageGrab
import cv2
import tkcap
import numpy as np

app = Flask(__name__)

sess=tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
set_session(sess)
model = load_model(r"shape.h5")

@app.route("/",methods=["GET","POST"])
def index(): 
    return render_template('index.html')

@app.route('/launch',methods=['GET', 'POST'])
def launch():
    class pred:
        def __init__(self, master):
            self.master = master
            self.res = ""
            self.d=[]
            self.pre = [None, None]
            self.bs = 4.5
            self.c = Canvas(self.master,bd=3,relief="ridge", width=300, height=282, bg='white')
            self.c.pack(side=LEFT)
            self.c.pack(expand=YES,fill=BOTH)
            
            f1 = Frame(self.master, padx=5, pady=5)
            Label(f1,text="Maths Tutor for shape",fg="green",font=("",15,"bold")).pack(pady=5)
            Label(f1,text="Draw a shape to get its formula",fg="green",font=("",15)).pack()
            Label(f1,text="(Circle,Square,Triangle)",fg="green",font=("",15)).pack()
            self.pr = Label(f1,text="Prediction: None",fg="blue",font=("",15,"bold"))
            self.pr.pack(pady=20)
            
            Button(f1,font=("",15),fg="white",bg="red", text="Clear Canvas",command=self.clear).pack(side=BOTTOM)
            Button(f1,font=("",15),fg="white",bg="red", text="Predict",command=self.getResult).pack(side=BOTTOM)
            f1.pack(side=RIGHT,fill=Y)
            self.c.bind("<Button-1>", self.putPoint)
            self.c.bind("<B1-Motion>", self.paint)
    
        def getResult(self):
            x = self.master.winfo_rootx() + self.c.winfo_x()
            y = self.master.winfo_rooty() + self.c.winfo_y()
            x1 = x + self.c.winfo_width()
            y1 = y + self.c.winfo_height()
            cap=tkcap.CAP(self.master)
            cap.capture('dist.png',overwrite=True)
            im = Image.open(r"dist.png")
            width, height = im.size
            im1 = im.crop((5,50,width-420,height))
            im1.save('dist.png')
            index=['circle','square','triangle']
            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                img=image.load_img("dist.png",target_size=(64,64))
                x=image.img_to_array(img)
                x=np.expand_dims(x,axis=0)
                p=model.predict_classes(x)        
                self.res=str(index[p[0]])
                self.pr['text'] = "Prediction:"+self.res
            
            if self.res=='circle':
                i = cv2.imread('circle.jpg')
                cv2.imshow('circle',i)
                
            
            elif self.res=='square':
                i = cv2.imread('square.jpg')
                im=cv2.resize(i,(500,400))
                cv2.imshow('square',im)  
                    
            elif self.res=='triangle':
                i = cv2.imread('triangle.png')
                im=cv2.resize(i,(500,455))
                cv2.imshow('triangle',im)  
                
    
        def clear(self):
            self.c.delete('all')
    
        def putPoint(self, e):
            self.c.create_oval(e.x-self.bs,e.y-self.bs,e.x+self.bs,e.y+self.bs,outline='black',fill='black')
            self.pre=[e.x,e.y]
    
        def paint(self,e):
            self.c.create_line(self.pre[0],self.pre[1],e.x,e.y,width=self.bs*2,fill='black',capstyle=ROUND,smooth=True)
            self.pre = [e.x, e.y]
            
    if __name__ == "__main__":
        root = Tk()
        pred(root)
        root.title('Digit Classifier')
        root.resizable(0, 0)
        root.mainloop()
    return render_template("index.html")


        
if __name__=='__main__':
    app.run(debug=True)