import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tkinter import font                          #importing libraries
from PIL import ImageTk,Image
import numpy as np

from keras.models import load_model
model=load_model('bestModel.h5')                #importing saved model


def upload_image():
    file_path=filedialog.askopenfilename()
    uploaded=Image.open(file_path)
    uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))           #upload image method
    im=ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image=im
    label.configure(text=' ')
    show_classify_button(file_path)


def show_classify_button(file_path):
        classify_btn=Button(top,text="Classify Image", command= lambda:classify(file_path),padx=10,pady=5)             #display the classify button
        classify_btn.configure(background="#364156",foreground="white",font=('arial',10,'bold'))
        classify_btn.place(relx=0.79,rely=0.46)


def classify(file_path):
    image = Image.open(file_path)
    image=image.resize((200,200))
    image=np.expand_dims(image,axis=0)                   #passes the image to the model and retrieves the binary output
    image=np.array(image)
   
    val=model.predict(image)
    if val==0:
        label.configure(foreground="#011638",text="With Mask")

    else :
     label.configure(foreground="#011638",text="Without Mask")





#initialize GUI
top=tk.Tk()
top.geometry('800x600')
top.title("Face Mask Classification")
top.configure(background="#CDCDCD")

#set heading
heading=Label(top, text="Face Mask Classification" ,pady=20,font=('arial',20,'bold'))
heading.configure(background="#CDCDCD",foreground='#364756')
heading.pack()

upload=Button(top,text="Upload an Image", command=upload_image,padx=10,pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)


#upload image
sign_image=Label(top)
sign_image.pack(side=BOTTOM,expand=True)


#predicted class
label=Label(top,background="#CDCDCD",font=('arial',15,'bold'))
label.pack(side=BOTTOM,expand=True)




top.mainloop()