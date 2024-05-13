from tkinter import *
import os


def cam():
        print("")
        os.system('python 07_camera.py')

def captureDataset():
        print("")
        os.system('python 01_face_dataset.py')

def faceRec():
        print("")
        os.system('python 03_face_recognition.py')

def realtime():
        print("")
        os.system('python 04_Object.py')


def STS():
        print("")
        os.system('python 05_signtospeech.py')


def OCR():
        print("")
        os.system('python 06_ocr.py')

def start():
        global root 
        root = Tk()
        root.title('Image Recognition For Dimentia Patients')
        canvas = Canvas(root,width = 720,height = 50, bg = 'yellow')
        canvas.grid(column = 0 , row = 0)

        heading = Label(root,text="Image Recognition",fg="#FFA500",bg="#fff")
        heading.config(font=('calibri 25'))
        heading.place(relx=0.32,rely=0.45)
        heading.grid(column = 0 , row = 0)
        headings = Label(root,text="Select any one of the option",fg="#bf00ff",bg="#fff")
        headings.config(font=('calibri 25'))
        headings.place(relx=0.32,rely=0.45)
        headings.grid(column = 0 , row = 1)
        
        buttoncd = Button(root, text='Capture Dataset',command = captureDataset,bg="red",fg="yellow") 
        buttoncd.configure(width = 102,height=2, activebackground = "#33B5E5", relief = RAISED)
        buttoncd.grid(column = 0 , row = 2)

        buttonfr = Button(root, text='Face Recognition',command = faceRec,bg="red",fg="yellow") 
        buttonfr.configure(width = 102,height=2, activebackground = "#33B5E5", relief = RAISED)
        buttonfr.grid(column = 0 , row = 3)

        buttonr = Button(root, text='Realtime Object Detection',command = realtime,bg="red",fg="yellow") 
        buttonr.configure(width = 102,height=2, activebackground = "#33B5E5", relief = RAISED)
        buttonr.grid(column = 0 , row = 4)


        buttonocr = Button(root, text='Optical Character Recognition',command = OCR,bg="red",fg="yellow") 
        buttonocr.configure(width = 102,height=2, activebackground = "#33B5E5", relief = RAISED)
        buttonocr.grid(column = 0 , row = 5)

        buttonc = Button(root, text='Camera',command = cam,bg="red",fg="yellow") 
        buttonc.configure(width = 102,height=2, activebackground = "#33B5E5", relief = RAISED)
        buttonc.grid(column = 0 , row = 6)
        
        exit_button = Button(root, text="Exit", command=root.destroy) 
        exit_button.configure(width = 102,height=2, activebackground = "#33B5E5", relief = RAISED)
        exit_button.grid(column = 0 , row =7)

        root.mainloop()
    
    
if __name__=='__main__':
    start()
