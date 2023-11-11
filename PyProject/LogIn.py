from tkinter import *
from tkinter import messagebox
import pymongo
import customtkinter as ctk
import time
from Register import Stockregister



import os

filepath = "C:/Users/sathv/OneDrive/Documents/VSCodes/Pyproject/stkp_searchbox.py"

def execute_python_file(file_path):
   try:
      os.system(f'python {file_path}')
   except FileNotFoundError:
      print(f"Error: The file '{file_path}' does not exist.")


def msg():
    messagebox.showinfo("Login Successful", "You have successfully logged in!")


def login(): 
    username = entry_username.get()
    password = entry_password.get()   # Check the credentials (you can replace this with your own authentication logic)
    name = entry_username.get()
    os.environ["USERNAME"] = name
    if check_login(username,password):
        messagebox.showinfo("Login Successful", "You have successfully logged in!")
        execute_python_file(filepath)
    else:
        messagebox.showerror("Login Failed", "Invalid credentials. Please try again.")
    
    #root.after(3000,root.destroy())


def check_login(username,password):
    db_url = "mongodb://localhost:27017"
    client = pymongo.MongoClient(db_url)
    db = client.get_database("Stock")
    login_collection = db.get_collection("Login")
    login = login_collection.find_one({"email_id":username,"password":password})
    return login is not None

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")
root = ctk.CTk()

# Create the main window
root.title("Login")


root.geometry("600X600")
root.minsize(600,600)
root.maxsize(600,600)
# Create and place widgets
text = ctk.CTkLabel(root, text = "S3 Stock Prediction", font = ("IBM Plex Sans Bold",20), text_color = "yellow")
text.place(x = 220, y = 30)

disclaimer = ctk.CTkLabel(root, text = "⚠Invest at your on risk", font = ("IBM Plex Sans Bold",20), text_color = "red")
disclaimer.place(x = 200, y = 55)

label_username = ctk.CTkLabel(root, text="Username:",font = ("IBM Plex Sans Bold",20))
label_username.place(x=250,y=150)

entry_username = ctk.CTkEntry(root,width = 350,font=("IBM Plex Sans Bold",20))
entry_username.place(x = 120, y = 200)



label_password = ctk.CTkLabel(root, text="Password:", font = ("IBM Plex Sans Bold",20))
label_password.place(x = 250, y = 300)

entry_password = ctk.CTkEntry(root, show="*", width = 350,font=("IBM Plex Sans Bold",20))
entry_password.place(x = 120, y = 350)

login_button = ctk.CTkButton(root, text="Login", command=login, font = ("IBM Plex Sans Bold",20))
login_button.place(x = 225, y = 400)


def un():
    uname = entry_username.get()
    print(uname)


message_label = ctk.CTkLabel(root, text="Dont have an account yet?!\nCreate one now!", font = ("IBM Plex Sans Bold",20))
message_label.place(x = 175, y = 500)

register = ctk.CTkButton(root, text = "Create Account",command=Stockregister ,font = ("IBM Plex Sans Bold", 20))
register.place(x = 220, y = 550)

# Start the main event loop
root.mainloop()