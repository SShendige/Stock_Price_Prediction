from tkinter import *
import customtkinter as ctk
import pymongo
import time

def Stockregister():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    root = ctk.CTk()

    # Create the main window
    root.title("Register")
    root.geometry("600X600")
    root.minsize(600,600)
    root.maxsize(600,600)

    db_url = "mongodb://localhost:27017"
    client = pymongo.MongoClient(db_url)
    db = client.get_database("Stock")
    login_collection = db.get_collection("Login")

    def check_register():
        if register():
            message_label.configure(text="Successfully registered")
        else:
            message_label.configure(text="Invalid registration")

    def register():
        user_data = {
            "name":name.get(),
            "email_id":email.get(),
            "password":password.get()
        }
        auth = login_collection.insert_one(user_data)
        return auth



    # Create and place widgets
    heading = ctk.CTkLabel(root, text = "Welcome to S3 Stock Price Prediction Registration Platform", font = ("IBM Plex Sans Bold",20), text_color = "green")
    heading.place(x = 25, y = 30)

    # Name
    label_name = ctk.CTkLabel(root, text = "Name:", font = ("IBM Plex Sans Bold",16))
    label_name.place(x = 75, y = 100)

    name = ctk.CTkEntry(root, width = 250, font = ("IBM Plex Sans",16))
    name.place(x = 175, y = 100)

    # Email ID
    label_nameemail = ctk.CTkLabel(root, text = "Email ID: ", font = ("IBM Plex Sans Bold",16))
    label_nameemail.place(x = 75, y = 200)

    email = ctk.CTkEntry(root, width = 250, font = ("IBM Plex Sans",16))
    email.place(x = 175, y=200)

    # Password
    label_password = ctk.CTkLabel(root, text = "Password : ",font=("IBM Plex Sans Bold",16))
    label_password.place(x=75,y=248)

    password = ctk.CTkEntry(root, show = "*", width  = 250)
    password.place(x = 175, y = 248)

    registerbtn = ctk.CTkButton(root, text = "Submit", font = ("IBM Plex Sans Bold",20),command=check_register)
    registerbtn.place(x = 225, y = 400)


    message_label = ctk.CTkLabel(root, text="Already have an account? Click on login", font = ("IBM Plex Sans Bold",20))
    message_label.place(x = 175, y = 500)


    loginbtn = ctk.CTkButton(root,text="LOGIN",font = ("IBM Plex Sans Bold",20))
    loginbtn.place(x=200,y=550)


    root.mainloop()