from tkinter import *
import os
from tkinter import ttk
import tkinter.font as tkFont
from PIL import Image, ImageTk
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import customtkinter
import colour
from functools import partial
import tkinter
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
#from Stock_login import un

continue_prediction = True
canvas = None
animation = None
toolbar = None

def toggle():
    global continue_prediction
    continue_prediction = not continue_prediction
    if not continue_prediction:
        if animation:
            animation.event_source.stop()
        if canvas:
            canvas.get_tk_widget().destroy()
        if toolbar:
            toolbar.destroy()

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")
root = customtkinter.CTk()
root.title("Stock Prediction")
root.geometry("800x600")
root.wm_attributes('-transparentcolor')


#root.attributes('-fullscreen',True)



logo_width = 210
logo_height = 180
logo = Image.open("C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/Logo1.png")
logo_resized = logo.resize((logo_width,logo_height),Image.LANCZOS)
logo_image = ImageTk.PhotoImage(logo_resized)

#bglabel = Frame(root,bg="#2193b0")
#bglabel.pack(fill='both',expand=True)

notebook = ttk.Notebook(root)
notebook.pack(fill=BOTH,expand=True)
#notebook.place(x=380,y=200)
frame1 = ttk.Frame(notebook)
frame2 = ttk.Frame(notebook)
frame3 = ttk.Frame(notebook)
frame4 = ttk.Frame(notebook)

Labelnote1 = Frame(frame1,bg='#2193b0')
Labelnote1.pack(fill=BOTH,expand=True)

Labelnote2 = Frame(frame2,bg='green')
Labelnote2.pack(fill=BOTH,expand=True)

Labelnote3 = Frame(frame3,bg='#2193b0')
Labelnote3.pack(fill=BOTH,expand=True)

Labelnote4 = Frame(frame4,bg='#2193b0')
Labelnote4.pack(fill=BOTH,expand=True)

notebook.add(frame1,text='Home')
notebook.add(frame2,text='Prediction')
notebook.add(frame3,text='Aboutus')
notebook.add(frame4,text='Analysis')

Account_logo = Image.open("C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/account.png")
Account_image = ImageTk.PhotoImage(Account_logo)

name = os.environ.get("USERNAME")
account_label = Label(Labelnote1,image=Account_image,text=name,compound=LEFT,bg='#2193b0',fg='yellow',font=("IBM Plex Sans Bold",20))
account_label.place(x=70,y=10)


#usernam = Stock_login.dummyname
#account_label.config(text=usernam)

#account_label


s = ttk.Style()
s.theme_use('default')
s.configure('TNotebook.Tab', background="#2193b0")
s.map("TNotebook", background= [("selected", "green3")])


Panel1 = Frame(Labelnote1,bg='aqua',width=1400,height=650)
Panel1.place(x=350,y=230)

Panel2 = Frame(Labelnote3,bg='aqua',width=1900,height=850)
Panel2.place(x=10,y=40)

AboutUs = Label(Panel2,text="ABOUT US",fg="red",bg='aqua',font=("Comic Sans MS Bold",25))
AboutUs.place(x=850,y=30)

logos = Label(Labelnote1,image=logo_image,bg='#2193b0')
logos.place(x=738,y=3)

Label1 = Label(Labelnote1,text="S3 STOCK PREDICTOR",fg="dark red",bg="#2193b0",font=("Comic Sans MS Bold",25))
Label1.place(x=920,y=50)


logos_width = 170
logos_height = 130
logo_small = Image.open("C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/Logo1.png")
logos_resized = logo_small.resize((logos_width,logos_height),Image.LANCZOS)
logo_small_image = ImageTk.PhotoImage(logos_resized)




companies = {
    "Amazon": "AMZN",
    "Tata Motors":"TATAMOTORS.NS",
    "Samvardhana MOTHERSON":"MOTHERSON.NS",
    "Apple":"AAPL",
    "Yes Bank" : "YESBANK.NS",
    "ICICI Bank":"ICICIBANK.NS",
    "TESLA":"TSLA",
    "Google": "GOOG"
}

company_logo = {
    "Amazon": "C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/amazon.png",
    "Google": "C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/google.png",
    "Tata Motors" : "C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/Tatamotors.png",
    "Apple" : "C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/apple.png",
    "Yes Bank" : "C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/yes.png",
    "ICICI Bank" : "C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/Icici.png",
    "TESLA" : "C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/Tesla.png"
}

#COMPANY DATA


def op(comp_ticker):
    global d
    d = yf.download(comp_ticker, period="1y")
    #print(data)




def comp_analyze():

    
    comp_logo1 = Image.open("C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/amazon.png")
    comp_logo_resized1 = comp_logo1.resize((330,200),Image.LANCZOS)
    comp_image1 = ImageTk.PhotoImage(comp_logo_resized1)

    comp_logo2 = Image.open("C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/google.png")
    comp_logo_resized2 = comp_logo2.resize((330,200),Image.LANCZOS)
    comp_image2 = ImageTk.PhotoImage(comp_logo_resized2)

    comp_logo3 = Image.open("C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/Tatamotors.png")
    comp_logo_resized3 = comp_logo3.resize((330,200),Image.LANCZOS)
    comp_image3 = ImageTk.PhotoImage(comp_logo_resized3)

    comp_logo4 = Image.open("C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/apple.png")
    comp_logo_resized4 = comp_logo4.resize((330,200),Image.LANCZOS)
    comp_image4 = ImageTk.PhotoImage(comp_logo_resized4)

    comp_logo5 = Image.open("C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/yes.png")
    comp_logo_resized5 = comp_logo5.resize((330,200),Image.LANCZOS)
    comp_image5 = ImageTk.PhotoImage(comp_logo_resized5)

    comp_logo6 = Image.open("C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/Icici.png")
    comp_logo_resized6 = comp_logo6.resize((330,200),Image.LANCZOS)
    comp_image6 = ImageTk.PhotoImage(comp_logo_resized6)

    comp_logo7 = Image.open("C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/Tesla.png")
    comp_logo_resized7 = comp_logo7.resize((330,200),Image.LANCZOS)
    comp_image7 = ImageTk.PhotoImage(comp_logo_resized7)


    
    def c_logo(x,y,text,comp_logo):
        comp1 = Label(Labelnote4,text=text,compound=TOP,bg='white',font=("IBM Plex Sans bold",14))
        comp1.image = comp_image1,comp_image2,comp_image3,comp_image4,comp_image5,comp_image6,comp_image7
        comp1.config(image=comp_logo)
        comp1.place(x=x,y=y)

    c_logo(300,30,"Amazon",comp_image1)
    c_logo(800,30,"Google",comp_image2)
    c_logo(1300,30,"Tata Motors",comp_image3)
    c_logo(300,340,"Apple",comp_image4)
    c_logo(800,340,"Yes Bank",comp_image5)
    c_logo(1300,340,"ICICI Bank",comp_image6)
    c_logo(300,650,"Tesla",comp_image7)
    c_logo(800,650,"Amazon",comp_image1)
    c_logo(1300,650,"Amazon",comp_image1)

    def openprice(x,y,comp_ticker):
        comp1_1 = Label(Labelnote4,text="Open: ",height=1,width=30,bg='white',font=('IBM Plex Sans Bold',14))
        comp1_1.place(x=x,y=y)

        #d = yf.download(comp_ticker, period="1y")
        #print(data)
        op(comp_ticker)
        p_open = d['Open']#.values.tolist()
        #print(p_open)
        d1 = np.array(p_open).flatten()[-1:][0].round(2)
        comp1_1.config(text='Open: '+str(d1))
    
    openprice(300,260,'AMZN')
    openprice(800,260,'GOOG')
    openprice(1300,260,'TATAMOTORS.NS')
    openprice(300,570,'AAPL')
    openprice(800,570,'YESBANK.NS')
    openprice(1300,570,'ICICIBANK.NS')
    openprice(300,880,'TSLA')
    openprice(800,880,'AMZN')
    openprice(1300,880,'AMZN')


    def closeprice(x,y,comp_ticker):
        comp1_2 = Label(Labelnote4,text="Close: ",height=1,width=30,bg='white',font=('IBM Plex Sans Bold',14))
        comp1_2.place(x=x,y=y)

        op(comp_ticker)
        p_close = d['Close']#.values.tolist()
        #print(prices_close)
        d2 = np.array(p_close).flatten()[-1:][0].round(2)
        comp1_2.config(text='Close: '+ str(d2))
        #print(data2)

    closeprice(300,290,'AMZN')
    closeprice(800,290,'GOOG')
    closeprice(1300,290,'TATAMOTORS.NS')
    closeprice(300,600,'AAPL')
    closeprice(800,600,'YESBANK.NS')
    closeprice(1300,600,'ICICIBANK.NS')
    closeprice(300,910,'TSLA')
    closeprice(800,910,'AMZN')
    closeprice(1300,910,'AMZN')




comp_analyze()


def data1(company_name):
    logos_width1 = 210
    logos_height1 = 180
    Company_info = Label(Panel1, height=20, width=50, font=('IBM Plex Sans', 14, 'bold'))
    Company_info.place(x=100, y=100)
    Company_info.config(text=company_name)
    
    image_path = company_logo.get(company_name)
    Amazon_logo = Image.open(image_path)
    logos_resized1 = Amazon_logo.resize((logos_width1, logos_height1), Image.LANCZOS)
    Amazon_image = ImageTk.PhotoImage(logos_resized1)
    
    # Create a label to display the image
    Company_logo = Label(Panel1, height=440, width=600, bg='white')
    Company_logo.image = Amazon_image  # Store the reference to the PhotoImage object
    Company_logo.config(image=Amazon_image)
    Company_logo.place(x=700, y=100)
    
def oc_field(o,c):
    o_label = Label(Panel1, bg = 'yellow', font = ("IBM Plex Sans Bold", 20))
    o_label.place(x = 100, y = 600)
    o_label.config(text = 'Open Price : '+str(o))
    
    c_label = Label(Panel1, bg = 'orange', font = ("IBM Plex Sans Bold", 20))
    c_label.place(x = 350, y = 600)
    c_label.config(text = 'Close Price : '+str(c))

def openclose(Comp_ticker):
    data = yf.download(Comp_ticker, period="1y")
    print(data)
    
    prices_open = data['Open']#.values.tolist()
    print(prices_open)
    data1 = np.array(prices_open).flatten()[-1:][0].round(2)
    print(data1)
    
    prices_close = data['Close']#.values.tolist()
    print(prices_close)
    data2 = np.array(prices_close).flatten()[-1:][0].round(2)
    print(data2)
    
    oc_field(data1,data2)
    
#openclose("AMZN")


def open_page():
    notebook.select(frame2)

def aboutus():
    Text_box1 = Label(Panel2,height=30,width=150,font=('IBM Plex Sans',14, 'bold'))
    Text_box1.place(x=50,y=120)
    string1 = '''
    Visionaries and Financial Heroes

    In the realm of financial innovation, there remarkable minds stand tall: Sathvik N Shendige (1BG21CS076), 
    Sujay G Kaushik (1BG21CS095) and Sumukha S Kashyap (1BG21CS096). 
    Each possessing distinct talents, they form the trinity behind groundbreaking solution that has reshaped 
    stock market prediction: “S3 Analytics”. 
    Heroes BTS (Behind the Scene) 
    Sathvik – The Thinker, helps understand why it is important to guess the price of stock. Making it easier 
    for the user to know what a stock is and leading the path to beginners to invest in stock so that they have 
    their own passive income
    Sujay – The Computer Whiz, the one who creates miracles, the one who plays with numbers and writes special 
    codes so that the digital numbers help us visualize the pattern in the stock price second to second. 
    It’s like a super smart robot to help you figure out things.
    Sumukha – The Mega Mind. Taking Sathvik’s ideas and Sujay’s code and makes a plan for how to use them. 
    Its like making a map to find the tressure based on the clues you have.
    Smart Money:
    Majority have very minimum to no knowledge about the stocks which makes many users take a step back from investing. 
    These 3 members help people make smarter decisions about buying and selling things.
    Changing the Money Game
    We all believe that money rules the world, and there are people who work for money. How many have you ever wished 
    to make the money work for you; earn money even while sleeping. 
    Well, the answer lies in passive income. 
    These passive incomes help you earn what you missed during your vacations as this grow even when you don’t work for them. 
    They multiply like a weed, thus increasing your net worth. 
    The financial heroes – Sathvik with big ideas ,Sujay uses computer to help and Sumukha plans everything. 
    Together, they are changing the way we think about money and helping us make better choices
    '''
    Text_box1.config(text=string1)

aboutus()

def data():
    Text_box = Label(Panel1,height=20,width=100,font=('IBM Plex Sans',14, 'bold'))
    Text_box.place(x=100,y=100)
    string1 = '''
    Welcome to our S3 STOCK PREDICTOR.

    This software is made by Sujay, Sathvik and Sumukha using python language which helps the developers
    in easy coding as it has libraries that can be directly imported such MatPLotLib that is used to visisualize 
    the data that is imported from yfinanace library which extracts the data of the comapny directly from the yfinance
    website. 
    Please select a company from the list below and click on predict button for stock prediction.

    '''
    Text_box.config(text=string1)
    

data()
value_label = Label(Labelnote1, text="TICKER SYMBOL: ",fg='red',font=("IBM Plex Sans Bold",20))
value_label.pack(pady=5)
value_label.place(x=500,y=900)

# home page logo 

#logos.config(width=200,height=200)



#Demo graph in Tkinter
#---------------------------------------------------------------------------------------------------

"""data1 = {'country': ['A', 'B', 'C', 'D', 'E'],
        'gdp_per_capita': [45000, 42000, 52000, 49000, 47000]
        }
df1 = pd.DataFrame(data1)
figure1 = plt.Figure(figsize=(6, 5), dpi=100)
ax1 = figure1.add_subplot(111)
bar1 = FigureCanvasTkAgg(figure1, frame1)
bar1.get_tk_widget().pack(fill='both',expand=True)
df1 = df1[['country', 'gdp_per_capita']].groupby('country').sum()
df1.plot(kind='bar', legend=True, ax=ax1)
ax1.set_title('Country Vs. GDP Per Capita')"""


#---------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------


def start_prediction(ticker_from,company_title):

    global continue_prediction,canvas,animation,toolbar

    if canvas:
        canvas.get_tk_widget().destroy()
    if toolbar:
        toolbar.destroy()
    canvas = None
    animation = None
    toolbar = None

    fig, ax = plt.subplots()

    # Create lines for actual and predicted prices (initially empty)
    actual_line, = ax.plot([], [], label='Actual Price', color='red')
    predicted_line, = ax.plot([], [], label='Predicted Price', color='blue')

    # Add labels and title to the plot
    ax.set_xlabel('Days')
    ax.set_ylabel('Price')
    ax.set_title(company_title.upper())

    # Add a legend
    ax.legend()

    # Set initial limits for the x-axis and y-axis
    ax.set_xlim(0, 100)  # Adjust the limits as needed
    ax.set_ylim(0, 100)  # Adjust the limits as needed

    ticker_symbol = ticker_from

    # Initialize an empty list for storing the predicted prices
    predictions = []

    # Retrieve the historical stock price data from Yahoo Finance
    data = yf.download(ticker_symbol, period="1y")

    # Extract the adjusted closing prices
    prices = data['Close'].values.tolist()  # Convert to Python list

    # Create an array of days (starting from 1)
    days = np.arange(1, len(prices) + 1).reshape(-1, 1)
    
    # Set initial limits for the x-axis and y-axis
    ax.set_xlim(0, len(prices))  # Adjust the limits based on the data length
    ax.set_ylim(min(prices) * 0.9, max(prices) * 1.1)  # Adjust the limits with a buffer

    # Perform rolling prediction
    prediction_days = 10

    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}

    predictions = []

    for i in range(len(prices) - prediction_days):
        X_train = days[i:i + prediction_days - 1]
        y_train = prices[i + 1:i + prediction_days]

        # Create a Ridge regression model
        regressor = Ridge()

        # Perform GridSearchCV to find the best hyperparameters
        grid_search = GridSearchCV(regressor, param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_alpha = grid_search.best_params_['alpha']

        # Train the model with the best hyperparameters
        regressor = Ridge(alpha=best_alpha)
        regressor.fit(X_train, y_train)

        # Generate the prediction for the next day
        next_day = days[i + prediction_days].reshape(1, -1)
        prediction = regressor.predict(next_day)
        predictions.append(prediction[0])  # Extract the scalar value from the prediction array

    # Flatten the predictions list
    predictions_flat = np.array(predictions).flatten()

    # Update the actual and predicted lines with the data
    actual_line.set_data(days, prices)
    predicted_line.set_data(days[prediction_days:], predictions_flat)

    #print(prediction_days)

    # Enable mplcursors for the actual line
    mplcursors.cursor(actual_line).connect("add", lambda sel: sel.annotation.set_text(f"${sel.target[1]:.2f}"))

    # Enable mplcursors for the plot
    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(f"Price: ${prices[int(sel.target[0]) - 1]:.2f}\nDay: {int(sel.target[0])}")
    )

    canvas = FigureCanvasTkAgg(fig, master=Labelnote2)
    canvas.draw()

    # Add the canvas to Labelnote2
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas,Labelnote2)
    toolbar.update()
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def update(frame):
        if not continue_prediction:
            ani.event_source.stop()
            return
        # Retrieve the latest stock price data from Yahoo Finance
        latest_data = yf.download(ticker_symbol, period="1d")

        # Extract the latest adjusted closing price
        latest_price = latest_data['Close'].values[-1]

        # Append the latest price to the prices list
        prices.append(latest_price)

        # Append the latest day to the days array
        days = np.arange(1, len(prices) + 1).reshape(-1, 1)
        
        # Perform rolling prediction for the last 30 days
        prediction_days = 15
        predictions = []

        for i in range(len(prices) - prediction_days):
            X_train = days[i:i + prediction_days - 1]
            y_train = prices[i + 1:i + prediction_days]

            # Create a Ridge regression model
            regressor = Ridge()

            # Perform GridSearchCV to find the best hyperparameters
            grid_search = GridSearchCV(regressor, param_grid, cv=5)
            grid_search.fit(X_train, y_train)

            # Get the best hyperparameters
            best_alpha = grid_search.best_params_['alpha']

            # Train the model with the best hyperparameters
            regressor = Ridge(alpha=best_alpha)
            regressor.fit(X_train, y_train)

            # Generate the prediction for the next day
            next_day = days[i + prediction_days].reshape(1, -1)
            prediction = regressor.predict(next_day)
            predictions.append(prediction[0])  # Extract the scalar value from the prediction array
            
        #print(prediction)
        mse = mean_squared_error(y_train, regressor.predict(X_train))
        # Update the plot with the new data
        actual_line.set_data(days, prices)
        predicted_line.set_data(days[prediction_days:], np.array(predictions).flatten())
        print(np.array(predictions).flatten())
        last_predicted = np.array(predictions).flatten()[-1:][0]
        print("Latest value:"+str(last_predicted))
        last_but_one_predicted = np.array(predictions).flatten()[-2]
        print("Second Latest value:"+str(last_but_one_predicted))

        #Mean SQUARED ERROR
        print(f"Mean Squared Error (MSE): {mse}")
        mae = mean_absolute_error(y_train, regressor.predict(X_train))
        print(f"Mean Absolute Error (MAE): {mae}")
        #actual_prices = prices[prediction_days:]
        #mape = calculate_mape(actual_prices, predictions_flat)
        #print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


        Prediction_label = Label(Labelnote2,text='Predicted Value:',font=("Helvetica",20))
        Prediction_label.place(x=1200,y=900)
        Pred_value = Label(Labelnote2,text='',font=('Comic Sans MS Bold',20))
        Pred_value.place(x=1450,y=900)
        if last_predicted > last_but_one_predicted:
            col = 'green'
            Pred_value.config(text=' ⬆'+str(last_predicted),fg=col)
            #Prediction_label.config(text='📈Predicted Value:'+str(last_predicted),fg=col)
        else:
            col = 'red'
            Pred_value.config(text=' ⬇'+str(last_but_one_predicted),fg=col)
            #Prediction_label.config(text='📉Predicted Value:'+str(last_predicted),fg=col)
        


    # Create an animation that calls the update function every 10 seconds
    ani = FuncAnimation(fig, update, interval=10000)

    # Display the plot
    plt.show()
    #ani.save()

#---------------------------------------------------------------------------------------------------
companies1 = {
    "AMZN": "Amazon",
    "GOOG": "Google",
    "TATAMOTORS.NS" : "Tata Motors",
    "MOTHERSON.NS" : "Samvardhana Motherson",
    "MSUMI.NS" : "Motherson Sumi Wiring India",
    "TSLA" : "Tesla",
    "AAPL" : "Apple",
    "YESBANK.NS" : "Yes Bank",
    "ICICIBANK.NS":"ICICI Bank",
    "RELIANCE.NS":"RELIANCE",
}



#---------------------------------------------------------------------------------------------------
def fade(widget, smoothness=5, cnf={}, **kw):
    """This function will show faded effect on widget's different color options.

    Args:
        widget (tk.Widget): Passed by the bind function.
        smoothness (int): Set the smoothness of the fading (1-10).
        background (str): Fade background color to.
        foreground (str): Fade foreground color to."""

    kw = tkinter._cnfmerge((cnf, kw))
    if not kw: raise ValueError("No option given, -bg, -fg, etc")
    if len(kw)>1: return [fade(widget,smoothness,{k:v}) for k,v in kw.items()][0]
    if not getattr(widget, '_after_ids', None): widget._after_ids = {}
    widget.after_cancel(widget._after_ids.get(list(kw)[0], ' '))
    c1 = tuple(map(lambda a: a/(65535), widget.winfo_rgb(widget[list(kw)[0]])))
    c2 = tuple(map(lambda a: a/(65535), widget.winfo_rgb(list(kw.values())[0])))
    colors = tuple(colour.rgb2hex(c, force_long=True)
                for c in colour.color_scale(c1, c2, max(1, smoothness*100)))

    def worker(count=0):
        if len(colors)-1 <= count: return
        widget.config({list(kw)[0] : colors[count]})
        widget._after_ids.update( { list(kw)[0]: widget.after(
            max(1, int(smoothness/10)), worker, count+1) } )
    worker()


def bg_config(widget, bg, fg, event):
    fade(widget, smoothness=5, fg=fg, bg=bg)

#-------------------------------------------------------------------------------------------------------
search_width = 50
search_height = 40
search_small = Image.open("C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/search_ticker.png")
search_resized = search_small.resize((search_width,search_height),Image.LANCZOS)
search_small_image = ImageTk.PhotoImage(search_resized)

search_label = Label(Labelnote1,image=search_small_image,width=40,height=40,bg='#2193b0')
search_label.place(x=1400,y=90)
Input_ticker = Label(Labelnote1,text='Enter Ticker Symbol: ',bg='#2193b0',fg='yellow',font=("IBM Plex Sans bold",20))
Input_ticker.place(x=1450,y=50)
Input_text=Entry(Labelnote1,width=20,font=("IBM Plex Sans Bold",25))
Input_text.place(x=1450,y=90)
Input_text.config(highlightbackground='red')
Input_text.get()


def input_field(e):
    if Input_text.get() in companies1:
        Comp = companies1[Input_text.get()]
    data1(Comp)
    openclose(Input_text.get())
    #navigation to frame2 of notebook
    open_page()
    start_prediction(Input_text.get(), Comp)

root.bind('<Return>',input_field)

# data1, openclose, open_page, start_prediction

#---------------------------------------------------------------------------------------------------------

"""start=Button(root, text="Start",font=("IBM Plex Sans Bold",16),bg='#008000')
start.pack()
start.place(x=1500, y=200)"""

end=Button(Labelnote2, text="Stop",font=("IBM Plex Sans Bold",16),bg='#FF0000',command=toggle)
end.pack()
end.place(x=1760, y=170)


def openabout():
    notebook.select(frame3)
#---------------------------------------------------------------------------------------------------------
def toggle_side_panel():
    side_panel = Frame(Labelnote1, width=300, height=1000,bg='#262626')
    side_panel.place(x=0, y=0)
    def combobox():
        combo = ttk.Combobox(side_panel,values=list(companies.keys()),width=18,background='red',font=("IBM Plex Sans",20))
        combo.place(x=15,y=540)
        def select_company():
            global selected_company
            selected_item = combo.get()
            selected_company = str(companies[selected_item])
            value_label.config(text="TICKER SYMBOL: " + selected_company)
            data1(selected_item)
            open_page()
            start_prediction(selected_company, combo.get())
            
        select_btn = Button(side_panel,text='SELECT',command=select_company,font=("IBM Plex Sans",15))
        select_btn.place(x=90,y=800)
        
    logos = Label(side_panel,image=logo_small_image,bg='#262626')
    logos.place(x=50,y=10)


    def btn(x,y,text,bcolor,fcolor,cmd):

        def enter(e):
            Button1['background'] = bcolor
            Button1['foreground'] = 'red'
        
        def exit(e):
            Button1['background'] = fcolor
            Button1['foreground'] = 'red'

        Button1 = Button(side_panel,
                        text=text,
                        width=28,
                        height=2,
                        fg='red',
                        border=0,
                        bg=fcolor,
                        activeforeground="red",
                        activebackground=bcolor,
                        command=cmd,
                        font=('IBM Plex Sans bold',15))

        Button1.bind("<Enter>",partial(bg_config, Button1, "red", "#262626"))
        Button1.bind("<Leave>",partial(bg_config, Button1, "#262626", "red"))
        Button1.place(x=x,y=y)
    
    btn(0,200,'HOME','grey','#262626',data)
    #btn(0,280,'ANALYSIS','grey','#262626',None)
    btn(0,300,'ABOUTUS','grey','#262626',openabout)
    btn(0,400,'PREDICT','grey','#262626',combobox)


    def hide_panel():
        side_panel.destroy()

    Button(side_panel,text="❌",command=hide_panel,border=0,activebackground='white',fg='red',bg="#262626",font=("Arial bold",15)).place(x=5,y=10)

    return



img2 = ImageTk.PhotoImage(Image.open("C:/Users/sathv/OneDrive/Documents/VSCodes/PyProject/menu_icon.png"))
Button(Labelnote1,command=toggle_side_panel,image=img2,border=0,fg='red',bg="#262626",activebackground='#262626').place(x=5,y=10)


root.mainloop()