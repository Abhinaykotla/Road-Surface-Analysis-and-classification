import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from MLP_nn import train_nn, test_nn
from bnb import train_bnb_model, test_bnb_model
from gnb import train_gnb_model, test_gnb_model
from predict_lifespan import predict_poor_quality_age

# Initialize GUI with a modern theme
root = tk.Tk()
root.title("Road Surface Quality Analysis")
root.geometry("800x600")
root.configure(bg='#f0f0f0')

style = ttk.Style()
style.theme_use('clam')
style.configure('TNotebook', background='#f0f0f0')
style.configure('TFrame', background='#f0f0f0')
style.configure('TButton', padding=6, relief="flat", background="#2196F3")
style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
style.configure('Header.TLabel', font=('Arial', 12, 'bold'))

# Notebook (Tab structure)
tabs = ttk.Notebook(root)
tab_train = ttk.Frame(tabs, padding=20)
tab_test = ttk.Frame(tabs, padding=20)
tab_custom = ttk.Frame(tabs, padding=20)
tabs.add(tab_train, text=" Train Model ")
tabs.add(tab_test, text=" Test Model ")
tabs.add(tab_custom, text=" Custom Input ")
tabs.pack(expand=1, fill="both", padx=10, pady=5)

# Model Selection
models = {"MLP Neural Network": train_nn, "BernoulliNB": train_bnb_model, "GaussianNB": train_gnb_model}
model_var = tk.StringVar(value="MLP Neural Network")

# Train Model Tab
ttk.Label(tab_train, text="Train Your Model", style='Header.TLabel').pack(pady=10)
frame_train = ttk.Frame(tab_train)
frame_train.pack(fill='x', padx=20)

ttk.Label(frame_train, text="Select Model:").pack(pady=5)
model_menu = ttk.Combobox(frame_train, textvariable=model_var, values=list(models.keys()), state='readonly', width=30)
model_menu.pack(pady=5)

train_btn = ttk.Button(frame_train, text="Train Model", command=lambda: train_model())
train_btn.pack(pady=10)

# Training output with scrollbar
train_frame = ttk.Frame(tab_train)
train_frame.pack(fill='both', expand=True, padx=5, pady=5)

scrollbar = ttk.Scrollbar(train_frame)
scrollbar.pack(side='right', fill='y')

train_output = tk.Text(train_frame, height=12, width=70, yscrollcommand=scrollbar.set)
train_output.pack(side='left', fill='both', expand=True)
scrollbar.config(command=train_output.yview)

# Test Model Tab
ttk.Label(tab_test, text="Test Pre-defined Cases", style='Header.TLabel').pack(pady=10)
frame_test = ttk.Frame(tab_test)
frame_test.pack(fill='x', padx=20)

ttk.Label(frame_test, text="Select Model:").pack(pady=5)
test_menu = ttk.Combobox(frame_test, textvariable=model_var, values=list(models.keys()), state='readonly', width=30)
test_menu.pack(pady=5)

test_btn = ttk.Button(frame_test, text="Run Test Cases", command=lambda: test_model())
test_btn.pack(pady=10)

# Test output with scrollbar
test_frame = ttk.Frame(tab_test)
test_frame.pack(fill='both', expand=True, padx=5, pady=5)

test_scrollbar = ttk.Scrollbar(test_frame)
test_scrollbar.pack(side='right', fill='y')

test_output = tk.Text(test_frame, height=12, width=70, yscrollcommand=test_scrollbar.set)
test_output.pack(side='left', fill='both', expand=True)
test_scrollbar.config(command=test_output.yview)

# Custom Input Tab
ttk.Label(tab_custom, text="Test Custom Input Data", style='Header.TLabel').pack(pady=10)
frame_custom = ttk.Frame(tab_custom)
frame_custom.pack(fill='x', padx=20)

ttk.Label(frame_custom, text="Enter values (comma-separated):").pack(pady=5)
custom_entry = ttk.Entry(frame_custom, width=60)
custom_entry.pack(pady=5)

# Add a helper text
helper_text = "Format: IDMachines, PeopleAtwork, StreetLights, Accidents, DamagedMovers, StRoadLength, \nRoadCurvature, HPBends, RoadType, AvgSpeed, RoadWidth, AgeOfRoad"
ttk.Label(frame_custom, text=helper_text, wraplength=500, foreground='gray').pack(pady=5)

test_custom_btn = ttk.Button(frame_custom, text="Analyze Road Quality", command=lambda: test_custom())
test_custom_btn.pack(pady=10)

# Custom output with scrollbar
custom_frame = ttk.Frame(tab_custom)
custom_frame.pack(fill='both', expand=True, padx=5, pady=5)

custom_scrollbar = ttk.Scrollbar(custom_frame)
custom_scrollbar.pack(side='right', fill='y')

custom_output = tk.Text(custom_frame, height=12, width=70, yscrollcommand=custom_scrollbar.set)
custom_output.pack(side='left', fill='both', expand=True)
custom_scrollbar.config(command=custom_output.yview)

# Function to Train Model
def train_model():
    train_output.delete(1.0, tk.END)
    model_name = model_var.get()
    train_func = models[model_name]
    result = train_func()
    if isinstance(result, str):
        train_output.insert(tk.END, result)
    else:
        train_output.insert(tk.END, f"{model_name} trained successfully!")

# Function to Test Model
def test_model():
    test_output.delete(1.0, tk.END)
    model_name = model_var.get()
    test_func = {"MLP Neural Network": test_nn, "BernoulliNB": test_bnb_model, "GaussianNB": test_gnb_model}[model_name]
    
    test_cases = [
        [1, 8, 544, 41, 1, 1158, 235.46, 44, 2, 28, 3.47, 2],
        [7, 17, 819, 13, 9, 6165, 19.62, 0, 0, 95, 0.28, 25],
        [5, 15, 739, 33, 6, 2681, 124.86, 29, 1, 55, 0.72, 5]
    ]
    result = test_func(test_cases)
    test_output.insert(tk.END, f"Predicted Classes: {result}")

# Function to Test Custom Input
def test_custom():
    custom_output.delete(1.0, tk.END)
    try:
        model_name = model_var.get()
        test_func = {"MLP Neural Network": test_nn, "BernoulliNB": test_bnb_model, "GaussianNB": test_gnb_model}[model_name]
        
        user_input = list(map(float, custom_entry.get().split(",")))
        result = test_func([user_input])
        output=""
        if result[0] == "A":
            output="Poor Quality"
        elif result[0] == "B":
            output="avg Quality"
        else:
            output="GOOD Quality"
        poorquality=predict_poor_quality_age(user_input)    
           
        custom_output.insert(tk.END, f"Predicted Class: {output}\nThe road quality turns to 'Poor' at age of: {poorquality - 1} to {poorquality} years\n")

    except Exception as e:
        custom_output.insert(tk.END, f"Error: {e}")

# Run GUI
root.mainloop()