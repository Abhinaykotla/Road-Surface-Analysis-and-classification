import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from mlp_nn import train_nn, test_nn
from bnb import train_bnb_model, test_bnb_model
from gnb import train_gnb_model, test_gnb_model

# Initialize GUI
root = tk.Tk()
root.title("Road Quality Prediction GUI")
root.geometry("600x500")

# Notebook (Tab structure)
tabs = ttk.Notebook(root)
tab_train = ttk.Frame(tabs)
tab_test = ttk.Frame(tabs)
tab_custom = ttk.Frame(tabs)
tabs.add(tab_train, text="Train Model")
tabs.add(tab_test, text="Test Model")
tabs.add(tab_custom, text="Custom Input")
tabs.pack(expand=1, fill="both")

# Model Selection
models = {"MLP Neural Network": train_nn, "BernoulliNB": train_bnb_model, "GaussianNB": train_gnb_model}
model_var = tk.StringVar(value="MLP Neural Network")

# Train Model Tab
tl_label = tk.Label(tab_train, text="Select Model:")
tl_label.pack()
model_menu = ttk.Combobox(tab_train, textvariable=model_var, values=list(models.keys()))
model_menu.pack()

train_btn = tk.Button(tab_train, text="Train Model", command=lambda: train_model())
train_btn.pack()

train_output = tk.Text(tab_train, height=10, width=60)
train_output.pack()

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

# Test Model Tab
test_label = tk.Label(tab_test, text="Select Model to Test:")
test_label.pack()
test_menu = ttk.Combobox(tab_test, textvariable=model_var, values=list(models.keys()))
test_menu.pack()

test_btn = tk.Button(tab_test, text="Test Model", command=lambda: test_model())
test_btn.pack()

test_output = tk.Text(tab_test, height=10, width=60)
test_output.pack()

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

# Custom Input Tab
custom_input_label = tk.Label(tab_custom, text="Enter values (comma-separated):")
custom_input_label.pack()
custom_entry = tk.Entry(tab_custom, width=60)
custom_entry.pack()

test_custom_btn = tk.Button(tab_custom, text="Test Custom Input", command=lambda: test_custom())
test_custom_btn.pack()

custom_output = tk.Text(tab_custom, height=10, width=60)
custom_output.pack()

# Function to Test Custom Input
def test_custom():
    custom_output.delete(1.0, tk.END)
    try:
        model_name = model_var.get()
        test_func = {"MLP Neural Network": test_nn, "BernoulliNB": test_bnb_model, "GaussianNB": test_gnb_model}[model_name]
        
        user_input = list(map(float, custom_entry.get().split(",")))
        result = test_func([user_input])
        custom_output.insert(tk.END, f"Predicted Class: {result[0]}")
    except Exception as e:
        custom_output.insert(tk.END, f"Error: {e}")

# Run GUI
root.mainloop()