import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from MLP_nn import train_nn, test_nn
from bnb import train_bnb_model, test_bnb_model
from gnb import train_gnb_model, test_gnb_model
from predict_lifespan import predict_poor_quality_age

# Color scheme
COLORS = {
    'primary': '#2E3B4E',      # Dark blue-gray
    'secondary': '#4A90E2',    # Bright blue
    'accent': '#50C878',       # Emerald green
    'background': '#F5F6F8',   # Light gray
    'text': '#2C3E50',         # Dark gray
    'error': '#E74C3C'         # Red
}

# Initialize GUI with modern theme
root = tk.Tk()
root.title("Road Surface Quality Analysis")
root.geometry("1000x700")  # Larger window
root.configure(bg=COLORS['background'])

# Custom styles
style = ttk.Style()
style.theme_use('clam')

# Configure styles
style.configure('TNotebook', background=COLORS['background'])
style.configure('TNotebook.Tab', padding=[15, 5], background=COLORS['primary'], foreground='white')
style.map('TNotebook.Tab', 
    background=[('selected', COLORS['secondary'])],
    foreground=[('selected', 'white')])

style.configure('TFrame', background=COLORS['background'])
style.configure('TButton', 
    padding=[20, 10],
    background=COLORS['secondary'],
    foreground='white',
    font=('Arial', 10, 'bold'))
style.map('TButton',
    background=[('active', COLORS['accent'])],
    foreground=[('active', 'white')])

style.configure('TLabel', 
    background=COLORS['background'],
    foreground=COLORS['text'],
    font=('Arial', 10))
style.configure('Header.TLabel',
    font=('Arial', 14, 'bold'),
    foreground=COLORS['primary'])
style.configure('Helper.TLabel',
    font=('Arial', 9),
    foreground='gray')

# Configure Text widget style
text_config = {
    'bg': 'white',
    'fg': COLORS['text'],
    'font': ('Consolas', 10),
    'padx': 10,
    'pady': 10,
    'relief': 'flat',
    'borderwidth': 0
}

# Notebook configuration
tabs = ttk.Notebook(root)
tab_train = ttk.Frame(tabs, padding=20)
tab_test = ttk.Frame(tabs, padding=20)
tab_custom = ttk.Frame(tabs, padding=20)
tabs.add(tab_train, text=" Train Model ")
tabs.add(tab_test, text=" Test Model ")
tabs.add(tab_custom, text=" Predict Road Age ")
tabs.pack(expand=1, fill="both", padx=20, pady=10)

# Model Selection
models = {"MLP Neural Network": train_nn, "BernoulliNB": train_bnb_model, "GaussianNB": train_gnb_model}
model_var = tk.StringVar(value="MLP Neural Network")

# Train Model Tab
ttk.Label(tab_train, text="Train Your Model", style='Header.TLabel').pack(pady=20)
frame_train = ttk.Frame(tab_train)
frame_train.pack(fill='x', padx=30)

model_frame = ttk.Frame(frame_train)
model_frame.pack(fill='x', pady=10)
ttk.Label(model_frame, text="Select Model:").pack(side='left', padx=5)
model_menu = ttk.Combobox(model_frame, textvariable=model_var, values=list(models.keys()), 
                         state='readonly', width=40)
model_menu.pack(side='left', padx=10)

train_btn = ttk.Button(frame_train, text="Train Model", command=lambda: train_model())
train_btn.pack(pady=20)

# Training output with scrollbar and styled text
train_frame = ttk.Frame(tab_train)
train_frame.pack(fill='both', expand=True, padx=5, pady=5)

scrollbar = ttk.Scrollbar(train_frame)
scrollbar.pack(side='right', fill='y')

train_output = tk.Text(train_frame, **text_config, height=15, width=80, yscrollcommand=scrollbar.set)
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
ttk.Label(tab_custom, text="Predict Road Lifespan", style='Header.TLabel').pack(pady=10)
ttk.Label(tab_custom, text="Enter road parameters to predict its lifespan", style='Helper.TLabel').pack()

frame_custom = ttk.Frame(tab_custom)
frame_custom.pack(fill='x', padx=20)

# Create a frame for input fields
input_fields_frame = ttk.Frame(frame_custom)
input_fields_frame.pack(fill='x', pady=10)

# Configure grid columns
input_fields_frame.columnconfigure(0, weight=1)
input_fields_frame.columnconfigure(1, weight=1)
input_fields_frame.columnconfigure(2, weight=1)

# Input field names and their default values
input_fields = {
    "Number of Machines": "1",
    "Workers on Site": "8",
    "Street Lights Count": "544",
    "Accident History": "41",
    "Damaged Equipment": "1",
    "Road Length (m)": "1158",
    "Curvature Degree": "235.46",
    "High Priority Bends": "44",
    "Road Category (0-2)": "2",
    "Average Speed (km/h)": "28",
    "Road Width (m)": "3.47"
}

# Create entry widgets dictionary to store references
entries = {}

# Create input fields in a grid layout
row = 0
col = 0
for field, default_value in input_fields.items():
    # Create container frame for each field
    field_frame = ttk.Frame(input_fields_frame)
    field_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
    
    # Label
    ttk.Label(field_frame, text=field).pack(anchor='w')
    
    # Entry with custom style
    entry = ttk.Entry(field_frame, style='Custom.TEntry', width=20)
    entry.insert(0, default_value)
    entry.pack(fill='x', pady=2)
    
    # Store entry reference
    entries[field] = entry
    
    # Update grid position
    col += 1
    if col > 2:  # 3 columns
        col = 0
        row += 1

# Analyze button
test_custom_btn = ttk.Button(
    frame_custom, 
    text="Predict Road Lifespan", 
    command=lambda: test_custom_with_fields()
)
test_custom_btn.pack(pady=20)

# Custom output with scrollbar
custom_frame = ttk.Frame(tab_custom)
custom_frame.pack(fill='both', expand=True, padx=5, pady=5)

custom_scrollbar = ttk.Scrollbar(custom_frame)
custom_scrollbar.pack(side='right', fill='y')

custom_output = tk.Text(custom_frame, **text_config, height=12, width=70, yscrollcommand=custom_scrollbar.set)
custom_output.pack(side='left', fill='both', expand=True)
custom_scrollbar.config(command=custom_output.yview)

# Add borders to Text widgets
def add_border_to_text_widgets():
    border_color = COLORS['secondary']
    for widget in [train_output, test_output, custom_output]:
        widget.configure(
            highlightthickness=1, 
            highlightcolor=border_color,
            highlightbackground=border_color
        )

add_border_to_text_widgets()

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
def test_custom_with_fields():
    custom_output.delete(1.0, tk.END)
    try:
        # Get values from all entry fields with validation
        user_input = []
        invalid_fields = []
        
        for field, entry in entries.items():
            try:
                value = entry.get().strip()
                if not value:
                    invalid_fields.append(f"{field} (empty)")
                else:
                    user_input.append(float(value))
            except ValueError:
                invalid_fields.append(f"{field} (invalid number: {entry.get()})")
        
        if invalid_fields:
            error_msg = "Please correct the following fields:\n"
            error_msg += "\n".join(f"• {field}" for field in invalid_fields)
            custom_output.insert(tk.END, error_msg)
            return
        
        # Predict road lifespan
        poor_quality_age = predict_poor_quality_age(user_input)
        
        if poor_quality_age == -1:
            custom_output.insert(tk.END, 
                "Road condition remains acceptable beyond 30 years.\n"
                "Consider re-evaluating the input parameters.")
        else:
            custom_output.insert(tk.END, 
                f"Road Lifespan Analysis:\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"• Road will maintain good/average quality for: {poor_quality_age - 1} years\n"
                f"• Road quality will become poor at: {poor_quality_age} years\n"
                f"• Recommended maintenance before: {poor_quality_age - 1} years\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Note: Predictions are based on current road conditions and usage patterns.")

    except Exception as e:
        custom_output.insert(tk.END, f"Error in prediction: {str(e)}")

# Replace the old test_custom function with the new one
test_custom = test_custom_with_fields

# Run GUI
root.mainloop()