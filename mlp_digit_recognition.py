import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Modello pre-addestrato
MODEL_FILENAME = "mlp_mnist_model_0.3.pkl"
#L'algoritmo scritto sotto è quello che ha prodotto il modello 0.4, questo 0.3 è uscito meglio, le differenze se non sbaglio sono:
# - la standardizzazione dei dati che nello 0.3 non esisteva;
# - early_stopping=True non c'era in 0.3
# - alpha=0.001 non c'era in 0.3



def load_or_train_model():
    if os.path.exists(MODEL_FILENAME):
        print("Caricamento modello...")
        return joblib.load(MODEL_FILENAME)
    
    print("Caricamento dataset MNIST...")
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardizzazione dei dati
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Applichiamo la standardizzazione ai dati di addestramento
    X_test = scaler.transform(X_test)  # Applichiamo la stessa trasformazione ai dati di test

    # Definiamo una rete neurale con 3 hidden layers e regularizzazione L2 (alpha)
    mlp = MLPClassifier(hidden_layer_sizes=(150, 100, 50), 
                        activation="tanh", 
                        tol=1e-5, 
                        solver="adam", 
                        max_iter=1000, 
                        verbose=True, 
                        random_state=42,
                        early_stopping=True,
                        alpha=0.001)  # Aggiungiamo la regularizzazione L2
                    
    print("Addestramento modello...")
    mlp.fit(X_train, y_train)

    scores = cross_val_score(mlp, X, y, cv=5)
    print(f"Accuratezza media: {scores.mean():.4f}")
    joblib.dump(mlp, MODEL_FILENAME)
    return mlp

class DigitRecognizerApp(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("Digit Recognizer")
        self.geometry("350x400")
        self.configure(bg="#2e2e2e")
        self.model = model
        
        self.canvas_size = 280
        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack(pady=10)
        
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        btn_frame = tk.Frame(self, bg="#2e2e2e")
        btn_frame.pack()
        
        self.predict_btn = ttk.Button(btn_frame, text="Predict", command=self.predict_digit)
        self.predict_btn.grid(row=0, column=0, padx=10)
        
        self.clear_btn = ttk.Button(btn_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.grid(row=0, column=1, padx=10)
        
        self.label = tk.Label(self, text="Disegna un numero", fg="white", bg="#2e2e2e", font=("Arial", 14))
        self.label.pack(pady=10)
    
    def paint(self, event):
        x, y = event.x, event.y
        r = 12  # Tratto più spesso
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="white")
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
    
    def predict_digit(self):
        img = self.image.resize((28, 28)).convert("L")
        img = np.array(img) / 255.0
        img = img.flatten().reshape(1, -1)
        
        prediction = self.model.predict(img)[0]
        self.label.config(text=f"Predizione: {prediction}")

if __name__ == "__main__":
    model = load_or_train_model()
    app = DigitRecognizerApp(model)
    app.mainloop()
