# Digit Recognizer con MLP e Interfaccia Grafica

Questo progetto è un'applicazione in Python che riconosce **numeri scritti a mano** usando una rete neurale MLP (Multi-Layer Perceptron) addestrata sul dataset **MNIST**. L'interfaccia grafica è realizzata con `tkinter` e consente all'utente di disegnare un numero con il mouse per poi farlo classificare dal modello in tempo reale.

## Funzionalità

- Allenamento o caricamento di un modello già pronto (`joblib`)
- Interfaccia semplice e intuitiva per disegnare numeri
- Riconoscimento in tempo reale con MLPClassifier (Scikit-Learn)
- Preprocessing automatico dell'immagine disegnata

## Tecnologie usate

- Python 3
- Scikit-learn
- PIL (Pillow)
- Tkinter
- Numpy / Pandas
- MNIST Dataset

## Come eseguirlo

1. Clona la repository:
   bash:
   git clone https://github.com/tuo_nome/digit-recognizer.git
   cd digit-recognizer
   
2. (Facoltativo) Crea un ambiente virtuale:
   python -m venv venv
   source venv/bin/activate  # oppure venv\Scripts\activate su Windows
   
4. Installa i pacchetti necessari:
   pip install -r requirements.txt
   
6. Esegui il programma:
   python digit_recognizer.py
