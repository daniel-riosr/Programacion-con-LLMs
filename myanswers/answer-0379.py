import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def predecir_victoria_naive_bayes(X, y, test_size, random_state):
    """
    Divide los datos, escala con MinMaxScaler, entrena GaussianNB
    y devuelve métricas de evaluación sobre el conjunto de prueba.
    """
    # Separar entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Escalado solo con entrenamiento
    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Entrenar el modelo
    model = GaussianNB()
    model.fit(X_train_sc, y_train)
    
    # Predicción
    y_pred = model.predict(X_test_sc)
    
    # Métricas
    resultados = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0))
    }
    
    return resultados