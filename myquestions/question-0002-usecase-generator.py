import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def generar_caso_tasa_error_por_clase() -> tuple[dict, np.ndarray]:
    """
    Genera un par (input, output) aleatorio para tasa_error_por_clase().

    Salidas:
        input  : dict con los argumentos de la función.
        output : np.ndarray con la tasa de error por clase ordenada.
    """
    rng = np.random.default_rng()

    # Parámetros aleatorios del caso
    n_samples     = int(rng.integers(80, 300))
    n_features    = int(rng.integers(2, 6))
    n_classes     = int(rng.integers(2, 5))
    n_neighbors   = int(rng.choice([3, 5, 7]))
    test_size     = float(rng.choice([0.2, 0.25, 0.3]))
    random_state  = int(rng.integers(0, 500))

    # Generar datos sintéticos con clases separables
    rng2 = np.random.default_rng(random_state)
    centers = rng2.uniform(-4, 4, size=(n_classes, n_features))
    samples_per_class = np.full(n_classes, n_samples // n_classes)
    samples_per_class[-1] += n_samples - samples_per_class.sum()

    X_parts, y_parts = [], []
    for cls_idx, n in enumerate(samples_per_class):
        X_parts.append(rng2.normal(loc=centers[cls_idx], scale=1.0,
                                   size=(int(n), n_features)))
        y_parts.append(np.full(int(n), cls_idx))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Calcular output esperado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    clases = np.unique(y)
    tasas = np.array([
        np.mean(y_pred[y_test == c] != c)
        for c in clases
    ])

    input_dict = {
        "X":            X,
        "y":            y,
        "n_neighbors":  n_neighbors,
        "test_size":    test_size,
        "random_state": random_state,
    }

    return input_dict, tasas


# --- Ejemplo de uso ---
if __name__ == "__main__":
    inp, out = generar_caso_tasa_error_por_clase()
    print("=== INPUT ===")
    print(f"  X.shape      : {inp['X'].shape}")
    print(f"  clases únicas: {np.unique(inp['y'])}")
    print(f"  n_neighbors  : {inp['n_neighbors']}")
    print(f"  test_size    : {inp['test_size']}")
    print(f"  random_state : {inp['random_state']}")
    print("\n=== OUTPUT ===")
    print(f"  tasa_error_por_clase: {out.round(4)}")
