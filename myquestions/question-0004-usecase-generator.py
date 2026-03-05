import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def generar_caso_residuos_estandarizados() -> tuple[dict, np.ndarray]:
    """
    Genera un par (input, output) aleatorio para residuos_estandarizados().

    Salidas:
        input  : dict con los argumentos de la función.
        output : np.ndarray con los residuos estandarizados del conjunto de test.
    """
    rng = np.random.default_rng()

    # Parámetros aleatorios del caso
    n_samples    = int(rng.integers(60, 250))
    n_features   = int(rng.integers(2, 6))
    noise_std    = float(rng.uniform(0.5, 4.0))
    test_size    = float(rng.choice([0.2, 0.25, 0.3]))
    random_state = int(rng.integers(0, 500))

    # Generar datos sintéticos
    rng2 = np.random.default_rng(random_state)
    feature_names = [f"x_{i}" for i in range(n_features)]
    X_raw = rng2.normal(size=(n_samples, n_features))
    coef  = rng2.uniform(-3, 3, size=n_features)
    y_raw = X_raw @ coef + rng2.normal(scale=noise_std, size=n_samples)

    df = pd.DataFrame(X_raw, columns=feature_names)
    target_col = "target"
    df[target_col] = y_raw

    # Calcular output esperado
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    residuos = y_test - y_pred
    output = np.round(
        (residuos - residuos.mean()) / residuos.std(),
        4
    )

    input_dict = {
        "df":           df,
        "target_col":   target_col,
        "test_size":    test_size,
        "random_state": random_state,
    }

    return input_dict, output


# --- Ejemplo de uso ---
if __name__ == "__main__":
    inp, out = generar_caso_residuos_estandarizados()
    print("=== INPUT ===")
    print(f"  df.shape     : {inp['df'].shape}")
    print(f"  target_col   : '{inp['target_col']}'")
    print(f"  test_size    : {inp['test_size']}")
    print(f"  random_state : {inp['random_state']}")
    print("\n=== OUTPUT ===")
    print(f"  shape   : {out.shape}")
    print(f"  primeros 5 residuos estandarizados: {out[:5]}")
    print(f"  |res| > 2: {(np.abs(out) > 2).sum()} muestras sospechosas")
