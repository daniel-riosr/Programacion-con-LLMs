import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def generar_caso_coeficientes_estandarizados() -> tuple[dict, np.ndarray]:
    """
    Genera un par (input, output) aleatorio para coeficientes_estandarizados().

    Salidas:
        input  : dict con los argumentos de la función.
        output : np.ndarray con los coeficientes ordenados por |valor| descendente.
    """
    rng = np.random.default_rng()

    # Parámetros aleatorios del caso
    n_samples  = int(rng.integers(50, 200))
    n_features = int(rng.integers(2, 6))
    noise_std  = float(rng.uniform(0.5, 3.0))
    seed       = int(rng.integers(0, 999))

    # Generar datos sintéticos
    rng2 = np.random.default_rng(seed)
    feature_names = [f"var_{i}" for i in range(n_features)]
    X_raw = rng2.normal(size=(n_samples, n_features))
    coef  = rng2.uniform(-5, 5, size=n_features)
    y_raw = X_raw @ coef + rng2.normal(scale=noise_std, size=n_samples)

    df = pd.DataFrame(X_raw, columns=feature_names)
    target_col = "precio"
    df[target_col] = y_raw

    # Calcular output esperado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    model = LinearRegression().fit(X_scaled, y_raw)
    coefs = model.coef_
    output = coefs[np.argsort(np.abs(coefs))[::-1]]

    input_dict = {
        "df":         df,
        "target_col": target_col,
    }

    return input_dict, output


# --- Ejemplo de uso ---
if __name__ == "__main__":
    inp, out = generar_caso_coeficientes_estandarizados()
    print("=== INPUT ===")
    print(f"  df.shape    : {inp['df'].shape}")
    print(f"  target_col  : '{inp['target_col']}'")
    print(f"  features    : {[c for c in inp['df'].columns if c != inp['target_col']]}")
    print("\n=== OUTPUT ===")
    print(f"  coeficientes ordenados: {out.round(4)}")
