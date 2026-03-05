import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def generar_caso_cluster_purity() -> tuple[dict, float]:
    """
    Genera un par (input, output) aleatorio para cluster_purity_score().

    Salidas:
        input  : dict con los argumentos de la función.
        output : float con el índice de pureza esperado.
    """
    rng = np.random.default_rng()

    # Parámetros aleatorios del caso
    n_classes     = int(rng.integers(2, 5))
    n_clusters    = int(rng.integers(n_classes, n_classes + 3))
    n_features    = int(rng.integers(2, 5))
    n_samples     = int(rng.integers(80, 250))
    random_state  = int(rng.integers(0, 500))
    spread        = float(rng.uniform(0.8, 2.5))

    # Generar datos con clases separables
    rng2 = np.random.default_rng(random_state)
    centers = rng2.uniform(-5, 5, size=(n_classes, n_features))
    samples_per_class = np.full(n_classes, n_samples // n_classes)
    samples_per_class[-1] += n_samples - samples_per_class.sum()

    X_parts, y_parts = [], []
    for cls_idx, n in enumerate(samples_per_class):
        X_parts.append(
            rng2.normal(loc=centers[cls_idx], scale=spread,
                        size=(int(n), n_features))
        )
        y_parts.append(np.full(int(n), cls_idx))

    X_raw = np.vstack(X_parts)
    y_raw = np.concatenate(y_parts)

    # Construir DataFrame
    feature_names = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X_raw, columns=feature_names)
    label_col = "categoria"
    df[label_col] = y_raw.astype(int)

    # Calcular output esperado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)

    total = len(y_raw)
    purity_sum = 0
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        counts = np.bincount(y_raw[mask].astype(int))
        purity_sum += np.max(counts)

    output = round(float(purity_sum / total), 4)

    input_dict = {
        "df":           df,
        "label_col":    label_col,
        "n_clusters":   n_clusters,
        "random_state": random_state,
    }

    return input_dict, output


# --- Ejemplo de uso ---
if __name__ == "__main__":
    inp, out = generar_caso_cluster_purity()
    print("=== INPUT ===")
    print(f"  df.shape     : {inp['df'].shape}")
    print(f"  label_col    : '{inp['label_col']}'")
    print(f"  clases únicas: {sorted(inp['df'][inp['label_col']].unique())}")
    print(f"  n_clusters   : {inp['n_clusters']}")
    print(f"  random_state : {inp['random_state']}")
    print("\n=== OUTPUT ===")
    print(f"  cluster_purity_score = {out}")
