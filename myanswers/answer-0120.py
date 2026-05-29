import pandas as pd
from sklearn.inspection import permutation_importance


def analizar_importancia_permutacion(
    modelo_entrenado,
    X_val,
    y_val
):
    """
    Calcula la importancia por permutación
    de las características de un modelo entrenado.
    """

    # Calcular importancia por permutación
    result = permutation_importance(
        modelo_entrenado,
        X_val,
        y_val,
        n_repeats=5,
        random_state=42,
        scoring='accuracy'
    )

    # Construir DataFrame resultado
    output_df = pd.DataFrame({
        'feature': X_val.columns,
        'importance': result.importances_mean
    })

    # Ordenar descendente
    output_df = output_df.sort_values(
        by='importance',
        ascending=False
    ).reset_index(drop=True)

    return output_df