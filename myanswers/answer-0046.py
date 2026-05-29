import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def modelo_polinomico_optimo(df, target_col, grado_max):

    # Separar variables predictoras y target
    x_data = df.drop(columns=[target_col]).to_numpy()
    y_data = df[target_col].to_numpy()

    # Configuración de validación cruzada
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Métrica RMSE
    rmse_scorer = make_scorer(
        root_mean_squared_error,
        greater_is_better=False
    )

    mejor_grado = None
    mejor_rmse = float("inf")

    # Probar distintos grados polinómicos
    for grado in range(1, grado_max + 1):

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(
                degree=grado,
                include_bias=False
            )),
            ("reg", LinearRegression()),
        ])

        scores = cross_val_score(
            model,
            x_data,
            y_data,
            cv=cv,
            scoring=rmse_scorer
        )

        rmse_prom = float(-scores.mean())

        if rmse_prom < mejor_rmse:
            mejor_rmse = rmse_prom
            mejor_grado = grado

    return (int(mejor_grado), float(mejor_rmse))