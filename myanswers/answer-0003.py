import pandas as pd
import numpy as np


def resumen_ventas_por_region(df):
    """
    Genera un resumen de ventas por región.
    """

    # Copia para evitar modificar el DataFrame original
    df = df.copy()

    # 1. Calcular ingreso neto
    df["ingreso_neto"] = (
        df["cantidad"] * df["precio_unitario"] * (1 - df["descuento"])
    )

    # 2. Agrupar por región
    summary = df.groupby("region").agg(
        total_ingresos=("ingreso_neto", "sum"),
        promedio_descuento=("descuento", "mean"),
        num_transacciones=("region", "count"),
    ).reset_index()

    # 3. Calcular porcentaje de ingresos
    total_general = summary["total_ingresos"].sum()

    summary["porcentaje_ingresos"] = np.round(
        summary["total_ingresos"] / total_general * 100,
        2
    )

    # 4. Ordenar y reiniciar índice
    summary = summary.sort_values(
        "total_ingresos",
        ascending=False
    ).reset_index(drop=True)

    return summary