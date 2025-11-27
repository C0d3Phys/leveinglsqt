import numpy as np

# --------------------------------------------------
# Funciones de lectura
# --------------------------------------------------
def leer_mediciones(ruta_mes):
    """
    Lee archivo .mes con líneas:
    desde  hasta  dH_obs  dsv
    donde dsv es la desviación estándar (σ) de la observación.
    Ignora líneas vacías y comentarios (#).
    Devuelve lista de tuplas: (desde, hasta, dH_obs, dsv)
    """
    obs = []
    with open(ruta_mes, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Línea inválida en {ruta_mes}: {line}")
            frm, to = parts[0], parts[1]
            try:
                dH = float(parts[2])
                dsv = float(parts[3])
            except ValueError:
                raise ValueError(f"No se pudo convertir a float en {ruta_mes}: {line}")
            obs.append((frm, to, dH, dsv))

    if not obs:
        raise ValueError(f"El archivo {ruta_mes} no contiene mediciones válidas.")
    return obs


def leer_cotas_conocidas(ruta_val):
    """
    Lee archivo .val con líneas:
    punto  cota
    Ignora líneas vacías y comentarios (#).
    Devuelve diccionario: {nombre_punto: cota}
    """
    H_known = {}
    with open(ruta_val, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Línea inválida en {ruta_val}: {line}")
            name = parts[0]
            try:
                H = float(parts[1])
            except ValueError:
                raise ValueError(f"No se pudo convertir a float en {ruta_val}: {line}")
            H_known[name] = H

    if not H_known:
        raise ValueError(f"El archivo {ruta_val} no contiene cotas válidas.")
    return H_known


# --------------------------------------------------
# Función principal de ajuste
# --------------------------------------------------
def ajustar_nivelacion(ruta_mes, ruta_val):
    # 1) Leer datos
    obs = leer_mediciones(ruta_mes)
    H_known = leer_cotas_conocidas(ruta_val)

    # 2) Determinar incógnitas
    points = {p for (frm, to, _, _) in obs for p in (frm, to)}
    unknowns = sorted(points - H_known.keys())
    if not unknowns:
        raise ValueError(
            "No hay incógnitas: todos los puntos del archivo .mes están en el archivo .val."
        )

    idx = {p: i for i, p in enumerate(unknowns)}

    m = len(obs)
    n = len(unknowns)

    if m <= n:
        raise ValueError(
            f"No hay redundancia suficiente: observaciones = {m}, incógnitas = {n}."
        )

    A = np.zeros((m, n))
    L = np.zeros((m, 1))
    P = np.zeros((m, m))

    # 3) Construir A, L y P
    for i, (frm, to, dH, dsv) in enumerate(obs):
        dh = dH  # copia local

        # Validar desviación estándar
        if dsv <= 0:
            raise ValueError(
                f"Desviación estándar no positiva en observación {i+1}: σ = {dsv}"
            )

        # Punto "hasta"
        if to in idx:                 # desconocido -> coef +1 en A
            A[i, idx[to]] += 1
        else:                         # conocido -> pasa a L con signo (-)
            if to not in H_known:
                raise ValueError(f"Punto conocido '{to}' no definido en {ruta_val}.")
            dh -= H_known[to]

        # Punto "desde"
        if frm in idx:                # desconocido -> coef -1 en A
            A[i, idx[frm]] -= 1
        else:                         # conocido -> pasa a L con signo (+)
            if frm not in H_known:
                raise ValueError(f"Punto conocido '{frm}' no definido en {ruta_val}.")
            dh += H_known[frm]

        # Término derecho de la ecuación
        L[i, 0] = dh

        # Peso: p = 1 / σ²  (regla "clásica"); cambia aquí si quieres otra
        P[i, i] = 1.0 / (dsv ** 2)

    # 4) Ecuaciones normales y solución
    try:
        N = A.T @ P @ A
        n_vec = A.T @ P @ L

        X_hat = np.linalg.solve(N, n_vec)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"No se pudo resolver las ecuaciones normales (N singular o mal condicionada): {e}"
        )

    V_hat = L - A @ X_hat
    f = m - n
    if f <= 0:
        raise ValueError(
            f"Grados de libertad no positivos: f = {f}. Revisa número de observaciones/incógnitas."
        )

    sigma0_sq = (V_hat.T @ P @ V_hat).item() / f
    sigma0 = sigma0_sq ** 0.5

    # 5) Precisión de las incógnitas
    try:
        Qxx = np.linalg.inv(N)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"No se pudo invertir N para obtener Qxx: {e}"
        )

    Cxx = sigma0_sq * Qxx
    sigmas = np.sqrt(np.diag(Cxx))

    # 6) Reporte
    print("Incógnitas:", unknowns)
    print("Matriz A =\n", A)
    print("Vector L =\n", L)
    print("Matriz de pesos P =\n", P)

    print("\nCotas ajustadas:")
    for name, val, s in zip(unknowns, X_hat.flatten(), sigmas):
        print(f"  {name} = {val:.6f} m   ± {s:.4f} m")

    print("\nResiduos v_i:")
    for i, v in enumerate(V_hat.flatten(), start=1):
        print(f"  v_{i} = {v:.6f} m")

    print(f"\nVarianza a posteriori σ0² = {sigma0_sq:.9f}")
    print(f"Desv. típica a posteriori σ0 = {sigma0:.6f} m")

    print("\nMatriz Qxx =\n", Qxx)
    print("Matriz Cxx =\n", Cxx)

    # Devuelvo resultados para uso posterior
    return {
        "unknowns": unknowns,
        "X_hat": X_hat,
        "V_hat": V_hat,
        "sigma0_sq": sigma0_sq,
        "sigma0": sigma0,
        "Qxx": Qxx,
        "Cxx": Cxx,
    }


# --------------------------------------------------
# Ejecución de ejemplo
# --------------------------------------------------
if __name__ == "__main__":
    try:
        resultados = ajustar_nivelacion("nivelacion.mes", "nivelacion.val")
    except FileNotFoundError as e:
        print(f"ERROR: no se encontró el archivo -> {e.filename}")
    except ValueError as e:
        print(f"ERROR de datos: {e}")
    except np.linalg.LinAlgError as e:
        print(f"ERROR numérico en el ajuste: {e}")
    except Exception as e:
        # Último catch por si algo inesperado revienta
        print(f"ERROR inesperado: {type(e).__name__}: {e}")
