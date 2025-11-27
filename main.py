import numpy as np

# --------------------------------------------------
# Funciones de lectura
# --------------------------------------------------
def leer_mediciones(ruta_mes):
    """
    Lee archivo .mes con líneas:
    desde  hasta  dH_obs  distancia_km
    Ignora líneas vacías y comentarios (#).
    Devuelve lista de tuplas: (desde, hasta, dH_obs, dist_km)
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
            dH = float(parts[2])
            dist = float(parts[3])
            obs.append((frm, to, dH, dist))
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
            H = float(parts[1])
            H_known[name] = H
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
    idx = {p: i for i, p in enumerate(unknowns)}

    m = len(obs)
    n = len(unknowns)

    A = np.zeros((m, n))
    L = np.zeros((m, 1))
    P = np.zeros((m, m))

    # 3) Construir A, L y P
    for i, (frm, to, dH, dsv) in enumerate(obs):
        dh = dH  # copia local

        # Punto "hasta"
        if to in idx:                 # desconocido -> coef +1 en A
            A[i, idx[to]] += 1
        else:                         # conocido -> pasa a L con signo (-)
            dh -= H_known[to]

        # Punto "desde"
        if frm in idx:                # desconocido -> coef -1 en A
            A[i, idx[frm]] -= 1
        else:                         # conocido -> pasa a L con signo (+)
            dh += H_known[frm]

        # Término derecho de la ecuación
        L[i, 0] = dh

        # Peso: aquí uso p = 1/desviacion_estandar (puedes cambiarlo)
        P[i, i] = 1.0 / dsv

    # 4) Ecuaciones normales y solución
    N = A.T @ P @ A
    n_vec = A.T @ P @ L

    X_hat = np.linalg.solve(N, n_vec)
    V_hat = L - A @ X_hat
    f = m - n

    sigma0_sq = (V_hat.T @ P @ V_hat).item() / f
    sigma0 = sigma0_sq ** 0.5

    # 5) Precisión de las incógnitas
    Qxx = np.linalg.inv(N)
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

    # Devuelvo por si quieres usar los resultados en otros módulos
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
    # Cambia por los nombres de tus archivos
    resultados = ajustar_nivelacion("nivelacion.mes", "nivelacion.val")
