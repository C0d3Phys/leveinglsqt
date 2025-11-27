import os
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

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
# Utilidad para escribir PDF sencillo
# --------------------------------------------------
def escribir_pdf(nombre_pdf, texto):
    """
    Genera un PDF básico con el contenido de 'texto' (string multilínea).
    """
    c = canvas.Canvas(nombre_pdf, pagesize=A4)
    width, height = A4
    x_margin = 40
    y = height - 40

    text_obj = c.beginText()
    text_obj.setTextOrigin(x_margin, y)
    text_obj.setFont("Courier", 9)  # monoespaciada para que las columnas cuadren

    for line in texto.splitlines():
        if text_obj.getY() < 40:  # nueva página si se acaba el espacio
            c.drawText(text_obj)
            c.showPage()
            text_obj = c.beginText()
            text_obj.setTextOrigin(x_margin, height - 40)
            text_obj.setFont("Courier", 9)
        text_obj.textLine(line)

    c.drawText(text_obj)
    c.showPage()
    c.save()


# --------------------------------------------------
# Función principal de ajuste
# --------------------------------------------------
def ajustar_nivelacion(ruta_mes, ruta_val, sigma0_apriori=1.0):
    """
    Ajusta la red de nivelación y genera:
      - reporte .txt
      - reporte .pdf

    sigma0_apriori: σ0 a priori (por defecto = 1.0) para interpretación tipo χ² global.
    """
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

        # Peso: p = 1 / σ² (forma clásica)
        # P[i, i] = 1.0 / (dsv ** 2)
        P[i, i] = 1.0 / (dsv)

    # 4) Ecuaciones normales y solución
    try:
        N = A.T @ P @ A
        n_vec = A.T @ P @ L
        X_hat = np.linalg.solve(N, n_vec)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"No se pudo resolver las ecuaciones normales (N singular o mal condicionada): {e}"
        )

    # CONVENCIÓN DEL LIBRO: V = A X - L
    V_hat = A @ X_hat - L

    f = m - n
    if f <= 0:
        raise ValueError(
            f"Grados de libertad no positivos: f = {f}. Revisa número de observaciones/incógnitas."
        )

    vTPv = (V_hat.T @ P @ V_hat).item()
    sigma0_sq = vTPv / f
    sigma0 = sigma0_sq ** 0.5

    # "Chi cuadrado" observado con σ0_apriori
    chi2_obs = vTPv / (sigma0_apriori ** 2)
    chi2_reducido = sigma0_sq / (sigma0_apriori ** 2)

    # 5) Precisión de las incógnitas
    try:
        Qxx = np.linalg.inv(N)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"No se pudo invertir N para obtener Qxx: {e}"
        )

    Cxx = sigma0_sq * Qxx
    sigmas = np.sqrt(np.diag(Cxx))

    # 6) Residuos tipificados
    p_diag = np.diag(P)
    std_residuals = V_hat.flatten() * np.sqrt(p_diag) / sigma0

    # 7) Cotas ajustadas de TODOS los puntos (bancos + incógnitas)
    H_ajustadas = H_known.copy()
    for name, val in zip(unknowns, X_hat.flatten()):
        H_ajustadas[name] = float(val)

    # 8) Construir tabla de observaciones tipo libro
    # ΔElevación ajustada = Δmedida + v   (con v según convención del libro)
    tabla_obs = []
    for i, ((frm, to, dH, dsv), v) in enumerate(zip(obs, V_hat.flatten()), start=1):
        dh_aj = dH + v
        tabla_obs.append((i, frm, to, dH, v, dh_aj, dsv))

    # 9) Construir reporte en texto
    base = os.path.splitext(os.path.basename(ruta_mes))[0]
    nombre_txt = f"{base}_reporte.txt"
    nombre_pdf = f"{base}_reporte.pdf"

    lines = []
    lines.append("REPORTE DE AJUSTE DE NIVELACIÓN")
    lines.append("=" * 60)
    lines.append(f"Archivo de observaciones : {ruta_mes}")
    lines.append(f"Archivo de cotas conocidas: {ruta_val}")
    lines.append("")
    lines.append(f"Nº observaciones (m) : {m}")
    lines.append(f"Nº incógnitas (n)   : {n}")
    lines.append(f"Grados de libertad f: {f}")
    lines.append("")
    lines.append(f"vᵀ P v           = {vTPv:.6f}")
    lines.append(f"sigma0² (a post) = {sigma0_sq:.9f}")
    lines.append(f"sigma0  (a post) = {sigma0:.6f} m")
    lines.append("")
    lines.append(f"sigma0 a priori          = {sigma0_apriori:.6f}")
    lines.append(f"chi² observado           = {chi2_obs:.6f}")
    lines.append(f"chi² reducido (≈sigma0²) = {chi2_reducido:.6f}")
    lines.append("")

    # --- Tabla estilo libro ---
    lines.append("TABLA DE MEDICIONES, RESIDUOS Y VALORES AJUSTADOS")
    lines.append("-" * 60)
    lines.append("i  Desde  Hasta   ΔElev_med   v        ΔElev_aj   σ")
    for i, frm, to, dH, v, dh_aj, dsv in tabla_obs:
        lines.append(
            f"{i:2d}  {frm:6s} {to:6s} "
            f"{dH:10.3f} {v:8.3f} {dh_aj:10.3f} ±{dsv:5.3f}"
        )
    lines.append("")

    # --- Cotas ajustadas de todos los puntos ---
    lines.append("COTAS AJUSTADAS (TODOS LOS PUNTOS)")
    lines.append("-" * 60)
    for name in sorted(H_ajustadas.keys()):
        lines.append(f"{name:8s}  {H_ajustadas[name]:12.3f} m")
    lines.append("")

    # --- Cotas ajustadas solo de incógnitas + precisión ---
    lines.append("COTAS AJUSTADAS DE PUNTOS DESCONOCIDOS")
    lines.append("-" * 60)
    lines.append("Punto      Cota (m)        ±σ (m)")
    for name, val, s in zip(unknowns, X_hat.flatten(), sigmas):
        lines.append(f"{name:8s}  {val:12.6f}   ± {s:8.4f}")
    lines.append("")

    # --- Residuos tipificados ---
    lines.append("RESIDUOS TIPIFICADOS")
    lines.append("-" * 60)
    lines.append("i   v_i (m)       r_i = v_i * sqrt(p_i) / sigma0")
    for i, (v, r) in enumerate(zip(V_hat.flatten(), std_residuals), start=1):
        lines.append(f"{i:2d}  {v:10.6f}      {r:10.3f}")
    lines.append("")

    # --- Matrices Qxx y Cxx ---
    lines.append("MATRIZ Qxx (cofactores de incógnitas)")
    lines.append(str(Qxx))
    lines.append("")
    lines.append("MATRIZ Cxx (covarianza de incógnitas)")
    lines.append(str(Cxx))
    lines.append("")

    reporte_texto = "\n".join(lines)

    # 10) Guardar TXT
    with open(nombre_txt, "w", encoding="utf-8") as f_out:
        f_out.write(reporte_texto)

    # 11) Guardar PDF
    escribir_pdf(nombre_pdf, reporte_texto)

    print(f"\nReporte TXT generado: {nombre_txt}")
    print(f"Reporte PDF generado: {nombre_pdf}")

    # Devuelvo resultados para uso posterior si quieres inspeccionar en código
    return {
        "unknowns": unknowns,
        "X_hat": X_hat,
        "V_hat": V_hat,
        "sigma0_sq": sigma0_sq,
        "sigma0": sigma0,
        "Qxx": Qxx,
        "Cxx": Cxx,
        "chi2_obs": chi2_obs,
        "chi2_reducido": chi2_reducido,
        "std_residuals": std_residuals,
        "H_ajustadas": H_ajustadas,
        "tabla_obs": tabla_obs,
    }


# --------------------------------------------------
# Ejecución de ejemplo
# --------------------------------------------------
if __name__ == "__main__":
    try:
        resultados = ajustar_nivelacion("nivelacion.mes", "nivelacion.val",
                                        sigma0_apriori=1.0)
    except FileNotFoundError as e:
        print(f"ERROR: no se encontró el archivo -> {e.filename}")
    except ValueError as e:
        print(f"ERROR de datos: {e}")
    except np.linalg.LinAlgError as e:
        print(f"ERROR numérico en el ajuste: {e}")
    except Exception as e:
        print(f"ERROR inesperado: {type(e).__name__}: {e}")
