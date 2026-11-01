"""Grad-CAM para los YOLO11 del proyecto: detección, mapas y pruebas exploratorias.

El objetivo es la puntuación de clase de una candidata pre-NMS, no el conteo ni
la regresión de su caja. Las pruebas conservan área, referencia uniforme y
controles aleatorios; no escriben archivos ni modifican los pesos.
"""
import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'oe2'))
import comun_oe2 as E  # noqa: E402

CAPAS = {'P3': 16, 'P4': 19, 'P5': 22, 'B3': 4}   # cuello (entrada de Detect, pasos 8, 16 y 32) y backbone a paso 8
PASOS = {'P3': 8, 'P4': 16, 'P5': 32}


class Salida(torch.nn.Module):
    """YOLO sin NMS: (B, 4 + nc, N) con cajas xywh en píxeles del tensor y logits de clase (antes de la sigmoide)."""

    def __init__(self, modelo):
        super().__init__()
        self.modelo = modelo

    def forward(self, x):
        y, preds = self.modelo(x)
        return torch.cat([y[:, :4], preds['scores']], 1)


class LogitClase:
    """Objetivo por candidata: logit, probabilidad sigmoide o margen entre logits.

    El margen resta el mayor logit de las otras clases; no es una probabilidad.
    Con varios índices se suman los objetivos, sin filtrar por clase predicha.
    """

    def __init__(self, indices, clase, objetivo='logit'):
        if objetivo not in ('logit', 'probabilidad', 'margen'):
            raise ValueError(f'Objetivo desconocido: {objetivo}')
        self.indices, self.clase = list(np.atleast_1d(indices)), int(clase)
        if not self.indices:
            raise ValueError('Se necesita al menos una candidata para el objetivo.')
        self.objetivo = objetivo

    def __call__(self, salida):   # salida: (4 + nc, N) de una imagen
        logits = salida[4:, self.indices]
        elegido = logits[self.clase]
        if self.objetivo == 'probabilidad':
            elegido = elegido.sigmoid()
        elif self.objetivo == 'margen':
            otras = torch.arange(logits.shape[0], device=logits.device) != self.clase
            if not otras.any():
                raise ValueError('El margen requiere al menos dos clases.')
            elegido = elegido - logits[otras].amax(0)
        return elegido.sum()


class Detector:
    """Un modelo entrenado: detección con el mismo preprocesado y NMS que `predict`, y mapas de activación."""

    def __init__(self, pesos, conf, imgsz=E.IMGSZ, iou=0.7):
        self.pesos = Path(pesos).resolve()
        self.dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
        modelo = copy.deepcopy(YOLO(str(pesos)).model).float().eval().to(self.dispositivo)
        modelo.requires_grad_(True)   # los pesos se cargan congelados: sin esto no hay gradientes en las capas
        self.red = Salida(modelo).eval()
        self.capas = {n: modelo.model[i] for n, i in CAPAS.items()}
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.stride = int(modelo.stride.max())

    def preparar(self, img_bgr, preprocesada=False):
        """Tensor de entrada, imagen RGB preprocesada y transformación (escala, dx, dy) de coordenadas originales.
        Una imagen nueva pasa por el preprocesamiento del dataset (`comun.preprocesar`: lado mayor a imgsz,
        normalización fotométrica y relleno); una de `Data_final` ya está preprocesada y se usa tal cual."""
        if preprocesada:
            pre, t = img_bgr, (1.0, 0, 0)
        else:
            pre, r, (dx, dy) = E.C.preprocesar(img_bgr, self.imgsz, 'norm')
            t = (r, dx, dy)
        rgb = np.ascontiguousarray(pre[..., ::-1])
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.dispositivo)
        return x, rgb, t

    def detectar(self, x):
        """Detecciones finales: cajas xyxy, clases, confianzas, índice de la candidata y nivel (P3, P4 o P5)."""
        with torch.no_grad():
            out = self.red(x)
        return self.decodificar(out, x.shape[-2:])

    def decodificar(self, out, forma):
        """NMS de una salida ya calculada; conserva los índices pre-NMS."""
        p = torch.cat([out[:, :4], out[:, 4:].sigmoid()], 1)
        det, idx = non_max_suppression(p, self.conf, self.iou, agnostic=True, max_det=300, return_idxs=True)
        det, idx = det[0].cpu().numpy(), idx[0].cpu().numpy().astype(int)
        H, W = forma
        limites = np.cumsum([(H // s) * (W // s) for s in PASOS.values()])
        nivel = np.array(list(PASOS))[np.searchsorted(limites, idx, side='right')]
        return {'cajas': det[:, :4], 'clases': det[:, 5].astype(int), 'conf': det[:, 4], 'indices': idx, 'nivel': nivel}

    def mapa(self, x, capas=('P3',), indices=None, clase=None, objetivo='logit'):
        """Grad-CAM de una clase en candidatas concretas, normalizado a [0, 1].

        Varios índices suman sus puntuaciones; varias capas promedian sus mapas
        normalizados. Para explicar una detección se usa un solo índice y la
        capa P3/P4/P5 que la produjo, o B3 para comparar niveles de representación.
        """
        if indices is None or clase is None:
            raise ValueError('Grad-CAM requiere candidata y clase explícitas.')
        targets = [LogitClase(indices, clase, objetivo)]
        with GradCAM(self.red, [self.capas[c] for c in np.atleast_1d(capas)]) as cam:
            return cam(x, targets=targets)[0]


def a_letterbox(cajas, t):
    r, dx, dy = t
    return cajas * r + [dx, dy, dx, dy]


def cargar_foto(stem):
    """Foto propia (BGR), cajas de referencia xyxy y clases."""
    img = cv2.imread(str(E.FOTOS / 'propias' / f'{stem}.jpeg'))
    H, W = img.shape[:2]
    c, cl = E.C.leer_yolo(E.FOTOS / 'propias' / 'referencia' / f'{stem}.txt')
    c = np.array(c, float).reshape(-1, 4)
    return img, np.stack([(c[:, 0] - c[:, 2] / 2) * W, (c[:, 1] - c[:, 3] / 2) * H,
                          (c[:, 0] + c[:, 2] / 2) * W, (c[:, 1] + c[:, 3] / 2) * H], 1), np.array(cl)


def borrar(img, cajas, margen=6):
    """Borra los granos de `cajas` por inpainting (misma prueba de contexto que en 2.2c)."""
    m = np.zeros(img.shape[:2], np.uint8)
    for x1, y1, x2, y2 in cajas.astype(int):
        m[max(y1 - margen, 0):y2 + margen, max(x1 - margen, 0):x2 + margen] = 255
    return cv2.inpaint(img, m, 7, cv2.INPAINT_TELEA)


def mascara(forma, cajas):
    m = np.zeros(forma, bool)
    for x1, y1, x2, y2 in np.round(np.asarray(cajas).reshape(-1, 4)).astype(int):
        h, w = forma
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        m[y1:y2, x1:x2] = True
    return m


def energia(mapa, cajas, clases, g, dominio=None):
    """Reparto del mapa del grano g: su caja, cajas de otros granos de su clase, cajas de la otra clase y resto (fondo).
    `cajas` y `clases` son las de referencia (las de granos borrados se conservan: indican la zona que ocupaban)."""
    mapa = np.asarray(mapa, dtype=np.float64)
    if mapa.ndim != 2 or not np.isfinite(mapa).all() or (mapa < 0).any():
        raise ValueError('El mapa debe ser 2D, finito y no negativo.')
    if dominio is not None:
        mapa = np.where(dominio, mapa, 0)
    total = mapa.sum(dtype=np.float64)
    if total <= 0:
        return dict.fromkeys(('propia', 'misma clase', 'otra clase', 'fondo'), np.nan)
    otros = np.arange(len(cajas)) != g
    mp = mascara(mapa.shape, cajas[g])
    mm = mascara(mapa.shape, cajas[otros & (clases == clases[g])]) & ~mp
    mo = mascara(mapa.shape, cajas[otros & (clases != clases[g])]) & ~mp & ~mm
    mf = ~(mp | mm | mo)
    f = [float(mapa[m].sum(dtype=np.float64) / total) for m in (mp, mm, mo, mf)]
    return dict(zip(('propia', 'misma clase', 'otra clase', 'fondo'), f))


def concentracion(mapa, cajas, clases, g, dominio=None):
    """Energía, referencia uniforme y enriquecimiento respecto al área de cada región.

    Es concentración espacial, no probabilidad ni porcentaje de causalidad.
    `dominio` permite excluir el relleno de letterbox de ambas medidas.
    """
    dominio = np.ones(mapa.shape, bool) if dominio is None else np.asarray(dominio, bool)
    if dominio.shape != mapa.shape or not dominio.any():
        raise ValueError('El dominio debe coincidir con el mapa y contener píxeles.')
    observado = energia(mapa, cajas, clases, g, dominio)
    uniforme = energia(dominio.astype(float), cajas, clases, g)
    resultado = {}
    for k in observado:
        resultado[k] = observado[k]
        resultado[f'area_{k}'] = uniforme[k]
        resultado[f'enriquecimiento_{k}'] = observado[k] / uniforme[k] if uniforme[k] > 0 else np.nan
    return resultado


def superponer(rgb, mapa, alfa=0.5):
    """Mapa de calor (jet) sobre la imagen RGB."""
    calor = cv2.applyColorMap(np.uint8(255 * np.clip(mapa, 0, 1)), cv2.COLORMAP_JET)[..., ::-1]
    return np.uint8((1 - alfa) * rgb + alfa * calor)


def iou_cajas(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(hi - lo, 0).prod(2)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    return inter / np.maximum(area_a[:, None] + area_b - inter, 1e-12)


def emparejar(a, b, iou_min=0.5):
    """Máxima cantidad de parejas válidas y después máxima IoU; sin usar la clase."""
    m = iou_cajas(a, b)
    if not m.size:
        return {}, m
    # Una pareja inválida cuesta más que la suma de todos los costes válidos.
    coste = np.where(m >= iou_min, 1 - m, min(m.shape) + 1.)
    r, c = linear_sum_assignment(coste)
    return {int(i): int(j) for i, j in zip(r, c) if m[i, j] >= iou_min}, m


def comparar_detecciones(a, b, iou_min=0.5, atol_caja=1e-3, atol_conf=1e-4):
    """Compara cajas completas, clase y confianza de cada pareja, sin ordenar columnas."""
    pares, matriz = emparejar(a['cajas'], b['cajas'], iou_min)
    diferencias = [np.max(np.abs(a['cajas'][i] - b['cajas'][j])) for i, j in pares.items()]
    confidencias = [abs(float(a['conf'][i]) - float(b['conf'][j])) for i, j in pares.items()]
    clases = all(a['clases'][i] == b['clases'][j] for i, j in pares.items())
    completos = len(pares) == len(a['cajas']) == len(b['cajas'])
    max_caja = float(max(diferencias, default=0))
    max_conf = float(max(confidencias, default=0))
    return {'parejas': len(pares), 'sin_pareja_a': len(a['cajas']) - len(pares),
            'sin_pareja_b': len(b['cajas']) - len(pares), 'mismas_clases_pareadas': clases,
            'min_iou': min((float(matriz[i, j]) for i, j in pares.items()), default=np.nan),
            'max_diferencia_caja_px': max_caja, 'max_diferencia_conf': max_conf,
            'equivalentes': completos and clases and max_caja <= atol_caja and max_conf <= atol_conf}


def inventario(ref, clases, det, iou_min=0.5):
    """Incluye aciertos, errores de clase, falsos negativos y predicciones sobrantes."""
    pares, matriz = emparejar(ref, det['cajas'], iou_min)
    filas = []
    for g, real in enumerate(clases):
        q = pares.get(g)
        estado = 'no detectado' if q is None else ('acierto' if int(det['clases'][q]) == int(real) else 'clase cambiada')
        filas.append({'grano': g, 'prediccion': q, 'estado': estado, 'referencia': int(real),
                      'predicho': None if q is None else int(det['clases'][q]),
                      'conf': None if q is None else float(det['conf'][q]),
                      'iou': None if q is None else float(matriz[g, q]),
                      'cam_disponible': q is not None})
    for q in sorted(set(range(len(det['cajas']))) - set(pares.values())):
        filas.append({'grano': None, 'prediccion': q, 'estado': 'sobrante', 'referencia': None,
                      'predicho': int(det['clases'][q]), 'conf': float(det['conf'][q]),
                      'iou': None, 'cam_disponible': True})
    return filas


def bloques(dominio, tam=32):
    """Unidades de ocultación comunes a CAM y controles; excluyen relleno de letterbox."""
    if tam <= 0 or not np.asarray(dominio).any():
        raise ValueError('Bloque positivo y dominio no vacío requeridos.')
    h, w = dominio.shape
    ids = np.arange(h * w).reshape(h, w)
    return [p for y in range(0, h, tam) for x in range(0, w, tam)
            if len(p := ids[y:y + tam, x:x + tam][dominio[y:y + tam, x:x + tam]])]


def orden_bloques(mapa, unidades, semilla):
    """Media CAM por bloque; desempate aleatorio reproducible (también para mapas planos)."""
    v = np.asarray(mapa).ravel()
    medias = np.array([v[u].mean() for u in unidades])
    desempate = np.random.default_rng(semilla).random(len(unidades))
    return np.lexsort((desempate, -medias))


def mascaras_area(forma, unidades, orden, fracciones):
    """Ranking por bloques con área idéntica entre controles y CAM.

    Se toma una fracción del último bloque si hace falta para igualar exactamente
    el número de píxeles, incluso con bloques de borde de tamaños distintos.
    """
    pixeles = np.concatenate([unidades[i] for i in orden])
    for f in fracciones:
        if not 0 <= f <= 1:
            raise ValueError('Las fracciones deben estar entre 0 y 1.')
        k = min(len(pixeles), int(np.ceil(f * len(pixeles))))
        m = np.zeros(np.prod(forma), bool)
        m[pixeles[:k]] = True
        yield f, k / len(pixeles), m.reshape(forma)


def mezclar(x, base, mascara, modo):
    import torch
    m = torch.as_tensor(mascara, device=x.device)[None, None]
    if modo == 'eliminacion':
        return torch.where(m, base, x)
    if modo == 'insercion':
        return torch.where(m, x, base)
    raise ValueError(modo)


def dominio_imagen(forma, original, transformacion):
    r, dx, dy = transformacion
    dominio = np.zeros(forma, bool)
    dominio[dy:dy + round(original[0] * r), dx:dx + round(original[1] * r)] = True
    return dominio


def desenfoque_base(x, rgb, dominio, sigma):
    """Desenfoca solo el dominio válido; conserva exactamente el relleno original."""
    import torch
    imagen = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma)
    base = torch.from_numpy(np.ascontiguousarray(imagen)).permute(2, 0, 1)[None].to(x.device, x.dtype) / 255
    return torch.where(torch.as_tensor(dominio, device=x.device)[None, None], base, x)


def medir(detector, entrada, indice, clase, caja):
    """Mide score de clase y localización; no confunde índice fijo con instancia reasociada.

    El score reasociado es max(IoU con la caja original * p de la clase objetivo)
    entre candidatas pre-NMS. Es un indicador operativo de este piloto, no la
    reproducción exacta de una métrica publicada ni una evaluación del conteo.
    """
    import torch
    with torch.no_grad():
        salida = detector.red(entrada)
    a = salida[0].detach().cpu().numpy()
    xywh = a[:4].T
    cajas = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    ious = iou_cajas(np.asarray(caja)[None], cajas)[0]
    prob = salida[0, 4 + clase].sigmoid().detach().cpu().numpy()
    similitud = ious * prob
    mejor = int(similitud.argmax())
    return {'logit_fijo': float(a[4 + clase, indice]), 'probabilidad_fija': float(prob[indice]),
            'iou_fija': float(ious[indice]), 'score_fijo': float(similitud[indice]),
            'score_reasociado': float(similitud[mejor]), 'candidata_reasociada': mejor,
            'clase_fija_predicha': int(a[4:, indice].argmax())}


def sha256(path):
    from hashlib import sha256 as resumen_sha256
    h = resumen_sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def perturbaciones(detector, x, rgb, dominio, prediccion, mapa,
                   fracciones=(0., .1, .3, .5, 1.), aleatorios=3,
                   tam_bloque=32, sigma=8., semilla=42):
    """Eliminación/inserción de Grad-CAM y controles aleatorios de idéntica área.

    Devuelve un DataFrame en memoria. Mantiene la candidata/caja/clase originales
    y registra tanto puntuaciones fijas como reasociadas. El desenfoque es una
    perturbación experimental, no una intervención causal validada.
    """
    import pandas as pd
    fracciones = tuple(fracciones)
    if aleatorios < 2 or sigma <= 0:
        raise ValueError('Se requieren al menos dos controles y sigma positivo.')
    if not fracciones or list(fracciones) != sorted(set(fracciones)) or fracciones[0] != 0 or fracciones[-1] != 1:
        raise ValueError('Fracciones únicas, crecientes y con extremos 0 y 1.')
    if mapa.shape != dominio.shape or not np.isfinite(mapa).all() or (mapa < 0).any():
        raise ValueError('Mapa finito y no negativo, con la forma del dominio.')
    unidades = bloques(dominio, tam_bloque)
    base = desenfoque_base(x, rgb, dominio, sigma)
    indice = int(prediccion['indice'])
    clase = int(prediccion['clase'])
    caja = np.asarray(prediccion['caja'])
    ordenes = [('Grad-CAM', -1, orden_bloques(mapa, unidades, semilla))]
    ordenes.extend(('aleatorio', i, np.random.default_rng(semilla + i).permutation(len(unidades)))
                   for i in range(aleatorios))
    resultados, extremos = [], {}
    for metodo, control, orden in ordenes:
        for fraccion, area, mascara_bloques in mascaras_area(dominio.shape, unidades, orden, fracciones):
            for modo in ('eliminacion', 'insercion'):
                clave = ('original' if (fraccion == 0 and modo == 'eliminacion') or (fraccion == 1 and modo == 'insercion')
                         else 'base' if fraccion in (0, 1) else None)
                if clave is not None and clave in extremos:
                    medicion = extremos[clave]
                else:
                    entrada = mezclar(x, base, mascara_bloques, modo)
                    medicion = medir(detector, entrada, indice, clase, caja)
                    if clave is not None:
                        extremos[clave] = medicion
                resultados.append({'metodo': metodo, 'control': control, 'modo': modo,
                                   'fraccion': fraccion, 'area_real': area, **medicion})
    return pd.DataFrame(resultados)
