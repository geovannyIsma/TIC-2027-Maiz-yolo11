"""Paquetes para Zenodo (zenodo/): Data_corr_v6, Data_final, fondos y los modelos finales de la 2.2, con su verificación.
Uso: .venv/bin/python zenodo.py"""
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEST = REPO / 'zenodo'
DEST.mkdir(exist_ok=True)
COMPRIMIDOS = {'.jpg', '.jpeg', '.png', '.pt', '.zip', '.heic'}
cfg = json.loads((REPO / 'oe2' / 'resultados' / 'configuracion_final.json').read_text())

def archivos(carpeta):
    return sorted(p for p in carpeta.rglob('*') if p.is_file() and not p.name.endswith('.cache'))

paquetes = {
    'Data_corr_v6.zip': [(p, p.relative_to(REPO / 'datos')) for p in archivos(REPO / 'datos' / 'Data_corr_v6')],
    'Data_final.zip': [(p, p.relative_to(REPO / 'datos')) for p in archivos(REPO / 'datos' / 'Data_final')],
    'fondos.zip': [(p, p.relative_to(REPO / 'datos')) for p in archivos(REPO / 'datos' / 'fondos')],
}
modelos = [(REPO / 'oe2' / 'resultados' / 'configuracion_final.json', Path('oe2/resultados/configuracion_final.json'))]
for m in cfg.values():
    for pesos in m['pesos']:
        corrida = (REPO / pesos).parent.parent
        for f in ('weights/best.pt', 'args.yaml', 'results.csv'):
            modelos.append((corrida / f, (corrida / f).relative_to(REPO)))
paquetes['modelos.zip'] = modelos

for nombre, lista in paquetes.items():
    z = DEST / nombre
    with zipfile.ZipFile(z, 'w') as zf:
        for origen, destino in lista:
            zf.write(origen, str(destino), compress_type=zipfile.ZIP_STORED if origen.suffix.lower() in COMPRIMIDOS else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(z) as zf:
        malo = zf.testzip()
        n = len(zf.namelist())
    assert malo is None and n == len(lista), (nombre, malo, n, len(lista))
    print(f'{nombre}: {n} archivos, {z.stat().st_size / 1e6:.0f} MB, CRC OK')
