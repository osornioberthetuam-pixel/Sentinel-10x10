#!/usr/bin/env python3
"""Pipeline gratuito Sentinel-2 L2A: GeoTIFF multibanda + PNG + metadatos.
Fuente: Earth Search STAC (Element 84), Sentinel-2 COGs públicos.
No publica en SIGMAR hasta configurar integración autenticada.
"""
import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse
import requests
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pyproj import Transformer
from PIL import Image

STAC = "https://earth-search.aws.element84.com/v1/search"
BANDS = [("B02", "blue"), ("B03", "green"), ("B04", "red"),
         ("B08", "nir"), ("B11", "swir16"), ("B12", "swir22")]

def https_asset(href):
    if href.startswith("s3://"):
        u = urlparse(href)
        return f"https://{u.netloc}.s3.us-west-2.amazonaws.com{u.path}"
    return href

def search(bbox, start, end, max_cloud):
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": 100,
        "query": {"eo:cloud_cover": {"lte": max_cloud}}
    }
    response = requests.post(STAC, json=payload, timeout=50)
    response.raise_for_status()
    features = response.json().get("features", [])
    features.sort(key=lambda f: (
        f["properties"].get("eo:cloud_cover", 100),
        f["properties"].get("datetime", "9999")
    ))
    return features

def crop_band(url, dst_crs, dst_transform, shape):
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
                      AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(url) as src:
            arr = np.zeros(shape, dtype=np.uint16)
            reproject(
                source=rasterio.band(src, 1),
                destination=arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                dst_nodata=0,
            )
            return arr

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument("--start", default="2026-11-01")
    p.add_argument("--end", default="2026-11-26")
    p.add_argument("--max-cloud", type=float, default=20)
    p.add_argument("--output", default="salida_sentinel")
    args = p.parse_args()
    bbox = args.bbox
    if not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        p.error("bbox inválido")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print("Consultando catálogo STAC...", flush=True)
    features = search(bbox, args.start, args.end, args.max_cloud)
    if not features:
        sys.exit("No se encontraron escenas; amplíe las fechas o nubosidad.")
    scene = next((f for f in features if all(k in f.get("assets", {}) for _, k in BANDS)), None)
    if not scene:
        sys.exit("Las escenas no incluyen las seis bandas requeridas.")
    props = scene["properties"]
    assets = scene["assets"]
    crs = "EPSG:32614"  # UTM zona 14N, Lerma
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x1, y1 = tf.transform(bbox[0], bbox[1])
    x2, y2 = tf.transform(bbox[2], bbox[3])
    from rasterio.transform import from_origin
    width = max(1, int(np.ceil((x2 - x1) / 10)))
    height = max(1, int(np.ceil((y2 - y1) / 10)))
    if width * height > 20_000_000:
        sys.exit("AOI demasiado grande para esta prueba.")
    transform = from_origin(x1, y2, 10, 10)
    stack = []
    for name, key in BANDS:
        href = https_asset(assets[key]["href"])
        print("Leyendo", name, flush=True)
        stack.append(crop_band(href, crs, transform, (height, width)))
    tif = out / "sentinel2_L2A_6bandas.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=width, height=height,
                       count=6, dtype="uint16", crs=crs, transform=transform,
                       compress="deflate", tiled=True, nodata=0) as dst:
        for i, ((name, _), band) in enumerate(zip(BANDS, stack), 1):
            dst.write(band, i)
            dst.set_band_description(i, name)
    rgb = []
    for band in [stack[2], stack[1], stack[0]]:
        good = band[band > 0]
        if good.size == 0:
            rgb.append(np.zeros_like(band, dtype=np.uint8))
            continue
        lo, hi = np.percentile(good, [2, 98])
        hi = max(hi, lo + 1)
        rgb.append((np.clip((band - lo) / (hi - lo), 0, 1) * 255).astype("uint8"))
    Image.fromarray(np.stack(rgb, axis=-1)).save(out / "vista_RGB.png")
    meta = {
        "fuente": "Element 84 Earth Search / Sentinel-2 L2A COGs",
        "catalogo": STAC,
        "id_escena": scene["id"],
        "fecha": props.get("datetime"),
        "nubosidad_escena_porcentaje": props.get("eo:cloud_cover"),
        "advertencia": "Nubosidad de escena, no verificada dentro del polígono. No es clasificación.",
        "bbox_epsg4326": bbox,
        "crs_salida": crs,
        "resolucion_salida_m": 10,
        "bandas": {name: {"asset": key, "url": https_asset(assets[key]["href"])}
                   for name, key in BANDS},
        "archivo_multibanda": tif.name,
        "vista": "vista_RGB.png",
        "nota": "B11/B12 remuestreadas desde 20 m; no aumenta resolución física."
    }
    (out / "metadatos.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("TERMINADO:", out.resolve(), flush=True)

if __name__ == "__main__":
    main()
