from pathlib import Path
from tkinter.messagebox import NO

from detectron2.data.catalog import Metadata
from uo.utils import load


def load_metadata(meta: Metadata) -> dict:
    path = Path(meta.json_file)
    if not path.exists():
        path = Path("~").expanduser() / path
    return load(str(path))


def load_image_info(meta: Metadata) -> dict[int, dict]:
    data = load_metadata(meta)
    license_names = {lic["id"]: lic["name"] for lic in data["licenses"]}
    license_urls = {lic["id"]: lic["url"] for lic in data["licenses"]}

    images = {
        img["id"]: {
            **img,
            "license": license_names[img["license"]],
            "license_url": license_urls[img["license"]],
        }
        for img in data["images"]
    }

    return images


def add_license(dset: dict, meta: Metadata) -> None:
    info = load_image_info(meta)
    for d in dset:
        d["license"] = info[d["image_id"]]["license"]
