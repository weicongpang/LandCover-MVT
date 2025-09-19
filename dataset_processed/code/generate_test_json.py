#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Dict, List

# Hard-coded input/output paths
IMAGE_DIR = Path("/root/openset/dataset_eval/Test_processed")  
OUTPUT_PATH = Path("/root/openset/llama_factory/LLaMA-Factory/data/test_rm_dataset.jsonl")

TAXONOMY: Dict[str, Dict] = {
    "01": {"name": "Cultivated Land", "subs": {"011": "Paddy field", "012": "Irrigated land", "013": "Dry land"}},
    "02": {"name": "Garden Land", "subs": {"021": "Orchard", "022": "Tea garden", "023": "Other gardens"}},
    "03": {"name": "Forest land", "subs": {"031": "Forest", "032": "Shrubland", "033": "Other forest land"}},
    "04": {"name": "Grassland", "subs": {"041": "Natural grassland", "042": "Artificial grassland", "043": "Other grassland"}},
    "05": {"name": "Commercial Service Land", "subs": {"051": "Retail land", "052": "Accommodation/Catering land", "053": "Business/Financial land", "054": "Other commercial land"}},
    "06": {"name": "Industrial, Mining & Storage land", "subs": {"061": "Industrial land", "062": "Mining land", "063": "Storage land"}},
    "07": {"name": "Residential Land", "subs": {"071":"Urban Residential Land","072":"Rural Homestead Land"}},
    "08": {"name": "Public Administration and Public Service Land", "subs": {"081": "Governmental Land","082": "Press and Publication Land","083": "Scientific and Educational Land","084": "Medical and Charity Land","085": "Cultural Service Land","086": "Public Facilities Land","087": "Park and Green Space","088": "Scenic and Natural Heritage Land"}},
    "09": {"name": "Special Land", "subs": {"091": "Military Facilities Land","092": "Embassies and Consulates Land","093": "Prison Land","094": "Religious Land","095": "Cemetery Land"}},
    "10": {"name": "Transportation Land", "subs": {"101": "Railway Land","102": "Road Land","103": "Street Land","104": "Rural Road Land","105": "Airport Land","106": "Port Land","107": "Pipeline Transportation Land"}},
    "11": {"name": "Water Bodies and Hydraulic Facility Land", "subs": {"111": "River Surface","112": "Lake Surface","113": "Reservoir Surface","114": "Pond Surface","115": "Coastal Tidal Flats","116": "Inland Tidal Flats","117": "Ditches","118": "Hydraulic Construction Land","119": "Glacier and Permanent Snow"}},
    "12": {"name": "Other Land","subs":{"121": "Idle Land","122": "Facility Agricultural Land","123": "Ridge Land","124": "Saline-Alkali Land","125": "Swamp","126": "Sandy Land","127": "Bare Land"}}
}


def build_taxonomy_text() -> str:
    lines: List[str] = []
    for lvl1_code in sorted(TAXONOMY.keys()):
        lvl1 = TAXONOMY[lvl1_code]
        lines.append(f"- {lvl1_code} {lvl1['name']}")
        for sub_code in sorted(lvl1["subs"].keys()):
            sub_name = TAXONOMY[lvl1_code]["subs"][sub_code]
            lines.append(f"  - {sub_code} {sub_name}")
    return "\n".join(lines)


def build_instruction(image_path: str) -> str:
    taxonomy_block = build_taxonomy_text()

    return f"""<image>

    You are a senior remote-sensing image analyst. Given ONE Remote-Sensing image, classify the scene using the two-level taxonomy below.

    Taxonomy:
    {taxonomy_block}

    Your Task:
    1) Determine the Level-1 category (You should only choose one!). Please RETURN CATEGORY NAME.  
    2) Under that Level-1 category, determine exactly ONE Level-2 subclass. Please RETURN CATEGORY NAME.  
    3) Provide descriptions (3-5 sentences) explaining clearly why this classification was made.

    Your response should strictly follow the template below:

    \"The image ({image_path}) is Level-1 category [Name]. Specifically, it is Level-2 subclass [Name]. The reason for this classification is as follows: [Your 3-5 sentence description here].\"

    """


def list_images(img_dir: Path) -> List[Path]:
    return [p for p in sorted(img_dir.iterdir()) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]

def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    images = list_images(IMAGE_DIR)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for p in images:
            abs_path = str(p.resolve())
            instruction = build_instruction(abs_path)

            obj = {"instruction": instruction, "input": f"The image path is: {abs_path}", "output": "", "images": [abs_path]}
            f.write(json.dumps(obj) + "\n")

    print(f"[OK] Generated {len(images)} samples at {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
