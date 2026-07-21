"""Synthetic product catalogs for three PAYBACK partner ecosystems.

Curated seed items (realistic German retail assortment) expanded deterministically
(seeded) with price jitter so the dataset is "synthetic" yet reproducible.

Partner profiles per the challenge:
  - partner_a "dm"     : high frequency, low price (drugstore)
  - partner_b "edeka"  : grocery, fresh produce
  - partner_c "amazon" : long-tail general merchandise
"""

import json
import random
from pathlib import Path
from typing import Dict, List

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"

# (name, brand, category, base_price, unit, tags[de+en], popularity 0..1)
_DM = [
    ("Windeln Größe 4 Maxi", "babylove", "Baby", 4.95, "42 St", ["windeln", "diapers", "baby", "wickeln"], 0.95),
    ("Feuchttücher Sensitiv", "babylove", "Baby", 1.75, "80 St", ["feuchttücher", "wipes", "baby"], 0.90),
    ("Babybrei Pastinake", "dmBio", "Baby", 1.15, "190 g", ["babybrei", "baby food", "bio", "organic"], 0.60),
    ("Schnuller Silikon 2er", "babylove", "Baby", 3.25, "2 St", ["schnuller", "pacifier", "baby"], 0.50),
    ("Shampoo Repair & Care", "Balea", "Haarpflege", 1.95, "300 ml", ["shampoo", "haare", "hair"], 0.92),
    ("Duschgel Meeresbrise", "Balea", "Körperpflege", 0.95, "300 ml", ["duschgel", "shower gel"], 0.88),
    ("Zahnpasta Kräuter", "Dontodent", "Mundhygiene", 0.85, "125 ml", ["zahnpasta", "toothpaste", "zähne"], 0.85),
    ("Zahnbürste Mittel", "Dontodent", "Mundhygiene", 1.45, "1 St", ["zahnbürste", "toothbrush"], 0.70),
    ("Sonnencreme LSF 50", "Sundance", "Sonnenschutz", 4.75, "200 ml", ["sonnencreme", "sunscreen", "sonne", "uv"], 0.75),
    ("Handcreme Kamille", "Balea", "Körperpflege", 1.25, "100 ml", ["handcreme", "hand cream"], 0.65),
    ("Vollwaschmittel Pulver", "Denkmit", "Haushalt", 3.95, "1.35 kg", ["waschmittel", "detergent", "wäsche", "laundry"], 0.80),
    ("Spülmittel Ultra", "Denkmit", "Haushalt", 0.75, "500 ml", ["spülmittel", "dish soap"], 0.78),
    ("Toilettenpapier 3-lagig", "Sanft&Sicher", "Haushalt", 2.35, "8 Rollen", ["toilettenpapier", "toilet paper"], 0.82),
    ("Allzweckreiniger Zitrone", "Denkmit", "Haushalt", 1.15, "1 l", ["reiniger", "cleaner", "putzen"], 0.55),
    ("Deo Roll-On Fresh", "Balea", "Körperpflege", 1.35, "50 ml", ["deo", "deodorant"], 0.72),
    ("Lippenpflege Classic", "Balea", "Körperpflege", 0.95, "4.8 g", ["lippenpflege", "lip balm"], 0.48),
    ("Mizellenwasser", "Balea", "Gesichtspflege", 2.45, "400 ml", ["mizellenwasser", "micellar water", "makeup"], 0.58),
    ("Rasierklingen 4er", "Balea Men", "Rasur", 4.55, "4 St", ["rasierklingen", "razor blades", "rasur", "shaving"], 0.52),
    ("Vitamin C Brausetabletten", "Mivolis", "Gesundheit", 0.65, "20 St", ["vitamin", "vitamins", "immunsystem"], 0.68),
    ("Taschentücher Box", "Soft&Sicher", "Haushalt", 1.05, "150 St", ["taschentücher", "tissues"], 0.62),
    ("Bodylotion Urea", "Balea", "Körperpflege", 2.25, "400 ml", ["bodylotion", "body lotion", "haut"], 0.57),
    ("Haargel Strong", "Balea Men", "Haarpflege", 1.55, "150 ml", ["haargel", "hair gel"], 0.42),
]

_EDEKA = [
    ("Spaghetti No.5", "EDEKA Italia", "Nudeln", 0.99, "500 g", ["spaghetti", "nudeln", "pasta"], 0.90),
    ("Penne Rigate", "EDEKA Italia", "Nudeln", 0.99, "500 g", ["penne", "nudeln", "pasta"], 0.75),
    ("Passierte Tomaten", "EDEKA", "Konserven", 0.89, "500 g", ["tomaten", "tomatoes", "passata", "sauce", "pasta"], 0.80),
    ("Parmesan gerieben", "EDEKA Italia", "Käse", 2.79, "100 g", ["parmesan", "käse", "cheese", "pasta"], 0.70),
    ("Olivenöl Extra Vergine", "EDEKA Bio", "Öle", 6.99, "500 ml", ["olivenöl", "olive oil", "bio", "organic", "pasta"], 0.72),
    ("Basilikum frisch Topf", "EDEKA", "Kräuter", 1.79, "1 Topf", ["basilikum", "basil", "kräuter", "herbs", "pasta"], 0.55),
    ("Knoblauch", "EDEKA", "Gemüse", 0.79, "3 St", ["knoblauch", "garlic", "gemüse", "pasta"], 0.65),
    ("Zwiebeln rot", "EDEKA", "Gemüse", 1.29, "1 kg", ["zwiebeln", "onions", "gemüse"], 0.68),
    ("Bio-Eier Freiland", "EDEKA Bio", "Eier", 3.29, "10 St", ["eier", "eggs", "bio", "organic"], 0.85),
    ("Frische Vollmilch 3.8%", "EDEKA", "Molkerei", 1.19, "1 l", ["milch", "milk", "frisch"], 0.92),
    ("Deutsche Markenbutter", "EDEKA", "Molkerei", 1.99, "250 g", ["butter"], 0.88),
    ("Vollkornbrot", "EDEKA Bäckerei", "Backwaren", 2.49, "750 g", ["brot", "bread", "vollkorn"], 0.78),
    ("Äpfel Elstar", "EDEKA Regional", "Obst", 2.79, "1 kg", ["äpfel", "apfel", "apples", "obst", "fruit"], 0.82),
    ("Bananen", "EDEKA", "Obst", 1.79, "1 kg", ["bananen", "bananas", "obst", "fruit"], 0.86),
    ("Rispentomaten", "EDEKA Regional", "Gemüse", 2.99, "500 g", ["tomaten", "tomatoes", "gemüse", "frisch"], 0.74),
    ("Salatgurke", "EDEKA", "Gemüse", 0.89, "1 St", ["gurke", "cucumber", "salat", "gemüse"], 0.66),
    ("Hähnchenbrustfilet", "EDEKA", "Fleisch", 6.49, "600 g", ["hähnchen", "chicken", "fleisch", "meat"], 0.76),
    ("Rinderhackfleisch", "EDEKA", "Fleisch", 4.99, "500 g", ["hackfleisch", "ground beef", "fleisch", "bolognese", "pasta"], 0.79),
    ("Lachsfilet", "EDEKA Fisch", "Fisch", 8.99, "250 g", ["lachs", "salmon", "fisch", "fish"], 0.58),
    ("Röstkaffee ganze Bohne", "EDEKA Gut&Günstig", "Kaffee", 9.99, "1 kg", ["kaffee", "coffee", "bohnen"], 0.71),
    ("Orangensaft Direktsaft", "EDEKA", "Getränke", 2.29, "1 l", ["orangensaft", "orange juice", "saft"], 0.63),
    ("Bio-Joghurt Natur", "EDEKA Bio", "Molkerei", 0.99, "500 g", ["joghurt", "yogurt", "bio", "organic"], 0.69),
    ("Gouda jung Scheiben", "EDEKA", "Käse", 2.49, "400 g", ["gouda", "käse", "cheese"], 0.73),
    ("Mozzarella", "EDEKA Italia", "Käse", 0.99, "125 g", ["mozzarella", "käse", "cheese", "pasta"], 0.67),
    ("Mineralwasser Classic 6er", "EDEKA", "Getränke", 2.34, "6x1.5 l", ["wasser", "water", "mineralwasser"], 0.81),
    ("Grillwürstchen Bratwurst", "EDEKA", "Fleisch", 3.99, "400 g", ["bratwurst", "grillen", "bbq", "sausage"], 0.54),
    ("Bio-Honig", "EDEKA Bio", "Frühstück", 4.49, "500 g", ["honig", "honey", "bio", "organic", "frühstück"], 0.49),
    ("Knuspermüsli Schoko", "EDEKA", "Frühstück", 2.19, "500 g", ["müsli", "muesli", "cereal", "frühstück", "breakfast"], 0.61),
]

_AMAZON = [
    ("Bluetooth Kopfhörer In-Ear", "Soundcore", "Elektronik", 39.99, "1 St", ["kopfhörer", "headphones", "bluetooth", "musik"], 0.85),
    ("USB-C Ladekabel 2m 3er", "Anker", "Elektronik", 12.99, "3 St", ["kabel", "cable", "usb", "laden"], 0.78),
    ("Powerbank 20000mAh", "Anker", "Elektronik", 29.99, "1 St", ["powerbank", "akku", "battery"], 0.72),
    ("E-Reader 6 Zoll", "Kindle", "Elektronik", 99.99, "1 St", ["ereader", "e-reader", "lesen", "reading", "buch"], 0.66),
    ("Fitness Tracker", "Xiaomi", "Elektronik", 34.99, "1 St", ["fitness", "tracker", "sport", "smartwatch"], 0.70),
    ("Webcam Full HD", "Logitech", "Elektronik", 49.99, "1 St", ["webcam", "kamera", "homeoffice"], 0.55),
    ("Smarte Glühbirne E27 WLAN", "TP-Link", "Smart Home", 11.99, "1 St", ["glühbirne", "smart bulb", "licht", "light", "smart home"], 0.58),
    ("LED Schreibtischlampe", "TaoTronics", "Haushalt", 25.99, "1 St", ["lampe", "lamp", "schreibtisch", "desk"], 0.52),
    ("Wasserkocher Edelstahl 1.7l", "Philips", "Küche", 32.99, "1 St", ["wasserkocher", "kettle", "küche"], 0.64),
    ("Milchaufschäumer elektrisch", "Severin", "Küche", 36.99, "1 St", ["milchaufschäumer", "milk frother", "kaffee", "coffee"], 0.47),
    ("Bratpfanne beschichtet 28cm", "Tefal", "Küche", 29.99, "1 St", ["pfanne", "pan", "braten", "küche", "kochen"], 0.63),
    ("Messerset 5-teilig", "Zwilling", "Küche", 89.99, "5 St", ["messer", "knife", "küche", "kochen"], 0.51),
    ("Kochbuch Pasta Amore", "GU Verlag", "Bücher", 24.99, "1 St", ["kochbuch", "cookbook", "pasta", "buch", "kochen"], 0.44),
    ("Roman Der Distelfink", "Goldmann", "Bücher", 12.99, "1 St", ["roman", "novel", "buch", "book", "lesen"], 0.46),
    ("Brettspiel Catan", "Kosmos", "Spielzeug", 34.99, "1 St", ["brettspiel", "board game", "spiel", "geschenk", "gift"], 0.59),
    ("Puzzle 1000 Teile Alpen", "Ravensburger", "Spielzeug", 14.99, "1 St", ["puzzle", "geschenk", "gift"], 0.48),
    ("Holzbausteine 100 Teile", "Eichhorn", "Spielzeug", 19.99, "100 St", ["bausteine", "building blocks", "spielzeug", "toy", "kinder", "baby", "geschenk"], 0.53),
    ("Lego City Feuerwehr", "LEGO", "Spielzeug", 49.99, "1 St", ["lego", "spielzeug", "toy", "kinder", "geschenk", "gift"], 0.74),
    ("Yogamatte rutschfest", "Lotuscrafts", "Sport", 24.99, "1 St", ["yogamatte", "yoga mat", "sport", "fitness"], 0.61),
    ("Rucksack 25l wasserdicht", "Deuter", "Reisen", 59.99, "1 St", ["rucksack", "backpack", "wandern", "reisen"], 0.56),
    ("Campingstuhl faltbar", "Coleman", "Outdoor", 27.99, "1 St", ["campingstuhl", "camping chair", "outdoor"], 0.41),
    ("Handyhülle iPhone klar", "Spigen", "Elektronik", 13.99, "1 St", ["handyhülle", "phone case", "hülle"], 0.62),
    ("Luftbefeuchter Ultraschall", "Levoit", "Haushalt", 44.99, "1 St", ["luftbefeuchter", "humidifier"], 0.39),
    ("Bürostuhl ergonomisch", "Songmics", "Möbel", 129.99, "1 St", ["bürostuhl", "office chair", "homeoffice"], 0.45),
    ("Geschenkkarte Universal", "Wunschgutschein", "Gutscheine", 25.00, "1 St", ["geschenkkarte", "gift card", "gutschein", "geschenk", "gift"], 0.57),
]

PARTNERS: Dict[str, Dict] = {
    "dm": {"label": "Partner A (dm)", "profile": "drugstore - high frequency, low price", "items": _DM},
    "edeka": {"label": "Partner B (EDEKA)", "profile": "grocery & fresh produce", "items": _EDEKA},
    "amazon": {"label": "Partner C (Amazon)", "profile": "long-tail general merchandise", "items": _AMAZON},
}


def build_catalog(seed: int = 42) -> List[dict]:
    """Deterministically simulate the joint catalog (price jitter emulates live feeds)."""
    rng = random.Random(seed)
    products = []
    for partner, meta in PARTNERS.items():
        for i, (name, brand, category, price, unit, tags, pop) in enumerate(meta["items"]):
            jitter = 1.0 + rng.uniform(-0.05, 0.05)
            products.append({
                "id": f"{partner}-{i + 1:03d}",
                "partner": partner,
                "name": name,
                "brand": brand,
                "category": category,
                "price": round(price * jitter, 2),
                "unit": unit,
                "tags": tags,
                "popularity": round(pop, 2),
            })
    return products


def load_catalog() -> List[dict]:
    """Load catalog from disk, generating it on first run (idempotent ingestion)."""
    if DATA_PATH.exists():
        return json.loads(DATA_PATH.read_text(encoding="utf-8"))
    catalog = build_catalog()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    return catalog


if __name__ == "__main__":
    cat = build_catalog()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(cat, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {len(cat)} products to {DATA_PATH}")
