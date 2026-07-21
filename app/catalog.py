"""Per-partner product taxonomies for three PAYBACK partner ecosystems.

Each partner has its OWN category tree (rich taxonomy) with a couple of products
per category — reflecting how disparate real partner feeds are — and everything is
normalized into one shared product schema at ingestion time.

Partner profiles per the challenge:
  - dm     : high frequency, low price (drugstore)
  - edeka  : grocery, fresh produce
  - amazon : long-tail general merchandise
"""

import json
import random
from pathlib import Path
from typing import Dict, List

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"

# taxonomy: {partner: {category: [(name, brand, base_price, unit, tags, popularity)]}}
_DM_TAXONOMY = {
    "Baby & Kind": [
        ("Windeln Größe 4 Maxi", "babylove", 4.95, "42 St", ["windeln", "diapers", "baby", "wickeln"], 0.95),
        ("Feuchttücher Sensitiv", "babylove", 1.75, "80 St", ["feuchttücher", "wipes", "baby"], 0.90),
        ("Babybrei Pastinake", "dmBio", 1.15, "190 g", ["babybrei", "baby food", "bio", "organic"], 0.60),
        ("Schnuller Silikon 2er", "babylove", 3.25, "2 St", ["schnuller", "pacifier", "baby"], 0.50),
    ],
    "Haarpflege": [
        ("Shampoo Repair & Care", "Balea", 1.95, "300 ml", ["shampoo", "haare", "hair"], 0.92),
        ("Haargel Strong", "Balea Men", 1.55, "150 ml", ["haargel", "hair gel"], 0.42),
    ],
    "Körperpflege": [
        ("Duschgel Meeresbrise", "Balea", 0.95, "300 ml", ["duschgel", "shower gel"], 0.88),
        ("Deo Roll-On Fresh", "Balea", 1.35, "50 ml", ["deo", "deodorant"], 0.72),
        ("Bodylotion Urea", "Balea", 2.25, "400 ml", ["bodylotion", "body lotion", "haut"], 0.57),
        ("Handcreme Kamille", "Balea", 1.25, "100 ml", ["handcreme", "hand cream"], 0.65),
    ],
    "Gesichtspflege": [
        ("Mizellenwasser", "Balea", 2.45, "400 ml", ["mizellenwasser", "micellar water", "makeup"], 0.58),
        ("Gesichtscreme Q10", "Balea", 3.45, "50 ml", ["gesichtscreme", "face cream", "haut", "anti-aging"], 0.55),
    ],
    "Mundhygiene": [
        ("Zahnpasta Kräuter", "Dontodent", 0.85, "125 ml", ["zahnpasta", "toothpaste", "zähne"], 0.85),
        ("Zahnbürste Mittel", "Dontodent", 1.45, "1 St", ["zahnbürste", "toothbrush"], 0.70),
    ],
    "Gesundheit": [
        ("Vitamin C Brausetabletten", "Mivolis", 0.65, "20 St", ["vitamin", "vitamins", "immunsystem"], 0.68),
        ("Magnesium Tabletten", "Mivolis", 1.95, "45 St", ["magnesium", "muskeln", "sport"], 0.51),
    ],
    "Sonnenschutz": [
        ("Sonnencreme LSF 50", "Sundance", 4.75, "200 ml", ["sonnencreme", "sunscreen", "sonne", "uv"], 0.75),
        ("After Sun Lotion", "Sundance", 3.25, "200 ml", ["after sun", "sonne", "haut"], 0.46),
    ],
    "Rasur": [
        ("Rasierklingen 4er", "Balea Men", 4.55, "4 St", ["rasierklingen", "razor blades", "rasur", "shaving"], 0.52),
        ("Rasierschaum Sensitive", "Balea Men", 1.75, "300 ml", ["rasierschaum", "shaving foam", "rasur"], 0.44),
    ],
    "Haushalt & Reinigung": [
        ("Vollwaschmittel Pulver", "Denkmit", 3.95, "1.35 kg", ["waschmittel", "detergent", "wäsche", "laundry"], 0.80),
        ("Spülmittel Ultra", "Denkmit", 0.75, "500 ml", ["spülmittel", "dish soap"], 0.78),
        ("Toilettenpapier 3-lagig", "Sanft&Sicher", 2.35, "8 Rollen", ["toilettenpapier", "toilet paper"], 0.82),
        ("Allzweckreiniger Zitrone", "Denkmit", 1.15, "1 l", ["reiniger", "cleaner", "putzen"], 0.55),
        ("Taschentücher Box", "Soft&Sicher", 1.05, "150 St", ["taschentücher", "tissues"], 0.62),
    ],
}

_EDEKA_TAXONOMY = {
    "Obst & Gemüse": [
        ("Äpfel Elstar", "EDEKA Regional", 2.79, "1 kg", ["äpfel", "apfel", "apples", "obst", "fruit"], 0.82),
        ("Bananen", "EDEKA", 1.79, "1 kg", ["bananen", "bananas", "obst", "fruit"], 0.86),
        ("Avocado essreif", "EDEKA", 1.49, "1 St", ["avocado", "obst", "frisch"], 0.64),
        ("Rispentomaten", "EDEKA Regional", 2.99, "500 g", ["tomaten", "tomatoes", "gemüse", "frisch"], 0.74),
        ("Salatgurke", "EDEKA", 0.89, "1 St", ["gurke", "cucumber", "salat", "gemüse"], 0.66),
        ("Knoblauch", "EDEKA", 0.79, "3 St", ["knoblauch", "garlic", "gemüse", "pasta"], 0.65),
        ("Zwiebeln rot", "EDEKA", 1.29, "1 kg", ["zwiebeln", "onions", "gemüse"], 0.68),
        ("Basilikum frisch Topf", "EDEKA", 1.79, "1 Topf", ["basilikum", "basil", "kräuter", "herbs", "pasta"], 0.55),
    ],
    "Molkerei & Eier": [
        ("Frische Vollmilch 3.8%", "EDEKA", 1.19, "1 l", ["milch", "milk", "frisch"], 0.92),
        ("Deutsche Markenbutter", "EDEKA", 1.99, "250 g", ["butter"], 0.88),
        ("Bio-Joghurt Natur", "EDEKA Bio", 0.99, "500 g", ["joghurt", "yogurt", "bio", "organic"], 0.69),
        ("Bio-Eier Freiland", "EDEKA Bio", 3.29, "10 St", ["eier", "eggs", "bio", "organic"], 0.85),
    ],
    "Käse": [
        ("Parmesan gerieben", "EDEKA Italia", 2.79, "100 g", ["parmesan", "käse", "cheese", "pasta"], 0.70),
        ("Gouda jung Scheiben", "EDEKA", 2.49, "400 g", ["gouda", "käse", "cheese"], 0.73),
        ("Mozzarella", "EDEKA Italia", 0.99, "125 g", ["mozzarella", "käse", "cheese", "pasta"], 0.67),
    ],
    "Fleisch & Fisch": [
        ("Hähnchenbrustfilet", "EDEKA", 6.49, "600 g", ["hähnchen", "chicken", "fleisch", "meat"], 0.76),
        ("Rinderhackfleisch", "EDEKA", 4.99, "500 g", ["hackfleisch", "ground beef", "fleisch", "bolognese", "pasta"], 0.79),
        ("Lachsfilet", "EDEKA Fisch", 8.99, "250 g", ["lachs", "salmon", "fisch", "fish"], 0.58),
        ("Grillwürstchen Bratwurst", "EDEKA", 3.99, "400 g", ["bratwurst", "grillen", "bbq", "sausage"], 0.54),
    ],
    "Nudeln & Trockensortiment": [
        ("Spaghetti No.5", "EDEKA Italia", 0.99, "500 g", ["spaghetti", "nudeln", "pasta"], 0.90),
        ("Penne Rigate", "EDEKA Italia", 0.99, "500 g", ["penne", "nudeln", "pasta"], 0.75),
        ("Basmati Reis", "EDEKA", 2.49, "1 kg", ["reis", "rice"], 0.62),
    ],
    "Konserven & Saucen": [
        ("Passierte Tomaten", "EDEKA", 0.89, "500 g", ["tomaten", "tomatoes", "passata", "sauce", "pasta"], 0.80),
        ("Olivenöl Extra Vergine", "EDEKA Bio", 6.99, "500 ml", ["olivenöl", "olive oil", "bio", "organic", "pasta"], 0.72),
        ("Pesto Genovese", "EDEKA Italia", 2.29, "190 g", ["pesto", "sauce", "pasta", "basilikum"], 0.59),
    ],
    "Backwaren": [
        ("Vollkornbrot", "EDEKA Bäckerei", 2.49, "750 g", ["brot", "bread", "vollkorn"], 0.78),
        ("Toastbrot Buttertoast", "EDEKA", 1.29, "500 g", ["toast", "toastbrot", "brot", "bread"], 0.71),
        ("Marmorkuchen", "EDEKA Bäckerei", 2.99, "400 g", ["kuchen", "cake", "marmorkuchen", "gebäck"], 0.54),
    ],
    "Backzutaten": [
        ("Weizenmehl Type 405", "EDEKA", 0.89, "1 kg", ["mehl", "flour", "backen", "baking"], 0.69),
        ("Zucker fein", "EDEKA", 1.09, "1 kg", ["zucker", "sugar", "backen"], 0.66),
        ("Backmischung Schokokuchen", "EDEKA", 1.99, "500 g", ["kuchen", "cake", "backmischung", "backen", "schokolade"], 0.52),
    ],
    "Süßigkeiten & Snacks": [
        ("Schokolade Vollmilch", "EDEKA", 1.19, "100 g", ["schokolade", "chocolate", "süßigkeiten", "sweets", "naschen"], 0.83),
        ("Gummibärchen Goldbären", "Haribo", 1.45, "175 g", ["gummibärchen", "gummy bears", "süßigkeiten", "sweets", "candy"], 0.77),
        ("Kartoffelchips Paprika", "funny-frisch", 1.99, "175 g", ["chips", "crisps", "snacks", "knabbern"], 0.72),
    ],
    "Frühstück": [
        ("Bio-Honig", "EDEKA Bio", 4.49, "500 g", ["honig", "honey", "bio", "organic", "frühstück"], 0.49),
        ("Knuspermüsli Schoko", "EDEKA", 2.19, "500 g", ["müsli", "muesli", "cereal", "frühstück", "breakfast"], 0.61),
    ],
    "Getränke": [
        ("Mineralwasser Classic 6er", "EDEKA", 2.34, "6x1.5 l", ["wasser", "water", "mineralwasser"], 0.81),
        ("Orangensaft Direktsaft", "EDEKA", 2.29, "1 l", ["orangensaft", "orange juice", "saft"], 0.63),
        ("Röstkaffee ganze Bohne", "EDEKA Gut&Günstig", 9.99, "1 kg", ["kaffee", "coffee", "bohnen"], 0.71),
        ("Apfelschorle 6er", "EDEKA", 3.49, "6x1 l", ["apfelschorle", "schorle", "saft"], 0.57),
    ],
    "Tiefkühl": [
        ("Tiefkühlpizza Salami", "EDEKA", 2.79, "320 g", ["pizza", "tiefkühlpizza", "salami"], 0.81),
        ("Gemüse-Mix TK", "EDEKA", 1.89, "750 g", ["gemüse", "vegetables", "tiefkühl", "frozen"], 0.53),
    ],
}

_AMAZON_TAXONOMY = {
    "Elektronik & Zubehör": [
        ("Bluetooth Kopfhörer In-Ear", "Soundcore", 39.99, "1 St", ["kopfhörer", "headphones", "bluetooth", "musik"], 0.85),
        ("USB-C Ladekabel 2m 3er", "Anker", 12.99, "3 St", ["kabel", "cable", "usb", "laden"], 0.78),
        ("Powerbank 20000mAh", "Anker", 29.99, "1 St", ["powerbank", "akku", "battery"], 0.72),
        ("Handyhülle iPhone klar", "Spigen", 13.99, "1 St", ["handyhülle", "phone case", "hülle"], 0.62),
        ("Webcam Full HD", "Logitech", 49.99, "1 St", ["webcam", "kamera", "homeoffice"], 0.55),
    ],
    "Smart Home": [
        ("Smarte Glühbirne E27 WLAN", "TP-Link", 11.99, "1 St", ["glühbirne", "smart bulb", "licht", "light", "smart home"], 0.58),
        ("Smarte Steckdose WLAN 2er", "TP-Link", 17.99, "2 St", ["steckdose", "smart plug", "smart home"], 0.49),
    ],
    "Bücher & Medien": [
        ("E-Reader 6 Zoll", "Kindle", 99.99, "1 St", ["ereader", "e-reader", "lesen", "reading", "buch"], 0.66),
        ("Roman Der Distelfink", "Goldmann", 12.99, "1 St", ["roman", "novel", "buch", "book", "lesen"], 0.46),
        ("Kochbuch Pasta Amore", "GU Verlag", 24.99, "1 St", ["kochbuch", "cookbook", "pasta", "buch", "kochen"], 0.44),
    ],
    "Küche & Haushaltsgeräte": [
        ("Wasserkocher Edelstahl 1.7l", "Philips", 32.99, "1 St", ["wasserkocher", "kettle", "küche"], 0.64),
        ("Milchaufschäumer elektrisch", "Severin", 36.99, "1 St", ["milchaufschäumer", "milk frother", "kaffee", "coffee"], 0.47),
        ("Bratpfanne beschichtet 28cm", "Tefal", 29.99, "1 St", ["pfanne", "pan", "braten", "küche", "kochen"], 0.63),
        ("Messerset 5-teilig", "Zwilling", 89.99, "5 St", ["messer", "knife", "küche", "kochen"], 0.51),
        ("LED Schreibtischlampe", "TaoTronics", 25.99, "1 St", ["lampe", "lamp", "schreibtisch", "desk"], 0.52),
        ("Luftbefeuchter Ultraschall", "Levoit", 44.99, "1 St", ["luftbefeuchter", "humidifier"], 0.39),
    ],
    "Spielzeug & Spiele": [
        ("Brettspiel Catan", "Kosmos", 34.99, "1 St", ["brettspiel", "board game", "spiel", "geschenk", "gift"], 0.59),
        ("Puzzle 1000 Teile Alpen", "Ravensburger", 14.99, "1 St", ["puzzle", "geschenk", "gift"], 0.48),
        ("Holzbausteine 100 Teile", "Eichhorn", 19.99, "100 St", ["bausteine", "building blocks", "spielzeug", "toy", "kinder", "baby", "geschenk"], 0.53),
        ("Lego City Feuerwehr", "LEGO", 49.99, "1 St", ["lego", "spielzeug", "toy", "kinder", "geschenk", "gift"], 0.74),
    ],
    "Sport & Fitness": [
        ("Yogamatte rutschfest", "Lotuscrafts", 24.99, "1 St", ["yogamatte", "yoga mat", "sport", "fitness"], 0.61),
        ("Fitness Tracker", "Xiaomi", 34.99, "1 St", ["fitness", "tracker", "sport", "smartwatch"], 0.70),
        ("Springseil Speed", "Beast Gear", 12.99, "1 St", ["springseil", "jump rope", "sport", "fitness"], 0.43),
    ],
    "Outdoor & Reisen": [
        ("Rucksack 25l wasserdicht", "Deuter", 59.99, "1 St", ["rucksack", "backpack", "wandern", "reisen"], 0.56),
        ("Campingstuhl faltbar", "Coleman", 27.99, "1 St", ["campingstuhl", "camping chair", "outdoor"], 0.41),
        ("Trinkflasche Edelstahl 1l", "Klean Kanteen", 21.99, "1 St", ["trinkflasche", "water bottle", "outdoor", "sport"], 0.50),
    ],
    "Möbel & Büro": [
        ("Bürostuhl ergonomisch", "Songmics", 129.99, "1 St", ["bürostuhl", "office chair", "homeoffice"], 0.45),
        ("Schreibtisch höhenverstellbar", "Flexispot", 249.99, "1 St", ["schreibtisch", "desk", "homeoffice", "büro"], 0.40),
    ],
    "Gutscheine": [
        ("Geschenkkarte Universal", "Wunschgutschein", 25.00, "1 St", ["geschenkkarte", "gift card", "gutschein", "geschenk", "gift"], 0.57),
        ("Streaming Gutschein", "Netflix", 25.00, "1 St", ["streaming", "gutschein", "geschenk", "gift", "filme"], 0.48),
    ],
}

PARTNERS: Dict[str, Dict] = {
    "dm": {"label": "Partner A (dm)", "profile": "drugstore - high frequency, low price",
           "taxonomy": _DM_TAXONOMY},
    "edeka": {"label": "Partner B (EDEKA)", "profile": "grocery & fresh produce",
              "taxonomy": _EDEKA_TAXONOMY},
    "amazon": {"label": "Partner C (Amazon)", "profile": "long-tail general merchandise",
               "taxonomy": _AMAZON_TAXONOMY},
}


def taxonomy_summary() -> Dict[str, Dict]:
    return {
        partner: {
            "label": meta["label"],
            "profile": meta["profile"],
            "categories": {cat: len(items) for cat, items in meta["taxonomy"].items()},
        }
        for partner, meta in PARTNERS.items()
    }


def build_catalog(seed: int = 42) -> List[dict]:
    """Flatten per-partner taxonomies into the shared schema (seeded price jitter)."""
    rng = random.Random(seed)
    products = []
    for partner, meta in PARTNERS.items():
        counter = 0
        for category, items in meta["taxonomy"].items():
            for name, brand, price, unit, tags, pop in items:
                counter += 1
                jitter = 1.0 + rng.uniform(-0.05, 0.05)
                products.append({
                    "id": f"{partner}-{counter:03d}",
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
