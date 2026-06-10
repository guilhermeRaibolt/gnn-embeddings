import ast
import gzip
import os
import urllib.request

import pandas as pd

DEFAULT_SUBCATEGORY_DEPTH = 2

# data from https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html

DATASET_URLS = {
    # 'data/books.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz',
    # 'data/electronics.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz',
    # 'data/movies_tv.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Movies_and_TV.json.gz',
    # 'data/cds_vinyl.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_CDs_and_Vinyl.json.gz',
    # 'data/clothing_shoes_jewelry.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz',
    # 'data/home_kitchen.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Home_and_Kitchen.json.gz',
    # 'data/kindle_store.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Kindle_Store.json.gz',
    # 'data/sports_outdoors.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz',
    # 'data/cell_phones_accessories.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Cell_Phones_and_Accessories.json.gz',
    # 'data/health_personal_care.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Health_and_Personal_Care.json.gz',
    # 'data/toys_games.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz',
    # 'data/video_games.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Video_Games.json.gz',
    # 'data/tools_home_improvement.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Tools_and_Home_Improvement.json.gz',
    # 'data/beauty.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz',
    # 'data/apps_android.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Apps_for_Android.json.gz',
    # 'data/office_products.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Office_Products.json.gz',
    # 'data/pet_supplies.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Pet_Supplies.json.gz',
    # 'data/automotive.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Automotive.json.gz',
    # 'data/grocery_gourmet_food.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Grocery_and_Gourmet_Food.json.gz',
    # 'data/patio_lawn_garden.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Patio_Lawn_and_Garden.json.gz',
    # 'data/baby.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Baby.json.gz',
    # 'data/digital_music.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Digital_Music.json.gz',
    'data/musical_instruments.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Musical_Instruments.json.gz',
    # 'data/amazon_instant_video.json.gz': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Amazon_Instant_Video.json.gz'
}

def ensure_data_exists(local_path):
    if os.path.exists(local_path):
        print(f"-> {local_path} already exists. Skipping download.")
        return

    if local_path not in DATASET_URLS:
        raise ValueError(f"URL for {local_path} is not defined in DATASET_URLS.")

    url = DATASET_URLS[local_path]
    dir_name = os.path.dirname(local_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print(f"-> Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"-> Successfully downloaded and saved to {local_path}")
    except Exception as e:
        print(f" Error downloading {url}: {e}")
        raise

def parse_entries(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield ast.literal_eval(line)
            

def extract_description(desc):
    # description is sometimes a string, sometimes a list, sometimes missing.
    if isinstance(desc, list):
        return ' '.join(str(d) for d in desc)
    return str(desc) if desc else ''


def extract_category(category_list, depth=DEFAULT_SUBCATEGORY_DEPTH):
    # fallback if the record doesn't contain category listings
    if not category_list or not isinstance(category_list, list):
        return None
    
    # in multi-branch items, we prioritize the branch that 
    # explicitly satisfies our required depth constraint.
    for path in category_list:
        if isinstance(path, list) and len(path) > depth:
            return path[depth]
            
    # if no single branch is long enough, grab the deepest leaf node 
    # of the very first branch to avoid throwing away data entirely.
    if category_list and isinstance(category_list[0], list) and len(category_list[0]) > 0:
        return category_list[0][-1]
        
    return None


def load_dataset_to_df(path, depth=DEFAULT_SUBCATEGORY_DEPTH):

    ensure_data_exists(path)

    data = []
    for i, entry in enumerate(parse_entries(path)):
        
        # for now we only include title and description, but 
        # later on we can include the review text or something else.
        title = entry.get('title', '')
        description = entry.get('description', '')
        description = extract_description(description)

        category = extract_category(entry.get('categories', []), depth)
        if not category:
            continue
            
        data.append({
            'text': f"{title} {description}".strip(),
            'category': category
        })
        
    return pd.DataFrame(data)

def load_all_datasets_to_df(depth=DEFAULT_SUBCATEGORY_DEPTH):
    df = pd.DataFrame()
    for local_path in DATASET_URLS.keys():

        if "musical_instruments" not in local_path.lower():
            continue

        dataset_df = load_dataset_to_df(local_path, depth)
        df = pd.concat([df, dataset_df], ignore_index=True)
    return df