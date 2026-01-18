import requests
from pathlib import Path
from tqdm import tqdm
import time

USERNAME = "amdc1802@gmail.com"  # Your Copernicus email
PASSWORD = "Chalfont#2006"            # Your Copernicus password

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCT_NAMES = [
    "S1A_IW_GRDH_1SDV_20260111T233937_20260111T234002_062726_07DD91_9254",
    "S1A_IW_GRDH_1SDV_20260111T095455_20260111T095520_062718_07DD3C_839D",
    "S1A_IW_GRDH_1SDV_20251114T225109_20251114T225134_061880_07BCB4_AC61",
    "S1C_IW_GRDH_1SDV_20251215T084123_20251215T084148_005460_00ADF4_F739",
    "S1C_IW_GRDH_1SDV_20260117T090611_20260117T090636_005941_00BEAF_CE7B",
    "S1A_IW_GRDH_1SDV_20260117T125115_20260117T125140_062807_07E0B1_E611",
    "S1A_IW_GRDH_1SDV_20260117T104256_20260117T104321_062806_07E0A7_E318",
    "S1A_IW_GRDH_1SDV_20260117T104141_20260117T104206_062806_07E0A7_469C",
    "S1C_IW_GRDH_1SDV_20260117T203427_20260117T203452_005948_00BEE7_F508",
    "S1A_IW_GRDH_1SDV_20260116T113541_20260116T113606_062792_07E02D_AC54",
    "S1A_IW_GRDH_1SDV_20260117T122220_20260117T122245_062807_07E0AD_FF7A",
    "S1A_IW_GRDH_1SDV_20260116T231538_20260116T231603_062799_07E070_FAFF",
    "S1C_IW_GRDH_1SDV_20260117T225204_20260117T225229_005949_00BEF3_EA11",
    "S1C_IW_GRDH_1SDV_20260117T225139_20260117T225204_005949_00BEF3_6D42",
]

def get_token():
    """Get authentication token."""
    print("Authenticating...")
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "password",
        "username": USERNAME,
        "password": PASSWORD,
        "client_id": "cdse-public"
    }
    response = requests.post(url, data=data)
    
    if response.status_code != 200:
        raise Exception(f"Authentication failed: {response.text}")
    
    print("Authenticated!")
    return response.json()["access_token"]


def search_product(product_name, token):
    """Search for product UUID by name."""
    search_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    
    filter_query = f"contains(Name, '{product_name}')"
    
    params = {
        "$filter": filter_query,
        "$top": 1
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(search_url, params=params, headers=headers)
    
    if response.status_code == 403:
        print(f"  ⚠️ Token expired or access denied (403)")
        return None, None  # Return tuple, not None
    
    if response.status_code != 200:
        print(f"  Search failed: {response.status_code}")
        return None, None  # Return tuple, not None
    
    results = response.json()
    
    if results.get("value") and len(results["value"]) > 0:
        product = results["value"][0]
        return product["Id"], product["Name"]
    
    return None, None


def download_product(product_id, product_name, token):
    """Download a single product by UUID."""
    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}
    
    output_path = OUTPUT_DIR / f"{product_name}.zip"
    
    # Skip if already downloaded
    if output_path.exists():
        size_gb = output_path.stat().st_size / 1e9
        if size_gb > 0.5:
            print(f"✓ Already exists: {product_name[:50]}... ({size_gb:.2f} GB)")
            return True
        else:
            output_path.unlink()
    
    print(f"Downloading: {product_name[:60]}...")
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        if response.status_code == 202:
            print(f"⏳ Product offline - needs to be ordered first")
            return "offline"
        
        response.raise_for_status()
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        return False
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✓ Saved!")
    return True

def main():
    print("="*60)
    print("SENTINEL-1 DOWNLOAD SCRIPT")
    print("="*60)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Products: {len(PRODUCT_NAMES)}")
    print()
    
    token = get_token()
    
    successful = 0
    failed = []
    offline = []
    
    for i, product_name in enumerate(PRODUCT_NAMES):
        print(f"\n[{i+1}/{len(PRODUCT_NAMES)}]")
        
        # Search for UUID
        product_id, full_name = search_product(product_name, token)
        
        if product_id is None:
            print(f"❌ Not found: {product_name[:50]}...")
            failed.append(product_name)
            continue
        
        # Download
        result = download_product(product_id, full_name, token)
        
        if result == True:
            successful += 1
        elif result == "offline":
            offline.append(product_name)
        else:
            failed.append(product_name)
        
        # Delay between downloads
        time.sleep(2)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Successful: {successful}")
    print(f"⏳ Offline (need ordering): {len(offline)}")
    print(f"❌ Failed: {len(failed)}")
    
    if offline:
        print("\nOffline products - go to Copernicus and click 'Order':")
        for name in offline:
            print(f"  {name[:60]}")


if __name__ == "__main__":
    main()