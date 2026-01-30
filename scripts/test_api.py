import requests

tif_path = r'data\raw\S1A_IW_GRDH_1SDV_20251114T225109_20251114T225134_061880_07BCB4_AC61.SAFE\measurement\s1a-iw-grd-vv-20251114t225109-20251114t225134-061880-07bcb4-001.tiff'
print(f'Compressing: {tif_path}')
with open(tif_path, 'rb') as f:
    resp = requests.post('http://localhost:8000/encode?model=8x', files={'file': f})
print(f'Compress status: {resp.status_code}')
if resp.status_code == 200:
    with open('test_compressed.npz', 'wb') as out:
        out.write(resp.content)
    print(f'Saved: test_compressed.npz ({len(resp.content):,} bytes)')
    print('Decompressing...')
    with open('test_compressed.npz', 'rb') as f:
        resp2 = requests.post('http://localhost:8000/decode?model=8x', files={'file': f})
    print(f'Decompress status: {resp2.status_code}')
    if resp2.status_code == 200:
        with open('test_reconstructed.tiff', 'wb') as out:
            out.write(resp2.content)
        print(f'Saved: test_reconstructed.tiff ({len(resp2.content):,} bytes)')
else:
    print(f'Error: {resp.text}')
