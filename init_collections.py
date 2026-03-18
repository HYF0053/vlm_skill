import json
import requests
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'memo.json')

def init_collections():
    if not os.path.exists(CONFIG_PATH):
        print(f"找不到設定檔: {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    qdrant_config = config.get("qdrant", {})
    host = qdrant_config.get("host", "10.1.1.7")
    port = qdrant_config.get("port", 26620)
    collections = qdrant_config.get("collections", [])

    base_url = f"http://{host}:{port}"

    for col in collections:
        name = col.get("name")
        vector_size = col.get("vector_size", 1024)
        distance = col.get("distance", "Cosine")

        print(f"正在檢查 Collection: {name} ...")
        
        # 檢查是否存在
        check_url = f"{base_url}/collections/{name}"
        resp = requests.get(check_url)
        
        if resp.status_code == 200:
            print(f"Collection '{name}' 已經存在，跳過建立。")
        else:
            print(f"準備建立 Collection '{name}' (維度: {vector_size}, 距離計算: {distance})")
            create_url = f"{base_url}/collections/{name}"
            payload = {
                "vectors": {
                    "size": vector_size,
                    "distance": distance
                }
            }
            create_resp = requests.put(create_url, json=payload)
            if create_resp.status_code == 200:
                print(f"✅ 成功建立 Collection '{name}'")
            else:
                print(f"❌ 建立失敗: {create_resp.text}")

if __name__ == "__main__":
    init_collections()
