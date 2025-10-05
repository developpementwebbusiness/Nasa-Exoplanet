"""
Test corrigé de l'API
"""
import requests

API = "http://localhost:8000"

# 1. Vérifier l'API
print("1. Test santé API...")
r = requests.get(f"{API}/health")
print(f"   → {r.json()}")

# 2. Import dans IA_1
print("\n2. Test import dans IA_1...")
with open("test1.txt", "w") as f: f.write("data1")
with open("test2.txt", "w") as f: f.write("data2")
with open("test3.txt", "w") as f: f.write("data3")

files = [
    ('files', ('test1.txt', open('test1.txt', 'rb'))),
    ('files', ('test2.txt', open('test2.txt', 'rb'))),
    ('files', ('test3.txt', open('test3.txt', 'rb')))
]

r = requests.post(
    f"{API}/import_ia_files",
    files=files,
    data={'ia_folder': 'AI/STAR_AI_v2', 'user_id': 'test'}  # ← IA_1
)
print(f"   → Status: {r.status_code}")
print(f"   → Importés: {r.json()['imported_count']}/3")

# 3. Export depuis IA_1 (PAS TEST)
print("\n3. Test export depuis IA_1...")
r = requests.get(f"{API}/export_ia_files/AI/STAR_AI_v2")  # ← IA_1 au lieu de TEST
if r.status_code == 200:
    with open("export_IA_1.zip", "wb") as f:
        f.write(r.content)
    print(f"   ✅ ZIP téléchargé: export_IA_1.zip ({len(r.content)} bytes)")
else:
    print(f"   ❌ Erreur: {r.status_code}")
    print(f"   → {r.json()}")

print("\n✅ Tests terminés!")

