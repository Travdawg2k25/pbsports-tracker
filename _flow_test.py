import time, requests, sys

BASE = "http://localhost:8000"

print("1. health")
print("  ", requests.get(f"{BASE}/api/health").json()["status"])

print("2. signup")
r = requests.post(f"{BASE}/api/signup", json={"email": "coach@test.com", "package": "starter"}).json()
key = r["api_key"]; print("   credits=", r["credits"])
H = {"X-API-Key": key}

print("3. account")
print("   ", requests.get(f"{BASE}/api/account", headers=H).json())

print("4. upload _clip.mp4")
with open("_clip.mp4", "rb") as f:
    up = requests.post(f"{BASE}/api/games", headers=H, files={"file": ("_clip.mp4", f, "video/mp4")}).json()
gid = up["game_id"]; print("   game_id=", gid, "status=", up["status"])

print("5. poll for player scan")
for _ in range(60):
    time.sleep(3)
    st = requests.get(f"{BASE}/api/games/{gid}", headers=H).json()
    print("   status=", st["status"], "players_found=", st.get("players_found"))
    if st["status"] in ("players_ready", "error"): break
if st["status"] == "error":
    print("   ERROR:", st.get("error")); sys.exit(1)

pl = requests.get(f"{BASE}/api/games/{gid}/players", headers=H).json()
ids = [p["track_id"] for p in pl["players"][:2]]
print("   selecting player_ids=", ids)

print("6. analyze")
an = requests.post(f"{BASE}/api/games/{gid}/analyze", headers=H, json={"player_ids": ids}).json()
print("   status=", an.get("status"), "credits_remaining=", an.get("credits_remaining"))

print("7. poll analysis")
for _ in range(120):
    time.sleep(4)
    st = requests.get(f"{BASE}/api/games/{gid}", headers=H).json()
    print("   status=", st["status"], "progress=", st.get("progress"))
    if st["status"] in ("complete", "error"): break
if st["status"] == "error":
    print("   ERROR:", st.get("error")); sys.exit(1)

print("8. stats")
stats = requests.get(f"{BASE}/api/games/{gid}/stats", headers=H).json()
print("   score=", stats.get("score"))
print("   event_counts=", stats.get("event_counts"))
print("   reels=", st.get("reels"))

# Verify credit was spent
acct = requests.get(f"{BASE}/api/account", headers=H).json()
print("9. credits after analyze:", acct["credits"], "(started 5)")
print("DONE game_id=", gid)
