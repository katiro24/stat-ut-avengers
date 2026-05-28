import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as pltfrom 
from sklearn.ensemble import IsolationForest

# +
# ONLY your selected posts
poster = [
    # EGENKAPITAL
    "20000","20100","20200","20300",
    "20410","20420","20450","20550","20590","20800",

    # FOREIGN OWNERSHIP
    "20001","20002","20003","20004",

    # LONG-TERM DEBT
    "21300","21600","22000","22100",
    "22200","22500","22600","22800","22900"]

valutaer = ["NOK","EUR","USD"]
land = ["NO","SE","DK","US"]
perioder = ["202103","202106","202309","202312"]

rows = []

for _ in range(2000):
    post = random.choice(poster)

    under_post = "00"

    org_nummer = str(random.randint(100000000,999999999))

    # ✅ IMPORTANT: keep as NUMBER (not string)
    verdi = round(random.uniform(0,100_000_000),2)

    periode = random.choice(perioder)
    landkode = random.choice(land)
    valuta = random.choice(valutaer)

    rows.append({
        "post": post,
        "under_post": under_post,
        "org_nummer": org_nummer,
        "verdi": verdi,
        "periode": periode,
        "land": landkode,
        "valuta": valuta
    })

df = pd.DataFrame(rows)

print(df.head())
# -

verdi = round(random.uniform(0,100_000_000),2)
#Keep values numeric#

#Create dataframe#
df = pd.DataFrame(rows)

#Convert dtype#
df["verdi"] = df["verdi"].astype(float)

# # Prepare ML features

# +
df["post_code"] = df["post"].astype(int)
df["periode_code"] = df["periode"].astype(int)

X = df[["verdi","post_code","periode_code"]]
# -

# # Apply ML model

# +
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05)

df["anomaly"] = model.fit_predict(X)
# -

print(df.head())

# # Sort data by period

df = df.sort_values(by="periode")

# # Line plot

# +
plt.figure(figsize=(10,6))

# Plot all data (faint lines)
for post in df["post"].unique():
    subset = df[df["post"] == post]
    plt.plot(subset["periode"], subset["verdi"], alpha=0.3)

# Highlight anomalies
anomaly = df[df["anomaly"] == -1]
plt.scatter(anomaly["periode"], anomaly["verdi"], color="red", s=50, label="Anomaly")

plt.title("Financial Trends with Anomalies")
plt.xlabel("Periode")
plt.ylabel("Verdi")
plt.legend()

plt.show()

# +
post_example = "20000"

subset = df[df["post"] == post_example]

plt.figure(figsize=(8,5))

plt.plot(subset["periode"], subset["verdi"], marker="o", label="Trend")
plt.scatter(subset[subset["anomaly"]==-1]["periode"],
            subset[subset["anomaly"]==-1]["verdi"],
            color="red", label="Anomaly", s=80)

plt.title(f"Post {post_example} over time")
plt.xlabel("Periode")
plt.ylabel("Verdi")
plt.legend()

plt.show()
# -

for post in ["20000","20590","22100"]:
    subset = df[df["post"] == post]

    plt.figure(figsize=(6,4))

    plt.plot(subset["periode"], subset["verdi"], marker="o")

    plt.scatter(subset[subset["anomaly"]==-1]["periode"],
                subset[subset["anomaly"]==-1]["verdi"],
                color="red")

    plt.title(f"Post {post}")
    plt.show()

subset = df[df["post"] == post_example]


