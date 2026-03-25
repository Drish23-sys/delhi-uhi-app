# 🌡️ Delhi Urban Heat Island Predictor

A Streamlit web app that classifies Urban Heat Island intensity zones from
Landsat 9 GeoTIFF imagery using a trained XGBoost model (94.01% accuracy).

---

## How it works

1. User uploads a Landsat 9 GeoTIFF (7 bands: B2–B7 + B10)
2. App computes 53 multi-scale spatial features per pixel
3. XGBoost Stage 5 model classifies each pixel as Cool / Moderate / Hot
4. Colour-coded UHI map is displayed with pixel-count statistics
5. User can download the output map as PNG

---

## Deploy in 5 Steps

### Step 1 — Get your Google Drive model file ID

1. Open Google Drive → find `Delhi_UHI_XGB_Stage5.pkl`
2. Right-click → **Share** → change to "Anyone with the link can view"
3. Copy the URL: `https://drive.google.com/file/d/FILE_ID_HERE/view`
4. Save that `FILE_ID_HERE` — you'll need it in Step 4

### Step 2 — Create a Google Cloud Service Account

1. Go to https://console.cloud.google.com
2. Create a new project (or use existing)
3. Enable the **Google Drive API**:
   - APIs & Services → Library → search "Google Drive API" → Enable
4. Create a Service Account:
   - APIs & Services → Credentials → Create Credentials → Service Account
   - Name it anything (e.g. `uhi-app-reader`)
   - Role: **Viewer** is enough
5. Download the JSON key:
   - Click the service account → Keys → Add Key → Create new key → JSON
   - Save the downloaded `.json` file

### Step 3 — Push to GitHub

```bash
# In your terminal
git init
git add app.py requirements.txt .gitignore
git commit -m "Initial UHI app"
git remote add origin https://github.com/YOUR_USERNAME/delhi-uhi-app.git
git push -u origin main
```

> ⚠️ Never commit `.streamlit/secrets.toml` — it's in `.gitignore`

### Step 4 — Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **New app** → select your repo → branch: `main` → file: `app.py`
4. Click **Advanced settings** → **Secrets** → paste the following:

```toml
model_file_id = "PASTE_FILE_ID_FROM_STEP_1"

[gcp_service_account]
type                        = "service_account"
project_id                  = "your-project-id"
private_key_id              = "abc123"
private_key                 = "-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n"
client_email                = "uhi-app-reader@your-project.iam.gserviceaccount.com"
client_id                   = "123456789"
auth_uri                    = "https://accounts.google.com/o/oauth2/auth"
token_uri                   = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url        = "https://www.googleapis.com/robot/v1/metadata/x509/..."
```

> Copy all values from the JSON key file you downloaded in Step 2.
> The `private_key` field: replace all newlines with `\n`.

5. Click **Deploy** — your app will be live at:
   `https://YOUR_USERNAME-delhi-uhi-app-app-XXXX.streamlit.app`

### Step 5 — Test it

Upload any Landsat 9 GeoTIFF with 7 bands (B2, B3, B4, B5, B6, B7, B10).
You can export one directly from Google Earth Engine:

```javascript
// In GEE Code Editor
var image = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
  .filterDate("2023-05-01", "2023-08-31")
  .filterBounds(delhi)
  .sort("CLOUD_COVER")
  .first()
  .select(["SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7","ST_B10"]);

Export.image.toDrive({
  image: image,
  description: "Delhi_L9_2023",
  scale: 30,
  region: delhi,
  fileFormat: "GeoTIFF"
});
```

---

## Run locally

```bash
pip install -r requirements.txt

# create .streamlit/secrets.toml with your credentials (see template)
mkdir -p .streamlit
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
# edit secrets.toml with your actual values

streamlit run app.py
```

---

## Model details

| Item | Value |
|---|---|
| Algorithm | XGBoost (tree_method=hist, GPU) |
| Features | 53 multi-scale spatial |
| Training pixels | 144,401 |
| Test accuracy | 94.01% |
| Classes | Cool (0) · Moderate (1) · Hot (2) |
| Study area | Delhi NCT, India |
| Imagery | Landsat 9 OLI/TIRS, Summer 2023 |
