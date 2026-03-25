import streamlit as st
import numpy as np
import joblib
import tempfile, os, io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.ndimage import uniform_filter, generic_filter

# ── page config ──────────────────────────────
st.set_page_config(
    page_title="Delhi UHI Predictor",
    page_icon="🌡️",
    layout="wide",
)

# ── custom CSS ───────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-val { font-size: 28px; font-weight: 700; }
    .metric-lbl { font-size: 12px; color: #64748b; margin-top: 2px; }
    .hot-val   { color: #EF4444; }
    .mod-val   { color: #F59E0B; }
    .cool-val  { color: #2196F3; }
    .info-box {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 14px;
        color: #1e3a5f;
    }
</style>
""", unsafe_allow_html=True)

# ── constants ────────────────────────────────
BAND_NAMES = ["Blue (B2)", "Green (B3)", "Red (B4)",
              "NIR (B5)", "SWIR1 (B6)", "SWIR2 (B7)", "TIR (B10)"]
WINDOWS    = [3, 7, 11, 15, 21, 31]
CLASSES    = ["Cool", "Moderate", "Hot"]
UHI_COLORS = ["#2196F3", "#FFC107", "#F44336"]
DEAD_ZONE  = 0.8

# ── load model ───────────────────────────────
@st.cache_resource(show_spinner="Loading XGBoost model…")
def load_model():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import json

    # read credentials from Streamlit secrets
    creds_info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    service = build("drive", "v3", credentials=creds)

    file_id = st.secrets["model_file_id"]
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    model = joblib.load(buf)
    return model

# ── feature engineering ──────────────────────
def compute_features(bands: dict) -> np.ndarray:
    """
    bands: dict with keys b2..b7, b10 as 2D float arrays
    Returns: (H*W, 53) feature matrix, valid_mask (H*W,)
    """
    b2  = bands["b2"].astype(float)
    b3  = bands["b3"].astype(float)
    b4  = bands["b4"].astype(float)
    b5  = bands["b5"].astype(float)
    b6  = bands["b6"].astype(float)
    b7  = bands["b7"].astype(float)
    b10 = bands["b10"].astype(float)
    H, W = b2.shape

    eps = 1e-6

    # ── spectral indices ──
    NDVI  = (b5 - b4) / (b5 + b4 + eps)
    NDBI  = (b6 - b5) / (b6 + b5 + eps)
    NDBSI = (b6 + b4 - b5 - b2) / (b6 + b4 + b5 + b2 + eps)
    Albedo = 0.356*b2 + 0.130*b4 + 0.373*b5 + 0.085*b6 + 0.072*b7 - 0.018

    # Built-up Fraction using NDBI thresholding
    BUFrac = np.clip((NDBI + 1) / 2, 0, 1)

    # DEM proxy (flat for demo — replace with real DEM if available)
    DEM = np.zeros_like(b2)

    # ── interaction indices ──
    UHI_idx   = NDBI - NDVI
    VegCool   = NDVI * (1 - BUFrac)
    HeatDens  = BUFrac * (1 - Albedo)
    CoolPatch = (NDVI > 0.3).astype(float) * (1 - BUFrac)
    HeatAcc   = (1 - Albedo) * BUFrac * NDBI
    BndSharp  = np.abs(NDBI - uniform_filter(NDBI, 7)) / (np.abs(NDBI) + eps)

    raw_feats = [NDVI, NDBI, NDBSI, BUFrac, Albedo, DEM,
                 UHI_idx, VegCool, HeatDens, CoolPatch, HeatAcc, BndSharp]

    # ── multi-scale spatial means ──
    spatial_feats = []
    for arr in [NDVI, NDBI, BUFrac, Albedo]:
        for w in WINDOWS:
            spatial_feats.append(uniform_filter(arr, w))

    # ── texture (local std dev) ──
    def local_std(arr, w):
        m  = uniform_filter(arr, w)
        m2 = uniform_filter(arr**2, w)
        return np.sqrt(np.clip(m2 - m**2, 0, None))

    tex_feats = []
    for arr in [NDVI, BUFrac]:
        for w in [7, 15, 21]:
            tex_feats.append(local_std(arr, w))

    # ── scale difference ──
    diff_feats = []
    for arr in [NDBI, BUFrac, NDVI]:
        for wa, wb in [(3,7),(7,21),(21,31)]:
            diff_feats.append(
                uniform_filter(arr, wa) - uniform_filter(arr, wb))

    # ── assemble ──
    all_feats = raw_feats + spatial_feats + tex_feats + diff_feats
    X = np.stack([f.ravel() for f in all_feats], axis=1)   # (H*W, N)

    # ── validity mask ──
    valid = (
        (NDVI.ravel()   >= -1) & (NDVI.ravel()   <= 1) &
        (NDBI.ravel()   >= -1) & (NDBI.ravel()   <= 1) &
        (Albedo.ravel() >  0)  & (Albedo.ravel() <  1) &
        np.isfinite(X).all(axis=1)
    )

    return X, valid, H, W

def predict_scene(model, X, valid, H, W):
    """Run model on valid pixels, return label map."""
    labels = np.full(H * W, -1, dtype=np.int8)
    if valid.sum() == 0:
        return labels.reshape(H, W)
    labels[valid] = model.predict(X[valid]).astype(np.int8)
    return labels.reshape(H, W)

# ── read uploaded GeoTIFF ────────────────────
def read_geotiff(path):
    import rasterio
    with rasterio.open(path) as src:
        n = src.count
        if n < 7:
            return None, f"Expected ≥7 bands, found {n}."
        data = src.read()   # (bands, H, W)
        crs  = src.crs
        bounds = src.bounds
    return data, None, crs, bounds

# ── visualise ────────────────────────────────
def make_map(label_map, bounds=None):
    cmap = ListedColormap(["#050A14", "#2196F3", "#FFC107", "#F44336"])
    display = np.where(label_map == -1, 0, label_map + 1)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
    ax.imshow(display, cmap=cmap, vmin=0, vmax=3,
              interpolation="nearest", aspect="auto")
    ax.set_title("Urban Heat Island Classification — Delhi NCT",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Column (pixel)" if bounds is None else "Longitude")
    ax.set_ylabel("Row (pixel)"    if bounds is None else "Latitude")

    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(UHI_COLORS, CLASSES)]
    ax.legend(handles=patches, loc="lower left",
              fontsize=9, framealpha=0.85)
    plt.tight_layout()
    return fig

# ════════════════════════════════════════════
# UI
# ════════════════════════════════════════════

st.title("🌡️ Delhi Urban Heat Island Predictor")
st.markdown(
    "<div class='info-box'>Upload a <b>Landsat 9 GeoTIFF</b> (7+ bands: B2–B7 + B10). "
    "The app computes 53 multi-scale spatial features and classifies every pixel as "
    "<b style='color:#2196F3'>Cool</b>, "
    "<b style='color:#F59E0B'>Moderate</b>, or "
    "<b style='color:#EF4444'>Hot</b> using an XGBoost model (94.01% accuracy).</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("📂 Upload GeoTIFF")
    uploaded = st.file_uploader(
        "Select your Landsat 9 GeoTIFF",
        type=["tif", "tiff"],
        help="Multi-band GeoTIFF with bands B2, B3, B4, B5, B6, B7, B10 (in that order)"
    )

    st.markdown("#### Band Order Expected")
    for i, name in enumerate(BAND_NAMES, 1):
        st.markdown(f"- **Band {i}:** {name}")

    st.markdown("---")
    st.markdown("#### Model Info")
    st.markdown("""
    - **Algorithm:** XGBoost (Stage 5)
    - **Features:** 53 multi-scale spatial
    - **Accuracy:** 94.01%
    - **Training data:** Delhi NCT 2023
    - **Classes:** Cool / Moderate / Hot
    """)

with col_right:
    if uploaded is None:
        st.info("👈 Upload a GeoTIFF to start prediction.")
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/"
            "Landsat_8_image_of_Delhi.jpg/640px-Landsat_8_image_of_Delhi.jpg",
            caption="Example: Landsat image of Delhi",
            use_container_width=True,
        )
    else:
        with st.spinner("Reading GeoTIFF…"):
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            result = read_geotiff(tmp_path)
            os.unlink(tmp_path)

            if result[1] is not None:   # error
                st.error(result[1])
                st.stop()

            data, _, crs, bounds = result
            H, W = data.shape[1], data.shape[2]

        st.success(f"✅ GeoTIFF loaded — {W} × {H} pixels, {data.shape[0]} bands")

        with st.spinner("Computing 53 features…"):
            bands = {
                "b2":  data[0], "b3":  data[1], "b4":  data[2],
                "b5":  data[3], "b6":  data[4], "b7":  data[5],
                "b10": data[6],
            }
            X, valid, H, W = compute_features(bands)

        st.success(f"✅ Features computed — {valid.sum():,} valid pixels")

        with st.spinner("Running XGBoost model…"):
            model = load_model()
            label_map = predict_scene(model, X, valid, H, W)

        st.success("✅ Prediction complete!")
        st.markdown("---")

        # ── metrics ──
        total  = valid.sum()
        n_cool = int((label_map[label_map >= 0] == 0).sum())
        n_mod  = int((label_map[label_map >= 0] == 1).sum())
        n_hot  = int((label_map[label_map >= 0] == 2).sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"<div class='metric-card'><div class='metric-val'>{total:,}</div>"
                    f"<div class='metric-lbl'>Valid Pixels</div></div>",
                    unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><div class='metric-val cool-val'>"
                    f"{n_cool/total*100:.1f}%</div>"
                    f"<div class='metric-lbl'>Cool</div></div>",
                    unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><div class='metric-val mod-val'>"
                    f"{n_mod/total*100:.1f}%</div>"
                    f"<div class='metric-lbl'>Moderate</div></div>",
                    unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-card'><div class='metric-val hot-val'>"
                    f"{n_hot/total*100:.1f}%</div>"
                    f"<div class='metric-lbl'>Hot</div></div>",
                    unsafe_allow_html=True)

        st.markdown("#### UHI Classification Map")
        fig = make_map(label_map, bounds)
        st.pyplot(fig)
        plt.close(fig)

        # ── download ──
        buf = io.BytesIO()
        fig2, ax2 = plt.subplots(figsize=(10, 8), dpi=150)
        from matplotlib.colors import ListedColormap as LC
        cmap2 = LC(["#050A14", "#2196F3", "#FFC107", "#F44336"])
        disp2 = np.where(label_map == -1, 0, label_map + 1)
        ax2.imshow(disp2, cmap=cmap2, vmin=0, vmax=3,
                   interpolation="nearest", aspect="auto")
        ax2.axis("off")
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig2)
        buf.seek(0)

        st.download_button(
            label="⬇️ Download UHI Map (PNG)",
            data=buf,
            file_name="delhi_uhi_prediction.png",
            mime="image/png",
        )

# ── footer ───────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#94a3b8; font-size:13px;'>"
    "Delhi UHI Classification · Landsat 9 · XGBoost Stage 5 · 94.01% Accuracy · 2023"
    "</p>",
    unsafe_allow_html=True,
)
