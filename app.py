import base64
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. PAGE CONFIG & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="KL Property Price Predictor", page_icon="🏙️", layout="wide"
)

st.markdown(
    """
    <style>
        .stButton>button, .stSelectbox, .stNumberInput>div>input {
            border-radius: 8px;
        }
        .stMetric label {
            font-size: 16px;
            font-weight: 600;
        }
        h1 {
            text-align: center;
            font-weight: 700;
            padding-bottom: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 12px 20px;
            font-size: 16px;
        }
        .block-container {
            padding-top: 30px;
            background-color: rgba(255, 255, 255, 0.25);
            border-radius: 10px;
        }
        h1, h2, h3, h4, h5, h6, p, label {
            color: white;
            text-shadow: 1px 1px 4px black;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Load background image safely if present
if os.path.exists("kl.jpeg"):
    with open("kl.jpeg", "rb") as f:
        img_bytes = f.read()
    b64_img = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{b64_img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stAppViewContainer"] > .main {{
        background-color: transparent;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

st.title("🏙️ Kuala Lumpur Condominium Price Intelligence Dashboard")
st.write(
    "Estimate real estate market values in Kuala Lumpur using your trained **Random Forest Regressor** model."
)

# ==========================================
# 2. LOAD MODEL & DATASET
# ==========================================
MODEL_PATH = "best_rf_model.pkl"
COLUMNS_PATH = "model_columns.pkl"
DATA_PATH = "df_cleaned_rf_trained.csv"


@st.cache_resource
def load_local_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        return (None, None)
    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_columns = joblib.load(COLUMNS_PATH)
        return (loaded_model, loaded_columns)
    except Exception:
        return (None, None)


assets = load_local_assets()

if assets is None or not isinstance(assets, tuple) or assets[0] is None:
    st.error(
        f"❌ **Failed to load model files!**\n\n"
        f"Please verify that `{MODEL_PATH}` and `{COLUMNS_PATH}` exist in folder:\n"
        f"`{os.getcwd()}`"
    )
    st.stop()

model, model_columns = assets

# Load dataset for visualizations
if os.path.exists(DATA_PATH):
    data = pd.read_csv(DATA_PATH)
else:
    st.error(f"❌ Dataset file `{DATA_PATH}` not found in `{os.getcwd()}`.")
    st.stop()

# ==================================================
# UN-LOG TRANSFORMS FOR DISPLAY & CHARTS
# ==================================================
if "PRICE_LOG" in data.columns:
    data["price"] = np.expm1(data["PRICE_LOG"])
elif "PRICE" in data.columns:
    data["price"] = data["PRICE"]

if "BUILT-UP_LOG" in data.columns:
    data["size_sqft"] = np.expm1(data["BUILT-UP_LOG"])
elif "BUILT-UP" in data.columns:
    data["size_sqft"] = data["BUILT-UP"]

if "LOCATION" in data.columns:
    data["location_clean"] = data["LOCATION"]
else:
    loc_cols = [c for c in data.columns if str(c).startswith("LOCATION_")]
    if loc_cols:
        data["location_clean"] = (
            data[loc_cols].idxmax(axis=1).str.replace("LOCATION_", "")
        )

furnish_col = (
    "FURNISHING _STATUS"
    if "FURNISHING _STATUS" in data.columns
    else "FURNISHING_STATUS"
)

df_vis = data.copy()

# Upper-cased coordinate dictionary mapping
location_coordinates = {
    "KLCC": (3.1573, 101.7122),
    "BUKIT BINTANG": (3.1468, 101.7113),
    "BUKIT JALIL": (3.0587, 101.6917),
    "MONT KIARA": (3.1685, 101.6512),
    "DESA PARKCITY": (3.1862, 101.6299),
    "DAMANSARA HEIGHTS": (3.1491, 101.6534),
    "BANGSAR": (3.1290, 101.6798),
    "BANGSAR SOUTH": (3.1107, 101.6664),
    "AMPANG": (3.1665, 101.7483),
    "AMPANG HILIR": (3.1540, 101.7446),
    "CHERAS": (3.1068, 101.7229),
    "SETAPAK": (3.2088, 101.7278),
    "WANGSA MAJU": (3.2038, 101.7367),
    "SEGAMBUT": (3.1917, 101.6734),
    "SENTUL": (3.1823, 101.6888),
    "JALAN KUCHING": (3.1922, 101.6721),
    "JALAN IPOH": (3.1752, 101.6866),
    "JALAN SULTAN ISMAIL": (3.1552, 101.7054),
    "PANTAI": (3.1197, 101.6669),
    "TAMAN DESA": (3.1030, 101.6845),
    "SRI PETALING": (3.0684, 101.6856),
    "BUKIT TUNKU (KENNY HILLS)": (3.1665, 101.6828),
    "SRI HARTAMAS": (3.1600, 101.6520),
    "TAMAN TUN DR ISMAIL": (3.1461, 101.6255),
    "BRICKFIELDS": (3.1292, 101.6861),
    "KL SENTRAL": (3.1342, 101.6861),
    "KL CITY": (3.1499, 101.6945),
    "CITY CENTRE": (3.1517, 101.6942),
    "SEPUTEH": (3.1150, 101.6797),
    "PANDAN PERDANA": (3.1172, 101.7421),
    "TAMAN MELAWATI": (3.2126, 101.7471),
    "TITIWANGSA": (3.1774, 101.7077),
    "KEPONG": (3.2140, 101.6356),
    "BANDAR MENJALARA": (3.1939, 101.6309),
    "KUCHAI LAMA": (3.0839, 101.6883),
    "JINJANG": (3.2091, 101.6560),
    "BANDAR TASIK SELATAN": (3.0720, 101.7096),
    "SALAK SELATAN": (3.1049, 101.7055),
    "SUNGAI BESI": (3.0574, 101.7179),
    "KERAMAT": (3.1689, 101.7277),
    "COUNTRY HEIGHTS DAMANSARA": (3.1784, 101.6211),
}

if "location_clean" in data.columns:
    data["lat"] = data["location_clean"].map(
        lambda x: location_coordinates.get(str(x).upper(), (None, None))[0]
    )
    data["lon"] = data["location_clean"].map(
        lambda x: location_coordinates.get(str(x).upper(), (None, None))[1]
    )

# Extract options for dropdowns
available_locations = (
    sorted(data["location_clean"].dropna().unique().tolist())
    if "location_clean" in data.columns
    else ["KLCC", "MONT KIARA", "CHERAS"]
)
available_bases = (
    sorted(data["PROPERTY_BASE"].dropna().unique().tolist())
    if "PROPERTY_BASE" in data.columns
    else ["HIGH-RISE", "LANDED"]
)
available_types = (
    sorted(data["PROPERTY_TYPE"].dropna().unique().tolist())
    if "PROPERTY_TYPE" in data.columns
    else ["CONDOMINIUM", "SERVICED RESIDENCE"]
)
available_furnishings = (
    sorted(data[furnish_col].dropna().unique().tolist())
    if furnish_col in data.columns
    else ["FULLY FURNISHED", "PARTLY FURNISHED", "UNFURNISHED"]
)

# ==========================================
# 3. INPUT FORM
# ==========================================
st.markdown("### 🏠 Enter Property Details")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Specifications")
    size = st.number_input(
        "BUILT-UP (sq ft)", min_value=100.0, max_value=10000.0, value=1000.0, step=50.0
    )
    rooms = st.number_input("ROOMS", min_value=1, max_value=10, value=3)
    baths = st.number_input("BATHROOMS", min_value=1, max_value=10, value=2)
    carparks = st.number_input("CAR_PARKS", min_value=0, max_value=6, value=1)

    location = st.selectbox("LOCATION", available_locations)
    property_base = st.selectbox("PROPERTY_BASE", available_bases)
    property_type = st.selectbox("PROPERTY_TYPE", available_types)
    furnishing = st.selectbox("FURNISHING_STATUS", available_furnishings)

    actual_price = st.number_input(
        "Actual Market Price (RM)",
        min_value=10000.0,
        step=10000.0,
        value=500000.0,
    )

# Compute log value of built-up area for model input
built_up_log = np.log1p(size)

# Construct raw input DataFrame safely typed as object to prevent pandas casting warnings
input_df = pd.DataFrame(columns=[
    "ROOMS", "BATHROOMS", "CAR_PARKS", "BUILT-UP_LOG", 
    "LOCATION", "PROPERTY_BASE", "PROPERTY_TYPE", furnish_col
]).astype(object)

input_df.loc[0] = [
    rooms, baths, carparks, built_up_log, 
    location, property_base, property_type, furnishing
]

# Run model prediction (Pipeline handles One-Hot Encoding internally)
pred_raw = model.predict(input_df)[0]
predicted_price = float(np.expm1(pred_raw)) if pred_raw < 25 else float(pred_raw)

# ==========================================
# 4. MARKET INSIGHTS & COMPARABLES
# ==========================================
filtered_similar = pd.DataFrame()
if "location_clean" in data.columns and "size_sqft" in data.columns:
    filtered_similar = data[
        (data["location_clean"] == location)
        & (data["size_sqft"].between(size * 0.8, size * 1.2))
    ].copy()

    if filtered_similar.empty:
        filtered_similar = data[data["location_clean"] == location].copy()

user_pps = actual_price / size if size > 0 else 0
if not filtered_similar.empty:
    filtered_similar["price_per_sqft"] = (
        filtered_similar["price"] / filtered_similar["size_sqft"]
    )

with col2:
    st.markdown("#### Price Evaluation Result")
    st.metric("Predicted Price (RM)", f"RM {predicted_price:,.2f}")
    st.metric("Actual Asking Price (RM)", f"RM {actual_price:,.2f}")

    diff_pct = ((actual_price - predicted_price) / predicted_price) * 100
    if actual_price < predicted_price:
        st.success(
            f"✅ Undervalued by {abs(diff_pct):.1f}% compared to model predictions."
        )
    else:
        st.error(
            f"❗ Overvalued by {diff_pct:.1f}% compared to model predictions."
        )

    st.markdown("---")
    st.markdown("#### 📊 Comparable Unit Insights")
    if not filtered_similar.empty:
        avg_price = filtered_similar["price"].mean()
        avg_pps = filtered_similar["price_per_sqft"].mean()
        st.write(f"- **Similar Properties Found**: {len(filtered_similar)}")
        st.write(f"- **Average Comparable Price**: RM {avg_price:,.2f}")
        st.write(f"- **Average Location Rate/Sqft**: RM {avg_pps:,.2f}")
        st.write(
            f"- **Your Rate/Sqft**: RM {user_pps:,.2f} "
            f"({'Below Average' if user_pps < avg_pps else 'Above Average'})"
        )

# ==========================================
# 5. VISUAL ANALYTICS TABS
# ==========================================
st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📍 Location Pricing",
        "🏗️ Size vs Price",
        "📊 Feature Importance",
        "📈 Market Distribution",
        "🗺️ Kuala Lumpur Map",
        "💰 Price per Sqft Analysis",
    ]
)

px.defaults.template = "plotly_white"

# Tab 1: Location Price Bar Chart
with tab1:
    if "location_clean" in df_vis.columns:
        loc_df = (
            df_vis.groupby("location_clean")["price"].mean().reset_index()
        )
        loc_df.columns = ["Location", "Average Price (RM)"]
        loc_df = loc_df.sort_values("Average Price (RM)", ascending=False)
        loc_df["Selected"] = loc_df["Location"].apply(
            lambda x: "Selected" if x == location else "Other"
        )

        fig = px.bar(
            loc_df,
            x="Location",
            y="Average Price (RM)",
            color="Selected",
            color_discrete_map={"Selected": "red", "Other": "gray"},
            title=f"Location Average Prices (Selected Location: {location})",
            text_auto=".2s",
        )
        fig.update_layout(xaxis_tickangle=45)
        fig.add_hline(
            y=predicted_price,
            line_dash="dash",
            line_color="blue",
            annotation_text="Your Predicted Price",
        )
        st.plotly_chart(fig, width="stretch")

# Tab 2: Size vs Price
with tab2:
    if not filtered_similar.empty:
        fig = px.scatter(
            filtered_similar,
            x="size_sqft",
            y="price",
            opacity=0.6,
            title=f"Built-Up Area (sqft) vs Price ({location})",
            labels={"size_sqft": "Size (sqft)", "price": "Price (RM)"},
        )
        fig.add_trace(
            go.Scatter(
                x=[size],
                y=[predicted_price],
                mode="markers",
                marker=dict(size=14, color="red", symbol="star"),
                name="Your Property",
            )
        )
        st.plotly_chart(fig, width="stretch")

# Tab 3: Feature Importance Bar Chart (CLEAN LABELS)
with tab3:
    fitted_model = (
        model.named_steps["model"] if hasattr(model, "named_steps") else model
    )

    if hasattr(fitted_model, "feature_importances_"):
        importances = fitted_model.feature_importances_

        feature_names = None
        if hasattr(model, "named_steps"):
            for step_name, step_obj in model.named_steps.items():
                if hasattr(step_obj, "get_feature_names_out"):
                    try:
                        feature_names = step_obj.get_feature_names_out()
                        break
                    except Exception:
                        pass

        if feature_names is None or len(feature_names) != len(importances):
            feature_names = (
                model_columns
                if len(model_columns) == len(importances)
                else [f"Feature {i}" for i in range(len(importances))]
            )

        clean_names = []
        for name in feature_names:
            c_name = str(name).replace("num__", "").replace("cat__", "")
            c_name = c_name.replace("_LOG", " (Log)").replace("_", " ")
            clean_names.append(c_name.strip())

        feat_imp = (
            pd.Series(importances, index=clean_names)
            .sort_values(ascending=False)
            .head(15)
        )

        fig = px.bar(
            feat_imp,
            x=feat_imp.values,
            y=feat_imp.index,
            orientation="h",
            title="Top 15 Most Important Predictor Features",
            labels={"x": "Importance Score", "index": "Feature Name"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Feature importance is not supported for this estimator.")

# Tab 4: Price Distribution Histogram
with tab4:
    if not filtered_similar.empty:
        fig = px.histogram(
            filtered_similar,
            x="price",
            nbins=30,
            title=f"Price Distribution for Comparable Properties in {location}",
        )
        fig.add_vline(
            x=predicted_price,
            line_dash="dash",
            line_color="red",
            annotation_text="Your Predicted Price",
        )
        st.plotly_chart(fig, width="stretch")

# Tab 5: Map Visual
with tab5:
    if "lat" in data.columns and "lon" in data.columns:
        map_df = (
            data.groupby("location_clean", as_index=False)
            .agg(
                avg_price=("price", "mean"),
                lat=("lat", "first"),
                lon=("lon", "first"),
            )
            .dropna(subset=["lat", "lon"])
        )

        center_lat, center_lon = location_coordinates.get(
            location.upper(), (3.1390, 101.6869)
        )

        fig = px.scatter_map(
            map_df,
            lat="lat",
            lon="lon",
            size="avg_price",
            color="avg_price",
            hover_name="location_clean",
            zoom=12,
            height=600,
        )
        fig.update_layout(
            map_style="carto-positron",
            map_center={"lat": center_lat, "lon": center_lon},
        )
        st.plotly_chart(fig, width="stretch")

# Tab 6: Price per Sqft Box Plot
with tab6:
    if not filtered_similar.empty and "price_per_sqft" in filtered_similar.columns:
        fig = px.box(
            filtered_similar,
            y="price_per_sqft",
            title=f"Price per Sqft Variance in {location}",
            labels={"price_per_sqft": "Price per Sqft (RM)"},
        )
        fig.add_hline(
            y=user_pps,
            line_dash="dash",
            line_color="blue",
            annotation_text="Your Listing Rate",
        )
        st.plotly_chart(fig, width="stretch")
