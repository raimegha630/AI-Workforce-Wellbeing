# app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import os
import random
import io
from sqlalchemy.exc import OperationalError
from utils import save_uploaded_image, ensure_imagepath_column

# Try to import OpenCV for webcam-based face/emotion detection. If unavailable, we'll fall back to manual input.
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# Try to import DeepFace for multi-class emotion recognition. If unavailable, we'll fall back to OpenCV heuristics.
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except Exception:
    HAS_DEEPFACE = False
# -------------------- CONFIG --------------------
st.set_page_config(page_title="🌿 AI WellTrack Dashboard", layout="wide")
CSV_PATH = "employee_30day_wellbeing.csv"
DB_PATH = "sqlite:///welltrack_database.db"
TABLE_NAME = "employee_wellbeing"
UPLOAD_DIR = "uploads"

# Ensure upload dir exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------- DB CONNECTION --------------------
engine = create_engine(DB_PATH)

# Ensure ImagePath column exists in CSV/SQL (migration helper)
try:
    ensure_imagepath_column(CSV_PATH, engine, TABLE_NAME)
except Exception:
    # If migration fails, continue; get_data will surface missing data issues
    pass

# -------------------- CSV + SQL SYNC --------------------
def sync_sql_and_csv():
    """Merge and sync CSV + SQL databases."""
    try:
        df_csv = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        st.error("CSV file not found.")
        return pd.DataFrame()

    try:
        df_sql = pd.read_sql_table(TABLE_NAME, con=engine)
    except Exception:
        df_sql = pd.DataFrame()

    if not df_sql.empty and not df_csv.empty:
        df_sql["Date"] = pd.to_datetime(df_sql["Date"], errors="coerce")
        df_csv["Date"] = pd.to_datetime(df_csv["Date"], errors="coerce")
        combined = pd.concat([df_sql, df_csv]).drop_duplicates(subset=["Employee", "Date"], keep="last")
    else:
        combined = df_csv if not df_csv.empty else df_sql

    combined.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)
    combined.to_csv(CSV_PATH, index=False)
    return combined


def detect_expression(image_bytes: bytes):
    """Detect a simple facial expression from image bytes.
    Uses OpenCV Haar cascades (face + smile) when available.
    Returns one of: 'Happy', 'Neutral' or None (if no face detected or model not available).
    This is intentionally lightweight and heuristic — for production use a proper emotion model.
    """
    # If DeepFace is available, use it for a richer emotion label
    if HAS_DEEPFACE:
        try:
            arr = np.frombuffer(image_bytes, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR) if HAS_CV2 else None
            if bgr is None:
                # Try using PIL as a fallback
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img_np = np.array(img)
            else:
                img_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            analysis = DeepFace.analyze(img_np, actions=["emotion"], enforce_detection=False)
            dominant = analysis.get("dominant_emotion") if isinstance(analysis, dict) else None
            if dominant:
                dom = dominant.lower()
                # Map DeepFace labels to our app labels
                if "happy" in dom:
                    return "Happy"
                if "sad" in dom:
                    return "Sad"
                if "angry" in dom:
                    return "Angry"
                if "surprise" in dom or "surprised" in dom:
                    return "Surprised"
                if "neutral" in dom or "calm" in dom:
                    return "Neutral"
                # catch-all
                return "Neutral"
        except Exception:
            # If DeepFace fails for any reason, fall through to the cv2 heuristic
            pass

    # Fallback: use OpenCV Haar-based smile detection if available
    if not HAS_CV2:
        return None

    try:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None

        # For the first detected face, check for a smile inside the face ROI
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
        if len(smiles) > 0:
            return "Happy"
        else:
            return "Neutral"
    except Exception:
        return None


def save_uploaded_image(image_bytes: bytes, prefix: str = "capture") -> str:
    """Save image bytes to the uploads directory and return the relative path."""
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{ts}.jpg"
        path = os.path.join(UPLOAD_DIR, filename)
        with open(path, "wb") as f:
            f.write(image_bytes)
        return path
    except Exception:
        return ""

@st.cache_data(ttl=86400)
def get_data():
    """Load data and sync daily."""
    return sync_sql_and_csv()

data = get_data()
if data.empty:
    st.warning("No data found. Please check CSV file.")
    st.stop()

# -------------------- SIDEBAR LOGIN --------------------
st.sidebar.title("🔐 Login Portal")
role = st.sidebar.radio("Select Role", ["Employee", "Manager"])

# DeepFace availability banner and quick install instructions in sidebar
if HAS_DEEPFACE:
    st.sidebar.success("DeepFace available — using it for emotion detection")
else:
    st.sidebar.warning("DeepFace not installed — using OpenCV heuristic or manual selection")
    if st.sidebar.button("Show DeepFace install instructions"):
        st.sidebar.code("pip install tensorflow; pip install deepface")

# ==========================================================
# EMPLOYEE VIEW
# ==========================================================
if role == "Employee":
    emp_name = st.sidebar.selectbox("Select your name:", sorted(data["Employee"].unique()))
    st.sidebar.success(f"✅ Logged in as {emp_name}")

    emp_data = data[data["Employee"] == emp_name].sort_values("Date")

    st.title(f"👤 {emp_name}'s Wellbeing Tracker")
    st.write("Submit your daily wellbeing details and view your progress trends.")
    st.subheader("📅 Daily Wellbeing Update of Hospital Staffs")


    def normalize_dates(df, date_col="Date"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        return df


    df_csv = pd.read_csv(CSV_PATH)
    df_csv = normalize_dates(df_csv, "Date")
    try:
        df_sql = pd.read_sql_table(TABLE_NAME, con=engine)
        df_sql = normalize_dates(df_sql, "Date")
    except Exception:
        df_sql = pd.DataFrame()

    combined = pd.concat([df_sql, df_csv]).drop_duplicates(subset=["Employee", "Date"], keep="last")
    combined = normalize_dates(combined, "Date")

    today_date = date.today()
    emp_data = combined[combined["Employee"] == emp_name]
    existing_today = emp_data[emp_data["Date"] == today_date]

    if not existing_today.empty:
        st.info("🟡 An entry for today already exists.")
        st.write("Existing entry for today:")
        st.dataframe(existing_today)

        st.write("You can overwrite or append a new record.")
        work_hours = st.number_input("Work Hours Today", min_value=1, max_value=12, value=int(existing_today["WorkHours"].iloc[0]))
        stress = st.slider("Stress Level (1-10)", 1, 10, int(existing_today["Stress"].iloc[0]))
        productivity = st.slider("Productivity Level (1-10)", 1, 10, int(existing_today["Productivity"].iloc[0]))

        # --- Camera-based expression detection with manual fallback ---
        expression = None
        use_cam = st.checkbox("Use webcam to auto-detect facial expression", key="use_cam_existing")
        image_path = ""
        if use_cam:
            img_file = st.camera_input("Take a photo for expression detection", key="cam_existing")
            if img_file is not None:
                image_bytes = img_file.getvalue()
                # save image and detect
                image_path = save_uploaded_image(image_bytes, prefix=emp_name.replace(" ", "_"))
                detected = detect_expression(image_bytes)
                if detected:
                    expression = detected
                    st.success(f"Detected expression: {expression}")
                else:
                    st.warning("No face/smiling mouth detected. Please select manually.")

        if expression is None:
            try:
                default_idx = ["Happy", "Neutral", "Sad", "Angry", "Calm", "Surprised"].index(existing_today["Expression"].iloc[0])
            except Exception:
                default_idx = 0
            expression = st.selectbox("Facial Expression", ["Happy", "Neutral", "Sad", "Angry", "Calm", "Surprised"], index=default_idx)

        emotion_scores = {"Happy": 0.2, "Neutral": 0.5, "Calm": 0.3, "Surprised": 0.6, "Sad": 0.8, "Angry": 0.9}
        facial_emotion_score = emotion_scores.get(expression, 0.5)

        burnout_risk = round(0.5 * (stress / 10) + 0.3 * facial_emotion_score + 0.2 * ((work_hours - 5) / 6), 3)
        predicted_future_burnout = round(min(1, burnout_risk + np.random.normal(0, 0.05)), 3)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔁 Overwrite Today's Entry"):
                combined = combined[~((combined["Employee"] == emp_name) & (combined["Date"] == today_date))]
                new_row = {
                    "Employee": emp_name,
                    "Department": combined[combined["Employee"] == emp_name]["Department"].iloc[0]
                    if "Department" in combined.columns and not combined[combined["Employee"] == emp_name].empty else "N/A",
                    "Day": int(len(combined[combined["Employee"] == emp_name]) + 1),
                    "Date": today_date,
                    "WorkHours": work_hours,
                    "Expression": expression,
                    "Stress": stress,
                    "Productivity": productivity,
                    "FacialEmotion_Score": facial_emotion_score,
                    "ImagePath": image_path,
                    "BurnoutRisk": burnout_risk,
                    "PredictedFutureBurnout": predicted_future_burnout
                }
                combined = pd.concat([combined, pd.DataFrame([new_row])], ignore_index=True)
                combined_to_write = combined.copy()
                combined_to_write["Date"] = combined_to_write["Date"].astype(str)
                combined_to_write.to_csv(CSV_PATH, index=False)
                combined_to_write.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)
                st.success("✅ Today's entry overwritten successfully!")

        with col_b:
            if st.button("➕ Append New Entry Anyway"):
                # If user used camera earlier in this branch, image_path was set; otherwise empty
                try:
                    image_path = image_path  # already set above if camera used
                except NameError:
                    image_path = ""

                new_row = {
                    "Employee": emp_name,
                    "Department": combined[combined["Employee"] == emp_name]["Department"].iloc[0]
                    if "Department" in combined.columns and not combined[combined["Employee"] == emp_name].empty else "N/A",
                    "Day": int(len(combined[combined["Employee"] == emp_name]) + 1),
                    "Date": today_date,
                    "WorkHours": work_hours,
                    "Expression": expression,
                    "Stress": stress,
                    "Productivity": productivity,
                    "FacialEmotion_Score": facial_emotion_score,
                    "ImagePath": image_path,
                    "BurnoutRisk": burnout_risk,
                    "PredictedFutureBurnout": predicted_future_burnout
                }
                combined = pd.concat([combined, pd.DataFrame([new_row])], ignore_index=True)
                combined_to_write = combined.copy()
                combined_to_write["Date"] = combined_to_write["Date"].astype(str)
                combined_to_write.to_csv(CSV_PATH, index=False)
                combined_to_write.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)
                st.success("✅ New entry appended successfully!")

    else:
        st.info("🆕 Submit your wellbeing data for today")
        work_hours = st.number_input("Work Hours Today", min_value=1, max_value=12, value=8)
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        productivity = st.slider("Productivity Level (1-10)", 1, 10, 7)

        # --- Camera-based expression detection with manual fallback ---
        expression = None
        use_cam = st.checkbox("Use webcam to auto-detect facial expression", key="use_cam_new")
        image_path = ""
        if use_cam:
            img_file = st.camera_input("Take a photo for expression detection", key="cam_new")
            if img_file is not None:
                image_bytes = img_file.getvalue()
                # save image and detect
                image_path = save_uploaded_image(image_bytes, prefix=emp_name.replace(" ", "_"))
                detected = detect_expression(image_bytes)
                if detected:
                    expression = detected
                    st.success(f"Detected expression: {expression}")
                else:
                    st.warning("No face/smiling mouth detected. Please select manually.")

        if expression is None:
            expression = st.selectbox("Facial Expression", ["Happy", "Neutral", "Sad", "Angry", "Calm", "Surprised"])

        emotion_scores = {"Happy": 0.2, "Neutral": 0.5, "Calm": 0.3, "Surprised": 0.6, "Sad": 0.8, "Angry": 0.9}
        facial_emotion_score = emotion_scores.get(expression, 0.5)

        burnout_risk = round(0.5 * (stress / 10) + 0.3 * facial_emotion_score + 0.2 * ((work_hours - 5) / 6), 3)
        predicted_future_burnout = round(min(1, burnout_risk + np.random.normal(0, 0.05)), 3)

        if st.button("💾 Save Today's Data"):
            # Ensure image_path variable exists even if camera not used
            try:
                image_path = image_path
            except NameError:
                image_path = ""

            new_row = {
                "Employee": emp_name,
                "Department": combined[combined["Employee"] == emp_name]["Department"].iloc[0]
                if "Department" in combined.columns and not combined[combined["Employee"] == emp_name].empty else "N/A",
                "Day": int(len(combined[combined["Employee"] == emp_name]) + 1),
                "Date": today_date,
                "WorkHours": work_hours,
                "Expression": expression,
                "Stress": stress,
                "Productivity": productivity,
                "FacialEmotion_Score": facial_emotion_score,
                "ImagePath": image_path,
                "BurnoutRisk": burnout_risk,
                "PredictedFutureBurnout": predicted_future_burnout
            }
            combined = pd.concat([combined, pd.DataFrame([new_row])], ignore_index=True)
            combined_to_write = combined.copy()
            combined_to_write["Date"] = combined_to_write["Date"].astype(str)
            combined_to_write.to_csv(CSV_PATH, index=False)
            combined_to_write.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)
            st.success("✅ Today's data saved successfully!")

    # -------------------- EMPLOYEE ANALYTICS --------------------
    st.markdown("---")
    st.subheader("📈 Your Wellbeing Trends")

    emp_data = combined[combined["Employee"] == emp_name].sort_values("Date")
    col1, col2, col3 = st.columns(3)
    col1.metric("📆 Days Tracked", len(emp_data))
    col2.metric("💼 Avg Work Hours", round(emp_data["WorkHours"].mean(), 1))
    col3.metric("🧠 Avg Stress", round(emp_data["Stress"].mean(), 2))

    st.line_chart(emp_data[["Stress", "BurnoutRisk", "PredictedFutureBurnout"]])
    st.area_chart(emp_data[["FacialEmotion_Score"]])
    st.bar_chart(emp_data[["Day", "WorkHours"]].set_index("Day"))

    latest = emp_data.iloc[-1]
    if latest["BurnoutRisk"] > 0.7:
        st.error("🚨 High Burnout Risk! Take rest.")
    elif latest["BurnoutRisk"] > 0.5:
        st.warning("⚠️ Moderate Burnout Risk — stay mindful.")
    else:
        st.success("🌿 Low Burnout Risk — great balance!")

    st.markdown("---")
    st.info("“Consistency in care leads to clarity in mind.” ✨")
    st.stop()

# ==========================================================
# MANAGER VIEW
# ==========================================================
st.title("💼 Manager Dashboard – Workforce Analytics")
st.write("Monitor stress, burnout, and wellbeing trends across employees.")

## Manager: Image preview (consent required)
st.markdown("---")
st.subheader("📸 Employee Photo Preview (Recent)")
st.info("Photos are shown for recent entries. Ensure employees consent to photo storage before viewing.")
try:
    pics = data[ (data.get("ImagePath", "") != "") & data["ImagePath"].notna() ]
    if not pics.empty:
        # show latest photo per employee (groupby Employee, take last)
        latest_pics = pics.sort_values("Date").groupby("Employee").tail(1)
        cols = st.columns(4)
        i = 0
        for _, row in latest_pics.iterrows():
            img_path = row.get("ImagePath", "")
            if img_path and os.path.exists(img_path):
                with cols[i % 4]:
                    st.image(img_path, width=200)
                    st.caption(f"{row.get('Employee')} — {row.get('Date')}")
            i += 1
    else:
        st.write("No stored photos found yet.")
except Exception:
    st.write("Error loading images (check ImagePath values).")

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Employees", data["Employee"].nunique())
col2.metric("💼 Avg Work Hours", round(data["WorkHours"].mean(), 1))
col3.metric("😟 Avg Stress", round(data["Stress"].mean(), 2))
col4.metric("🔥 Avg Burnout Risk", round(data["BurnoutRisk"].mean(), 2))

st.markdown("---")
st.subheader("📊 Average Burnout Trend (All Employees)")
avg_daily = data.groupby("Date")[["BurnoutRisk", "PredictedFutureBurnout"]].mean()
st.line_chart(avg_daily)

if "Department" in data.columns:
    st.subheader("🏢 Department-wise Burnout Risk")
    dept_avg = data.groupby("Department")["BurnoutRisk"].mean()
    st.bar_chart(dept_avg)

st.subheader("📈 Stress vs Productivity")
fig, ax = plt.subplots()
ax.scatter(data["Stress"], data["Productivity"], alpha=0.6)
ax.set_xlabel("Stress")
ax.set_ylabel("Productivity")
st.pyplot(fig)

st.subheader("🔍 Employee Burnout Progress")
selected_emp = st.selectbox("Select Employee", sorted(data["Employee"].unique()))
emp_trend = data[data["Employee"] == selected_emp]
st.line_chart(emp_trend[["BurnoutRisk", "PredictedFutureBurnout"]])

st.markdown("---")
st.subheader("🌈 Daily Motivation")
quotes = [
    "Take small steps every day — they matter 🌱",
    "Balance is not something you find, it’s something you create 🌞",
    "Rest is productive too 🧘‍♀️",
    "Smile often — it’s the cheapest stress reliever 😄"
]
st.info(random.choice(quotes))