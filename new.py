import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf2 import FPDF
from io import BytesIO
import numpy as np
import unicodedata
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tempfile
import os
import logging
from sklearn.linear_model import LinearRegression

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Türkçe karakterleri kaldıran fonksiyon
def remove_accents(text):
    """Remove accents from Turkish characters for filename safety"""
    if not isinstance(text, str):
        text = str(text)
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def create_performance_chart(student_df, selected_name, selected_subject):
    """Create performance chart and return figure and image bytes"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(student_df["week"], student_df["grade"], marker="o", linewidth=2, markersize=8)
        ax.set_xlabel("Hafta", fontsize=12)
        ax.set_ylabel("Not", fontsize=12)
        ax.set_title(f"{selected_name} - {selected_subject} Notları", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)  # Assume grades are 0-100
        
        # Save to bytes
        img_bytes = BytesIO()
        fig.savefig(img_bytes, format="PNG", dpi=300, bbox_inches='tight')
        img_bytes.seek(0)
        
        return fig, img_bytes
    except Exception as e:
        logger.error(f"Error creating performance chart: {e}")
        return None, None

def create_attendance_chart(student_df, selected_name, selected_subject, max_week):
    """Create attendance chart"""
    try:
        attendance_df = pd.DataFrame({"week": range(1, max_week + 1)})
        attendance_df["attendance"] = attendance_df["week"].isin(student_df["week"]).astype(int)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if x == 0 else 'green' for x in attendance_df["attendance"]]
        ax.bar(attendance_df["week"], attendance_df["attendance"], color=colors, alpha=0.7)
        ax.set_title(f"{selected_name} - {selected_subject} Devam Durumu", fontsize=14, fontweight='bold')
        ax.set_xlabel("Hafta", fontsize=12)
        ax.set_ylabel("Devam (1=Var, 0=Yok)", fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3)
        
        return fig
    except Exception as e:
        logger.error(f"Error creating attendance chart: {e}")
        return None

def predict_next_grade(student_df):
    """Predict next week's grade using linear regression"""
    try:
        X = student_df["week"].values.reshape(-1, 1)
        y = student_df["grade"].values

        if len(X) >= 2 and len(np.unique(y)) > 1:
            model = LinearRegression()
            model.fit(X, y)
            next_week = np.array([[X[-1][0] + 1]])
            prediction = model.predict(next_week)[0]
            
            # Ensure prediction is within reasonable bounds
            prediction = max(0, min(100, prediction))
            
            return int(next_week[0][0]), prediction
        else:
            return None, None
    except Exception as e:
        logger.error(f"Error in grade prediction: {e}")
        return None, None

def create_pdf(student_name, student_df, plot_image_bytes):
    """Create PDF report with improved error handling"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Try to use Unicode font, fallback to default
        try:
            pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
            pdf.set_font('DejaVu', size=16)
        except (FileNotFoundError, RuntimeError):
            st.warning("DejaVuSans.ttf font dosyası bulunamadı. Varsayılan font kullanılıyor.")
            pdf.set_font("Arial", size=16)

        # Title
        pdf.cell(0, 15, f"{student_name} Haftalık Performans Raporu", ln=True, align="C")
        pdf.ln(10)

        # Grades table
        pdf.set_font_size(12)
        pdf.cell(0, 10, "NOTLAR:", ln=True)
        pdf.ln(5)
        
        for _, row in student_df.iterrows():
            pdf.cell(0, 8, f"Hafta {int(row['week'])}: {row['grade']}", ln=True)

        pdf.ln(10)
        
        # Add chart if available
        if plot_image_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                tmpfile.write(plot_image_bytes.getbuffer())
                tmpfilepath = tmpfile.name

            try:
                # Check if there's enough space for the image
                if pdf.get_y() + 80 > pdf.h - 20:  # If not enough space, add new page
                    pdf.add_page()
                
                pdf.image(tmpfilepath, x=10, y=pdf.get_y(), w=pdf.w - 20, h=80)
            finally:
                os.unlink(tmpfilepath)  # Always clean up temp file

        pdf_bytes = pdf.output()
        return pdf_bytes
    except Exception as e:
        logger.error(f"PDF creation error: {e}")
        st.error(f"PDF oluşturma hatası: {e}")
        return None

def send_email(from_email, password, to_email, subject, body, pdf_bytes, student_name):
    """Send email with PDF attachment"""
    try:
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # PDF attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(pdf_bytes)
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={remove_accents(student_name)}_report.pdf"
        )
        msg.attach(part)

        # Send email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        
        return True
    except smtplib.SMTPAuthenticationError:
        st.error("E-posta kimlik doğrulama hatası. App Password'u kontrol edin.")
        return False
    except smtplib.SMTPException as e:
        st.error(f"SMTP hatası: {e}")
        return False
    except Exception as e:
        logger.error(f"Email sending error: {e}")
        st.error(f"Mail gönderme hatası: {e}")
        return False

def validate_csv_data(df):
    """Validate CSV data structure and content"""
    required_columns = ["name", "subject", "week", "grade", "email"]
    df.columns = df.columns.str.strip().str.lower()
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Eksik sütunlar: {', '.join(missing_columns)}"
    
    # Check for empty data
    if df.empty:
        return False, "CSV dosyası boş"
    
    # Validate data types
    try:
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
        df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    except Exception as e:
        return False, f"Veri tipi hatası: {e}"
    
    # Check for missing values in critical columns
    if df["name"].isna().any() or df["subject"].isna().any():
        return False, "Öğrenci adı veya ders adı eksik"
    
    if df["week"].isna().any() or df["grade"].isna().any():
        return False, "Hafta veya not bilgisi eksik/geçersiz"
    
    return True, "OK"

# Streamlit UI
st.set_page_config(page_title="Öğrenci Takip Sistemi", page_icon="📊", layout="wide")
st.title("📊 Öğrenci Not ve Devam Takip Uygulaması")

# Sidebar for instructions
with st.sidebar:
    st.markdown("### 📋 Kullanım Kılavuzu")
    st.markdown("""
    1. CSV dosyanızı yükleyin
    2. Öğrenci ve ders seçin
    3. Grafikleri inceleyin
    4. PDF raporu indirin
    5. İsterseniz e-posta gönderin
    
    **CSV Format:**
    - name: Öğrenci adı
    - subject: Ders adı  
    - week: Hafta numarası
    - grade: Not (0-100)
    - email: E-posta adresi
    """)

# File upload
uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate data
        is_valid, error_msg = validate_csv_data(df)
        if not is_valid:
            st.error(error_msg)
            st.stop()
        
        # Student selection
        student_names = sorted(df["name"].unique())
        if len(student_names) == 0:
            st.error("CSV dosyasında öğrenci bulunamadı.")
            st.stop()
        
        col1, col2 = st.columns(2)
        with col1:
            selected_name = st.selectbox("🎓 Öğrenci Seçin", student_names)
        
        # Subject selection
        subjects = sorted(df[df["name"] == selected_name]["subject"].unique())
        if len(subjects) == 0:
            st.warning(f"{selected_name} için ders bulunamadı.")
            st.stop()
        
        with col2:
            selected_subject = st.selectbox("📚 Ders Seçin", subjects)
        
        # Filter data
        student_df = df[(df["name"] == selected_name) & (df["subject"] == selected_subject)].copy()
        student_df = student_df.sort_values("week")
        
        if student_df.empty:
            st.warning("Seçilen öğrenci ve ders için veri bulunamadı.")
            st.stop()
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📈 Ortalama Not", f"{student_df['grade'].mean():.1f}")
        with col2:
            st.metric("⭐ En Yüksek Not", f"{student_df['grade'].max():.0f}")
        with col3:
            st.metric("📉 En Düşük Not", f"{student_df['grade'].min():.0f}")
        with col4:
            st.metric("📅 Toplam Hafta", len(student_df))
        
        # Performance chart
        st.markdown(f"### 📈 {selected_name} - {selected_subject} Not Grafiği")
        perf_fig, img_bytes = create_performance_chart(student_df, selected_name, selected_subject)
        if perf_fig:
            st.pyplot(perf_fig)
            plt.close(perf_fig)
        
        # Attendance chart
        st.markdown("### ✅ Devam Grafiği")
        max_week = int(df["week"].max())
        attend_fig = create_attendance_chart(student_df, selected_name, selected_subject, max_week)
        if attend_fig:
            st.pyplot(attend_fig)
            plt.close(attend_fig)
        
        # Grade prediction
        st.markdown("### 🔮 Gelecek Hafta Not Tahmini")
        next_week, prediction = predict_next_grade(student_df)
        if next_week and prediction:
            st.success(f"📌 {next_week}. hafta için tahmini not: **{prediction:.1f}**")
        else:
            st.info("Tahmin için en az 2 hafta veri ve farklı notlar gerekli.")
        
        # PDF generation and email
        if img_bytes:
            pdf_bytes = create_pdf(selected_name, student_df, img_bytes)
            
            if pdf_bytes:
                st.markdown("### 📄 Rapor İşlemleri")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📄 PDF Raporunu İndir",
                        data=pdf_bytes,
                        file_name=f"{remove_accents(selected_name)}_rapor.pdf",
                        mime="application/pdf"
                    )
                
                with col2:
                    # Email form
                    with st.expander("📩 E-posta Gönder"):
                        st.info("Gmail App Password gerekli: https://myaccount.google.com/security")
                        
                        from_email = st.text_input("Gönderici E-posta", placeholder="ornek@gmail.com")
                        password = st.text_input("App Password", type="password")
                        
                        if st.button("E-posta Gönder", type="primary"):
                            if from_email and password:
                                to_email = student_df.iloc[0]["email"]
                                subject = f"{selected_name} - Haftalık Performans Raporu"
                                body = f"""Merhaba {selected_name},

Haftalık performans raporunuz ektedir.

Özet Bilgiler:
- Ortalama Not: {student_df['grade'].mean():.1f}
- En Yüksek Not: {student_df['grade'].max():.0f}
- Toplam Hafta: {len(student_df)}

İyi çalışmalar dileriz.
"""
                                
                                with st.spinner("E-posta gönderiliyor..."):
                                    result = send_email(from_email, password, to_email, subject, body, pdf_bytes, selected_name)
                                    if result:
                                        st.success("✅ E-posta başarıyla gönderildi!")
                            else:
                                st.warning("Lütfen e-posta ve App Password girin.")
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"Uygulama hatası: {e}")
        st.info("Lütfen CSV dosyanızın doğru formatta olduğundan emin olun.")

else:
    st.info("👆 Başlamak için yukarıdan bir CSV dosyası yükleyin.")
    st.markdown("### 📝 Örnek CSV Formatı")
    sample_data = pd.DataFrame({
        'name': ['Ali Veli', 'Ali Veli', 'Ayşe Kaya', 'Ayşe Kaya'],
        'subject': ['Matematik', 'Matematik', 'Fizik', 'Fizik'],
        'week': [1, 2, 1, 2],
        'grade': [85, 90, 78, 82],
        'email': ['ali@example.com', 'ali@example.com', 'ayse@example.com', 'ayse@example.com']
    })
    st.dataframe(sample_data)
