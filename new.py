import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
import time
import threading
from datetime import datetime, timedelta
import json
import schedule

# Try to import optional dependencies
try:
    from fpdf2 import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    try:
        from fpdf import FPDF
        FPDF_AVAILABLE = True
    except ImportError:
        FPDF_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for enhanced scheduler
if 'email_scheduler' not in st.session_state:
    st.session_state.email_scheduler = {
        'active': False,
        'schedule_type': 'frequency',  # 'frequency' or 'custom'
        'frequency': 'weekly',
        'custom_schedules': [],  # List of custom schedule entries
        'last_sent': None,
        'next_send': None,
        'email_settings': {}
    }

if 'email_logs' not in st.session_state:
    st.session_state.email_logs = []

# Turkish character removal function
def remove_accents(text):
    """Remove accents from Turkish characters for filename safety"""
    if not isinstance(text, str):
        text = str(text)
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )
    teacher_name = student_df["teacher"].iloc[0]
st.markdown(f"#### 👨‍🏫 Teacher: **{teacher_name}**")


def create_performance_chart(student_df, selected_name, selected_subject):
    """Create performance chart and return figure and image bytes"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(student_df["week"], student_df["grade"], marker="o", linewidth=2, markersize=8)
        ax.set_xlabel("Hafta", fontsize=12)
        ax.set_ylabel("Not", fontsize=12)
        ax.set_title(f"{selected_name} - {selected_subject} Notları", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
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
    if not SKLEARN_AVAILABLE:
        st.warning("Tahmin özelliği için scikit-learn kütüphanesi gerekli. 'pip install scikit-learn' ile yükleyebilirsiniz.")
        return None, None
        
    try:
        from sklearn.linear_model import LinearRegression
        
        X = student_df["week"].values.reshape(-1, 1)
        y = student_df["grade"].values

        if len(X) >= 2 and len(np.unique(y)) > 1:
            model = LinearRegression()
            model.fit(X, y)
            next_week = np.array([[X[-1][0] + 1]])
            prediction = model.predict(next_week)[0]
            
            prediction = max(0, min(100, prediction))
            
            return int(next_week[0][0]), prediction
        else:
            return None, None
    except Exception as e:
        logger.error(f"Error in grade prediction: {e}")
        return None, None

def create_pdf(student_name, student_df, plot_image_bytes):
    """Create PDF report with improved error handling"""
    if not FPDF_AVAILABLE:
        st.error("PDF kütüphanesi yüklenmemiş. Lütfen 'pip install fpdf2' veya 'pip install fpdf' komutu ile yükleyin.")
        return None
        
    try:
        pdf = FPDF()
        pdf.add_page()
        
        try:
            pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
            pdf.set_font('DejaVu', size=16)
        except (FileNotFoundError, RuntimeError):
            pdf.set_font("Arial", size=16)
            student_name = remove_accents(student_name)

        # Title
        pdf.cell(0, 15, f"{student_name} Haftalik Performans Raporu", ln=True, align="C")
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
                if pdf.get_y() + 80 > pdf.h - 20:
                    pdf.add_page()
                
                pdf.image(tmpfilepath, x=10, y=pdf.get_y(), w=pdf.w - 20, h=80)
            finally:
                if os.path.exists(tmpfilepath):
                    os.unlink(tmpfilepath)

        try:
            pdf_output = pdf.output(dest='S')
            if isinstance(pdf_output, str):
                pdf_bytes = pdf_output.encode('latin-1')
            else:
                pdf_bytes = pdf_output
        except:
            pdf_output = pdf.output()
        
        if isinstance(pdf_output, bytearray):
            pdf_bytes = bytes(pdf_output)
        elif isinstance(pdf_output, str):
            pdf_bytes = pdf_output.encode('latin-1')
        else:
            pdf_bytes = pdf_output
            
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

def send_bulk_reports(df, from_email, password, frequency_type):
    """Send reports to all students"""
    success_count = 0
    error_count = 0
    
    students = df['name'].unique()
    
    for student_name in students:
        try:
            student_subjects = df[df['name'] == student_name]['subject'].unique()
            
            for subject in student_subjects:
                student_df = df[(df['name'] == student_name) & (df['subject'] == subject)]
                
                if not student_df.empty:
                    fig, img_bytes = create_performance_chart(student_df, student_name, subject)
                    
                    if fig and img_bytes:
                        pdf_bytes = create_pdf(student_name, student_df, img_bytes)
                        plt.close(fig)
                        
                        if pdf_bytes:
                            to_email = student_df.iloc[0]["email"]
                            subject_line = f"{student_name} - {frequency_type.title()} Performans Raporu"
                            body = f"""Merhaba {student_name},

{frequency_type.title()} performans raporunuz ektedir.

Özet Bilgiler ({subject}):
- Ortalama Not: {student_df['grade'].mean():.1f}
- En Yüksek Not: {student_df['grade'].max():.0f}
- Toplam Hafta: {len(student_df)}

Bu rapor otomatik olarak gönderilmiştir.

İyi çalışmalar dileriz.
"""
                            
                            result = send_email(from_email, password, to_email, subject_line, body, pdf_bytes, student_name)
                            if result:
                                success_count += 1
                                log_entry = {
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'student': student_name,
                                    'subject': subject,
                                    'status': 'success',
                                    'email': to_email
                                }
                            else:
                                error_count += 1
                                log_entry = {
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'student': student_name,
                                    'subject': subject,
                                    'status': 'error',
                                    'email': to_email
                                }
                            
                            st.session_state.email_logs.append(log_entry)
                            time.sleep(1)
                        
        except Exception as e:
            error_count += 1
            logger.error(f"Error sending report for {student_name}: {e}")
    
    return success_count, error_count

def get_next_send_time_frequency(frequency):
    """Calculate next send time based on frequency"""
    now = datetime.now()
    if frequency == 'daily':
        return now + timedelta(days=1)
    elif frequency == 'weekly':
        return now + timedelta(weeks=1)
    elif frequency == 'monthly':
        return now + timedelta(days=30)
    return now

def get_next_custom_send_time(custom_schedules):
    """Calculate next send time based on custom schedules"""
    now = datetime.now()
    next_times = []
    
    for schedule_item in custom_schedules:
        day_of_week = schedule_item['day']  # 0=Monday, 6=Sunday
        hour = schedule_item['hour']
        minute = schedule_item['minute']
        
        # Calculate days until next occurrence of this day
        days_ahead = day_of_week - now.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        
        next_time = now + timedelta(days=days_ahead)
        next_time = next_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If the time has passed today and it's the same day, schedule for next week
        if days_ahead == 0 and next_time <= now:
            next_time += timedelta(days=7)
        
        next_times.append(next_time)
    
    return min(next_times) if next_times else None

def check_and_send_scheduled_emails(df):
    """Check if it's time to send scheduled emails"""
    if not st.session_state.email_scheduler['active']:
        return
    
    now = datetime.now()
    next_send = st.session_state.email_scheduler.get('next_send')
    
    if next_send and now >= datetime.fromisoformat(next_send):
        settings = st.session_state.email_scheduler['email_settings']
        
        if settings.get('from_email') and settings.get('password'):
            with st.spinner("Otomatik raporlar gönderiliyor..."):
                schedule_type = st.session_state.email_scheduler['schedule_type']
                frequency_text = "Otomatik" if schedule_type == 'custom' else st.session_state.email_scheduler['frequency']
                
                success, errors = send_bulk_reports(
                    df, 
                    settings['from_email'], 
                    settings['password'],
                    frequency_text
                )
                
                st.success(f"✅ {success} rapor gönderildi, {errors} hata oluştu")
                
                # Update schedule
                if schedule_type == 'frequency':
                    frequency = st.session_state.email_scheduler['frequency']
                    next_send = get_next_send_time_frequency(frequency)
                else:  # custom
                    custom_schedules = st.session_state.email_scheduler['custom_schedules']
                    next_send = get_next_custom_send_time(custom_schedules)
                
                if next_send:
                    st.session_state.email_scheduler['last_sent'] = now.isoformat()
                    st.session_state.email_scheduler['next_send'] = next_send.isoformat()
                
                st.rerun()

def validate_csv_data(df):
    """Validate CSV data structure and content"""
    required_columns = ["name", "subject", "week", "grade", "email"]
    df.columns = df.columns.str.strip().str.lower()
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Eksik sütunlar: {', '.join(missing_columns)}"
    
    if df.empty:
        return False, "CSV dosyası boş"
    
    try:
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
        df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    except Exception as e:
        return False, f"Veri tipi hatası: {e}"
    
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
    5. İsteğe bağlı e-posta gönderin  

    **CSV Formatı:**  
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
        
        # Check for scheduled emails
        check_and_send_scheduled_emails(df)
        
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
            if FPDF_AVAILABLE:
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
                        # Single email form
                        with st.expander("📩 Tek Sefer E-posta Gönder"):
                            from_email = st.text_input("Gönderici E-posta", placeholder="ornek@gmail.com", key="single_email")
                            password = st.text_input("App Password", type="password", key="single_password")
                            
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
                
                # Enhanced Scheduled Email Section
                st.markdown("### ⏰ Otomatik Rapor Gönderimi")
                
                # Email settings for scheduler
                st.markdown("#### 📧 E-posta Ayarları")
                col1, col2 = st.columns(2)
                with col1:
                    scheduler_email = st.text_input("Gönderici E-posta", placeholder="ornek@gmail.com", key="scheduler_email")
                with col2:
                    scheduler_password = st.text_input("App Password", type="password", key="scheduler_password")
                
                # Schedule type selection
                st.markdown("#### 📅 Zamanlama Türü")
                schedule_type = st.radio(
                    "Zamanlama türünü seçin:",
                    ["frequency", "custom"],
                    format_func=lambda x: "Standart Sıklık" if x == "frequency" else "Özel Zamanlama",
                    horizontal=True,
                    key="schedule_type_radio"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if schedule_type == "frequency":
                        st.markdown("##### 📊 Standart Sıklık")
                        frequency = st.selectbox(
                            "Gönderim Sıklığı",
                            ["daily", "weekly", "monthly"],
                            format_func=lambda x: {"daily": "Günlük", "weekly": "Haftalık", "monthly": "Aylık"}[x],
                            index=1
                        )
                    else:
                        st.markdown("##### 🎯 Özel Zamanlama")
                        
                        # Day and time selection
                        day_names = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
                        selected_day = st.selectbox("Gün seçin:", day_names)
                        selected_day_index = day_names.index(selected_day)
                        
                        col_time1, col_time2 = st.columns(2)
                        with col_time1:
                            selected_hour = st.selectbox("Saat:", range(0, 24), index=9)
                        with col_time2:
                            selected_minute = st.selectbox("Dakika:", [0, 15, 30, 45], index=0)
                        
                        # Add schedule button
                        if st.button("➕ Zamanlama Ekle"):
                            new_schedule = {
                                'day': selected_day_index,
                                'day_name': selected_day,
                                'hour': selected_hour,
                                'minute': selected_minute
                            }
                            
                            if 'custom_schedules' not in st.session_state.email_scheduler:
                                st.session_state.email_scheduler['custom_schedules'] = []
                            
                            st.session_state.email_scheduler['custom_schedules'].append(new_schedule)
                            st.success(f"✅ {selected_day} {selected_hour:02d}:{selected_minute:02d} zamanlaması eklendi!")
                            st.rerun()
                        
                        # Display current custom schedules
                        if st.session_state.email_scheduler.get('custom_schedules'):
                            st.markdown("**Mevcut Zamanlamalar:**")
                            for i, schedule_item in enumerate(st.session_state.email_scheduler['custom_schedules']):
                                col_schedule, col_delete = st.columns([3, 1])
                                with col_schedule:
                                    st.text(f"📅 {schedule_item['day_name']} {schedule_item['hour']:02d}:{schedule_item['minute']:02d}")
                                with col_delete:
                                    if st.button("🗑️", key=f"delete_schedule_{i}"):
                                        st.session_state.email_scheduler['custom_schedules'].pop(i)
                                        st.rerun()
                
                with col2:
                    st.markdown("##### 📊 Kontrol Paneli")
                    
                    # Start/Stop buttons
                    col_start, col_stop = st.columns(2)
                    
                    with col_start:
                        start_disabled = False
                        if schedule_type == "custom" and not st.session_state.email_scheduler.get('custom_schedules'):
                            start_disabled = True
                            st.warning("⚠️ Önce zamanlama ekleyin")
                        
                        if st.button("🚀 Başlat", type="primary", disabled=start_disabled):
                            if scheduler_email and scheduler_password:
                                st.session_state.email_scheduler['active'] = True
                                st.session_state.email_scheduler['schedule_type'] = schedule_type
                                st.session_state.email_scheduler['email_settings'] = {
                                    'from_email': scheduler_email,
                                    'password': scheduler_password
                                }
                                
                                if schedule_type == "frequency":
                                    st.session_state.email_scheduler['frequency'] = frequency
                                    next_send = get_next_send_time_frequency(frequency)
                                else:
                                    custom_schedules = st.session_state.email_scheduler['custom_schedules']
                                    next_send = get_next_custom_send_time(custom_schedules)
                                
                                if next_send:
                                    st.session_state.email_scheduler['next_send'] = next_send.isoformat()
                                    st.success(f"✅ Otomatik gönderim başlatıldı!")
                                    st.success(f"📅 Sonraki gönderim: {next_send.strftime('%d/%m/%Y %H:%M')}")
                                    st.rerun()
                            else:
                                st.warning("⚠️ E-posta bilgilerini girin.")
                    
                    with col_stop:
                        if st.button("⏹️ Durdur"):
                            st.session_state.email_scheduler['active'] = False
                            st.session_state.email_scheduler['next_send'] = None
                            st.warning("⏸️ Otomatik gönderim durduruldu.")
                            st.rerun()
                    
                    # Status display
                    st.markdown("##### 📈 Durum Bilgisi")
                    
                    if st.session_state.email_scheduler['active']:
                        st.success("🟢 Aktif")
                        
                        if st.session_state.email_scheduler['schedule_type'] == 'frequency':
                            frequency_text = {
                                "daily": "Günlük", 
                                "weekly": "Haftalık", 
                                "monthly": "Aylık"
                            }[st.session_state.email_scheduler['frequency']]
                            st.info(f"📊 Tip: {frequency_text}")
                        else:
                            st.info("📊 Tip: Özel Zamanlama")
