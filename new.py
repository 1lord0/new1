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

# Türkçe karakterleri kaldıran fonksiyon
def remove_accents(text):
    if not isinstance(text, str):
        text = str(text)
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

# PDF oluşturma fonksiyonu
def create_pdf(student_name, grades_dict, plot_image_bytes):
    try:
        pdf = FPDF()
        pdf.add_page()
        # Unicode destekli font (DejaVuSans.ttf dosyasını proje klasörüne ekleyin)
        try:
            pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
            pdf.set_font('DejaVu', size=12)
        except FileNotFoundError:
            st.warning("DejaVuSans.ttf font dosyası bulunamadı. Varsayılan font kullanılıyor.")
            pdf.set_font("Helvetica", size=12)

        pdf.cell(0, 10, f"{student_name} Haftalık Performans Raporu", ln=True, align="C")
        pdf.ln(10)

        for subject, grade in grades_dict.items():
            pdf.cell(0, 10, f"{subject}: {grade}", ln=True)

        # Grafik için geçici dosya
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            tmpfile.write(plot_image_bytes.getbuffer())
            tmpfilepath = tmpfile.name

        pdf.image(tmpfilepath, x=10, y=pdf.get_y() + 5, w=pdf.w - 20)
        pdf_bytes = pdf.output()  # fpdf2 varsayılan olarak bytes döndürür
        os.unlink(tmpfilepath)  # Geçici dosyayı sil
        return pdf_bytes
    except Exception as e:
        st.error(f"PDF oluşturma hatası: {e}")
        return None

# Mail gönderme fonksiyonu (PDF ekli)
def send_email(from_email, password, to_email, subject, body, pdf_bytes, student_name):
    try:
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # PDF eki ekleme
        part = MIMEBase("application", "octet-stream")
        part.set_payload(pdf_bytes)
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={remove_accents(student_name)}_report.pdf"
        )
        msg.attach(part)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Mail gönderme hatası: {e}")
        return False

# Streamlit arayüzü
st.title("📊 Öğrenci Not ve Devam Takip Uygulaması")

# CSV dosya yükleme
uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()
        required_columns = ["name", "subject", "week", "grade", "email"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Eksik sütunlar: {', '.join(missing_columns)}. Lütfen doğru formatta bir CSV yükleyin.")
        else:
            student_names = df["name"].unique()
            if len(student_names) == 0:
                st.error("CSV dosyasında öğrenci bulunamadı.")
            else:
                selected_name = st.selectbox("Öğrenci Seçin", student_names)
                subjects = df[df["name"] == selected_name]["subject"].unique()
                if len(subjects) == 0:
                    st.warning(f"{selected_name} için ders bulunamadı.")
                else:
                    selected_subject = st.selectbox("Ders Seçin", subjects)
                    student_df = df[(df["name"] == selected_name) & (df["subject"] == selected_subject)]

                    if not student_df.empty:
                        st.markdown(f"### 📈 {selected_name} - {selected_subject} Not Grafiği")
                        fig, ax = plt.subplots()
                        ax.plot(student_df["week"], student_df["grade"], marker="o")
                        ax.set_xlabel("Hafta")
                        ax.set_ylabel("Not")
                        ax.set_title(f"{selected_name} - {selected_subject} Notları")
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)  # Figürü kapat

                        st.markdown("### ✅ Devam Grafiği")
                        max_week = df["week"].max()
                        attendance_df = pd.DataFrame({"week": range(1, max_week + 1)})
                        attendance_df["attendance"] = attendance_df["week"].isin(student_df["week"]).astype Roku

                        fig2, ax2 = plt.subplots()
                        ax2.bar(attendance_df["week"], attendance_df["attendance"], color="green")
                        ax2.set_title(f"{selected_name} - {selected_subject} Devam")
                        ax2.set_xlabel("Hafta")
                        ax2.set_ylabel("Devam (1=Var, 0=Yok)")
                        ax2.set_yticks([0, 1])
                        ax2.set_ylim(0, 1.2)
                        st.pyplot(fig2)
                        plt.close(fig2)  # Figürü kapat

                        st.markdown("### 🔮 Gelecek Hafta Not Tahmini")
                        from sklearn.linear_model import LinearRegression

                        X = student_df["week"].values.reshape(-1, 1)
                        y = student_df["grade"].values

                        if len(X) >= 2 and len(np.unique(y)) > 1:
                            model = LinearRegression()
                            model.fit(X, y)
                            next_week = np.array([[X[-1][0] + 1]])
                            prediction = model.predict(next_week)[0]
                            st.success(f"📌 {int(next_week[0][0])}. hafta için tahmini not: **{prediction:.2f}**")
                        else:
                            st.info("Tahmin için en az 2 hafta veri ve farklı notlar gerekli.")

                        # PDF oluşturma
                        grades = dict(zip(student_df["subject"], student_df["grade"]))
                        img_bytes = BytesIO()
                        fig.savefig(img_bytes, format="PNG")
                        img_bytes.seek(0)

                        pdf_bytes = create_pdf(selected_name, grades, img_bytes)
                        plt.close(fig)  # Figürü kapat

                        if pdf_bytes:
                            st.download_button(
                                label="📄 PDF Raporunu İndir",
                                data=pdf_bytes,
                                file_name=f"{remove_accents(selected_name)}_rapor.pdf",
                                mime="application/pdf"
                            )

                            # Mail gönderme formu
                            with st.form("email_form"):
                                st.markdown("### 📩 E-posta Ayarları")
                                st.info("Gmail App Password için: https://myaccount.google.com/security")
                                from_email = st.text_input("Gönderici E-posta (Gmail)", placeholder="ornek@gmail.com")
                                password = st.text_input("App Password", type="password", placeholder="Gmail App Password")
                                submitted = st.form_submit_button("E-posta Gönder")

                                if submitted:
                                    if from_email and password:
                                        to_email = student_df.iloc[0]["email"]
                                        subject = f"{selected_name} - Haftalık Rapor"
                                        body = f"Merhaba {selected_name},\n\nHaftalık performans raporunuz ektedir.\n\nİyi çalışmalar."

                                        result = send_email(from_email, password, to_email, subject, body, pdf_bytes, selected_name)
                                        if result:
                                            st.success("E-posta başarıyla gönderildi!")
                                    else:
                                        st.warning("Lütfen e-posta ve şifre girin.")
                    else:
                        st.warning("Seçilen öğrenci ve ders için veri bulunamadı.")
    except Exception as e:
        st.error(f"CSV dosyası okunurken hata oluştu: {e}")
else:
    st.info("Lütfen bir CSV dosyası yükleyin.")
