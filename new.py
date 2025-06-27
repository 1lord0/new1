import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO
import numpy as np
import unicodedata

def remove_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

# Kullanım:
student_name_clean = remove_accents(student_name)
pdf.cell(0, 10, f"{student_name_clean} Weekly Performance Report", ln=True, align="C")

import unicodedata
from fpdf import FPDF

def remove_accents(text):
    if not isinstance(text, str):
        text = str(text)
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def create_pdf(student_name, grades_dict, plot_image_bytes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Türkçe karakterlerden arındırılmış öğrenci adı
    student_name_clean = remove_accents(student_name)
    pdf.cell(0, 10, f"{student_name_clean} Weekly Performance Report", ln=True, align="C")
    pdf.ln(10)

    for subject, grade in grades_dict.items():
        subject_clean = remove_accents(subject)
        pdf.cell(0, 10, f"{subject_clean}: {grade}", ln=True)

    # Grafik resmi için geçici dosya oluşturma (FPDF image için gerekiyor)
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        tmpfile.write(plot_image_bytes.getbuffer())
        tmpfilepath = tmpfile.name

    pdf.image(tmpfilepath, x=10, y=pdf.get_y() + 5, w=pdf.w - 20)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes
st.download_button(
    label="Download PDF",
    data=pdf_bytes,
    file_name=f"{remove_accents(selected_student)}_report.pdf",
    mime="application/pdf"
)

st.title("Student Weekly Report Generator")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    selected_name = st.selectbox("Select Student", df["name"].unique())

    student_data = df[df["name"] == selected_name]
    grades = dict(zip(student_data["subject"], student_data["grade"]))

    fig, ax = plt.subplots()
    ax.bar(grades.keys(), grades.values(), color='skyblue')
    ax.set_ylim(0, 100)
    ax.set_title(f"{selected_name} - Weekly Grades")
    ax.set_ylabel("Grade")
    ax.set_xlabel("Subject")
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='PNG')
    plt.close(fig)
    img_bytes.seek(0)

    pdf_bytes = create_pdf(selected_name, grades, img_bytes)

    st.download_button(
        label="📄 Download PDF",
        data=pdf_bytes,
        file_name=f"{selected_name}_report.pdf",
        mime="application/pdf"
    )



# Mail gönderme fonksiyonu
def send_email(from_email, password, to_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Mail gönderme hatası: {e}")
        return False

st.title("📊 Öğrenci Not ve Katılım Takip Uygulaması")

# CSV Dosya Yükleme
uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={"mail": "email"}, inplace=True)

    # Öğrenci ve Ders Seçimi
    student_names = df["name"].unique()
    selected_name = st.selectbox("Öğrenci seçin", student_names)

    subjects = df[df["name"] == selected_name]["subject"].unique()
    selected_subject = st.selectbox("Ders seçin", subjects)

    student_df = df[(df["name"] == selected_name) & (df["subject"] == selected_subject)]

    if not student_df.empty:
        st.markdown(f"### 📈 {selected_name} - {selected_subject} Not Grafiği")

        # Not grafiği çizimi
        fig, ax = plt.subplots()
        ax.plot(student_df["week"], student_df["grade"], marker="o")
        ax.set_xlabel("Hafta")
        ax.set_ylabel("Not")
        ax.set_title(f"{selected_name} - {selected_subject} Notları")
        ax.grid(True)
        st.pyplot(fig)

        # Katılım grafiği
        st.markdown("### ✅ Derse Katılım Grafiği")
        max_week = df["week"].max()
        participation_df = pd.DataFrame({
            "week": range(1, max_week + 1)
        })
        participation_df["katılım"] = participation_df["week"].isin(student_df["week"]).astype(int)

        fig2, ax2 = plt.subplots()
        ax2.bar(participation_df["week"], participation_df["katılım"], color="green")
        ax2.set_title(f"{selected_name} - {selected_subject} Derse Katılım")
        ax2.set_xlabel("Hafta")
        ax2.set_ylabel("Katılım (1=Var, 0=Yok)")
        ax2.set_yticks([0, 1])
        ax2.set_ylim(0, 1.2)
        st.pyplot(fig2)

        # Gelecek hafta not tahmini
        st.markdown("### 🔮 Gelecek Hafta Not Tahmini")
        from sklearn.linear_model import LinearRegression
        X = student_df["week"].values.reshape(-1, 1)
        y = student_df["grade"].values

        if len(X) >= 2:
            model = LinearRegression()
            model.fit(X, y)

            next_week = np.array([[X[-1][0] + 1]])
            prediction = model.predict(next_week)[0]

            st.success(f"📌 Tahmini {int(next_week[0][0])}. hafta notu: **{prediction:.2f}**")
        else:
            st.info("Tahmin için en az 2 hafta verisi gerekli.")
        
        # PDF oluştur ve indir butonu
        grades = dict(zip(student_df["subject"], student_df["grade"]))

        img_bytes = BytesIO()
        fig.savefig(img_bytes, format="PNG")
        plt.close(fig)
        img_bytes.seek(0)

        pdf_bytes = create_pdf(selected_name, grades, img_bytes)

        st.download_button(
            label="PDF Raporu İndir",
            data=pdf_bytes,
            file_name=f"{selected_name}_rapor.pdf",
            mime="application/pdf",
        )

        # Mail ayarları formu
        with st.form("email_form"):
            st.markdown("### 📩 Mail Ayarları (Öğretmen Girişi)")
            from_email = st.text_input("Gönderen E-posta (Gmail)", placeholder="ornek@gmail.com")
            password = st.text_input("Uygulama Şifresi", type="password", placeholder="Gmail uygulama şifresi")
            submitted = st.form_submit_button("Mail Gönder")

            if submitted:
                to_email = student_df.iloc[0]["email"]
                subject = f"{selected_name} - Haftalık Rapor"
                body = f"Merhaba {selected_name},\n\nHaftalık performans raporun ektedir.\n\nİyi çalışmalar!"

                if from_email and password:
                    result = send_email(from_email, password, to_email, subject, body)
                    if result:
                        st.success("Mail gönderildi!")
                else:
                    st.warning("Lütfen e-posta ve şifre bilgilerini girin.")
    else:
        st.warning("Seçilen öğrenci ve ders için veri bulunamadı.")
else:
    st.info("Lütfen CSV dosyası yükleyin.")
