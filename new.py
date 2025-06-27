import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("📊 Öğrenci Not ve Katılım Takip Uygulaması")

# 1. CSV dosyası yükle
uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type=["csv"], key="file1")

if uploaded_file is not None:
    # 2. Veriyi oku ve temizle
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # sütun isimlerindeki boşlukları temizle

    # 3. Öğrenci seçimi
    student_names = df["name"].unique()
    selected_name = st.selectbox("Öğrenci seçin", student_names)

    # 4. Ders seçimi (seçilen öğrencinin dersleri)
    subjects = df[df["name"] == selected_name]["subject"].unique()
    selected_subject = st.selectbox("Ders seçin", subjects)

    # 5. Filtrele: seçilen öğrenci ve ders için veri
    student_df = df[(df["name"] == selected_name) & (df["subject"] == selected_subject)]

    if not student_df.empty:
        st.markdown(f"### 📈 {selected_name} - {selected_subject} Not Grafiği")

        # 6. Not grafiği
        fig, ax = plt.subplots()
        ax.plot(student_df["week"], student_df["grade"], marker="o")
        ax.set_xlabel("Hafta")
        ax.set_ylabel("Not")
        ax.set_title(f"{selected_name} - {selected_subject} Notları")
        ax.grid(True)
        st.pyplot(fig)

        # 7. Katılım grafiği
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

        # 8. Gelecek hafta not tahmini (lineer regresyon ile)
        st.markdown("### 🔮 Gelecek Hafta Not Tahmini")

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

    else:
        st.warning("Seçilen öğrenci ve ders için veri bulunamadı.")

else:
    st.info("Lütfen CSV dosyası yükleyin.")
with st.form("email_settings"):
    st.markdown("### 📩 Mail Ayarları (Öğretmen Girişi)")
    from_email = st.text_input("Gönderen E-posta (Gmail)", value="", placeholder="ornek@gmail.com")
    password = st.text_input("Uygulama Şifresi", type="password", placeholder="Gmail uygulama şifresi")

    submitted = st.form_submit_button("Kaydet")
