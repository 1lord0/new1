import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Öğrenci Not ve Katılım Takip Uygulaması")

uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    student_names = df["name"].unique()
    selected_name = st.selectbox("👤 Öğrenci Seçin", student_names)

    student_info = df[df["name"] == selected_name].iloc[0]
    st.markdown("### 👩‍🏫 Öğrenci Bilgileri")
    st.write(f"**Ad:** {student_info['name']}")
    st.write(f"**Mail:** {student_info['mail']}")
    st.write(f"**Sınıf:** {student_info['class_no']}")

    subjects = df[df["name"] == selected_name]["subject"].unique()
    selected_subject = st.selectbox("📘 Ders Seçin", subjects)

    teacher_name = df[(df["name"] == selected_name) & (df["subject"] == selected_subject)]["teacher"].iloc[0]
    st.write(f"**Öğretmen:** {teacher_name}")

    student_df = df[(df["name"] == selected_name) & (df["subject"] == selected_subject)]

    if not student_df.empty:
        # NOT GRAFİĞİ
        st.markdown("### 📈 Haftalık Not Grafiği")
        fig, ax = plt.subplots()
        ax.plot(student_df["week"], student_df["grade"], marker="o")
        ax.set_title(f"{selected_name} - {selected_subject} Notları")
        ax.set_xlabel("Hafta")
        ax.set_ylabel("Not")
        ax.grid(True)
        st.pyplot(fig)

        # KATILIM GRAFİĞİ
        st.markdown("### ✅ Derse Katılım Grafiği")
        participation_df = pd.DataFrame({
            "week": range(1, 6),  # 5 hafta olduğunu varsayıyoruz
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
    else:
        st.warning("Seçilen öğrenci ve ders için veri bulunamadı.")
else:
    st.info("Lütfen bir CSV dosyası yükleyin.")
