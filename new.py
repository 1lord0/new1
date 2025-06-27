

import pandas as pd

import seaborn as sns


df=pd.read_csv(r"student_data.csv")




class person:
    def __init__(self, name):
        self.name = name
        
        print("person oluşturuldu")

class student(person):
    def __init__(self,student_no, name,email,classroom, subject,teacher,week,grade,attendance ):
        super().__init__(name)
        self.student_no = student_no
        self.email = email
        self.classroom=classroom
        self.week=week
        self.grade=grade
        self.attendance=attendance
        print("student oluşturuldu")

    def __str__(self):
        return f"{self.name}|no: {self.student_no} | room: {self.classroom} | Grade: {self.grade} | Mail: {self.email}|attendance: {self.attendance}"


students = []  # Student nesnelerini tutacak liste

# Her benzersiz öğrenciyi gez
for name in df["name"].unique():
    # Bu öğrencinin ilk satırını al
    sub_df = df[df["name"] == name].iloc[0]

    s = student(
        sub_df["student_id"],
        sub_df["name"],
        sub_df["email"],
        sub_df["classroom"],
        sub_df["subject"],
        sub_df["teacher"],
        sub_df["week"],
        sub_df["grade"],
        sub_df["attendance"]
    )
    students.append(s)
    print(s)
print(students[1])



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Öğrenci Not Takip Uygulaması")

# 1. Dosya yükleme
uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Boşlukları sil

    # 2. Öğrenci seçimi
    student_names = df["name"].unique()
    selected_name = st.selectbox("👤 Öğrenci Seçin", student_names)

    # 3. Seçilen öğrencinin tüm bilgileri
    student_info = df[df["name"] == selected_name].iloc[0]
    st.markdown("### 👩‍🏫 Öğrenci Bilgileri")
    st.write(f"**Ad:** {student_info['name']}")
    st.write(f"**Mail:** {student_info['mail']}")
    st.write(f"**Sınıf:** {student_info['class_no']}")

    # 4. Ders seçimi
    subjects = df[df["name"] == selected_name]["subject"].unique()
    selected_subject = st.selectbox("📘 Ders Seçin", subjects)

    teacher_name = df[(df["name"] == selected_name) & (df["subject"] == selected_subject)]["teacher"].iloc[0]
    st.write(f"**Öğretmen:** {teacher_name}")

    # 5. Veriyi filtrele ve çizdir
    student_df = df[(df["name"] == selected_name) & (df["subject"] == selected_subject)]

    if not student_df.empty:
        fig, ax = plt.subplots()
        ax.plot(student_df["week"], student_df["grade"], marker="o")
        ax.set_title(f"{selected_name} - {selected_subject} Notları")
        ax.set_xlabel("Hafta")
        ax.set_ylabel("Not")
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning("Seçilen öğrenci ve ders için veri bulunamadı.")
else:
    st.info("Lütfen bir CSV dosyası yükleyin.")





import streamlit as st
import matplotlib.pyplot as plt

# Örnek: df veri çerçevesinde 'name' sütunu olduğunu varsayıyoruz

# Kullanıcıya isim listesi dropdown olarak sunuluyor
names = df["name"].unique()
selected_name = st.selectbox("Öğrenci seçin:", names)

# Kullanıcı seçimine göre filtreleme
student_df = df[(df["name"] == selected_name) & (df["subject"] == "Math")]

# Grafik çizimi
fig, ax = plt.subplots()
ax.plot(student_df["week"], student_df["grade"], marker="o")
ax.set_title(f"{selected_name} - Math Notları (Haftalık)")
ax.set_xlabel("Hafta")
ax.set_ylabel("Not")
ax.grid(True)

st.pyplot(fig)



