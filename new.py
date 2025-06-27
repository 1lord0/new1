

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

st.title("Öğrenci Notları Görselleştirme")

uploaded_file = st.file_uploader("CSV dosyanı yükle", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Kullanıcı isim seçsin
    names = df["name"].unique()
    selected_name = st.selectbox("Öğrenci seçin:", names)

    # İstersen ders seçimi de ekleyebilirsin
    subjects = df["subject"].unique()
    selected_subject = st.selectbox("Ders seçin:", subjects)

    # Filtrele
    student_df = df[(df["name"] == selected_name) & (df["subject"] == selected_subject)]

    if not student_df.empty:
        fig, ax = plt.subplots()
        ax.plot(student_df["week"], student_df["grade"], marker="o")
        ax.set_title(f"{selected_name} - {selected_subject} Notları (Haftalık)")
        ax.set_xlabel("Hafta")
        ax.set_ylabel("Not")
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning("Seçilen öğrenci ve ders için veri bulunamadı.")
else:
    st.info("Lütfen önce CSV dosyasını yükleyin.")






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



