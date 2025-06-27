import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO
import numpy as np
import unicodedata
import tempfile
from sklearn.linear_model import LinearRegression
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


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

    student_name_clean = remove_accents(student_name)
    pdf.cell(0, 10, f"{student_name_clean} Weekly Performance Report", ln=True, align="C")
    pdf.ln(10)

    for subject, grade in grades_dict.items():
        subject_clean = remove_accents(subject)
        pdf.cell(0, 10, f"{subject_clean}: {grade}", ln=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        tmpfile.write(plot_image_bytes.getbuffer())
        tmpfilepath = tmpfile.name

    pdf.image(tmpfilepath, x=10, y=pdf.get_y() + 5, w=pdf.w - 20)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes


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
        st.error(f"Mail sending error: {e}")
        return False


st.title("📊 Student Grade and Attendance Tracking App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv1")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={"mail": "email"}, inplace=True)

    student_names = df["name"].unique()
    selected_name = st.selectbox("Select Student", student_names)

    subjects = df[df["name"] == selected_name]["subject"].unique()
    selected_subject = st.selectbox("Select Subject", subjects)

    student_df = df[(df["name"] == selected_name) & (df["subject"] == selected_subject)]

    if not student_df.empty:
        st.markdown(f"### 📈 {selected_name} - {selected_subject} Grade Chart")

        fig, ax = plt.subplots()
        ax.plot(student_df["week"], student_df["grade"], marker="o")
        ax.set_xlabel("Week")
        ax.set_ylabel("Grade")
        ax.set_title(f"{selected_name} - {selected_subject} Grades")
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("### ✅ Attendance Chart")
        max_week = df["week"].max()
        participation_df = pd.DataFrame({"week": range(1, max_week + 1)})
        participation_df["attendance"] = participation_df["week"].isin(student_df["week"]).astype(int)

        fig2, ax2 = plt.subplots()
        ax2.bar(participation_df["week"], participation_df["attendance"], color="green")
        ax2.set_title(f"{selected_name} - {selected_subject} Attendance")
        ax2.set_xlabel("Week")
        ax2.set_ylabel("Attendance (1=Present, 0=Absent)")
        ax2.set_yticks([0, 1])
        ax2.set_ylim(0, 1.2)
        st.pyplot(fig2)

        st.markdown("### 🔮 Next Week Grade Prediction")

        X = student_df["week"].values.reshape(-1, 1)
        y = student_df["grade"].values

        if len(X) >= 2:
            model = LinearRegression()
            model.fit(X, y)
            next_week = np.array([[X[-1][0] + 1]])
            prediction = model.predict(next_week)[0]
            st.success(f"📌 Predicted grade for week {int(next_week[0][0])}: **{prediction:.2f}**")
        else:
            st.info("At least 2 weeks of data needed for prediction.")

        grades = dict(zip(student_df["subject"], student_df["grade"]))

        img_bytes = BytesIO()
        fig.savefig(img_bytes, format="PNG")
        plt.close(fig)
        img_bytes.seek(0)

        pdf_bytes = create_pdf(selected_name, grades, img_bytes)

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{remove_accents(selected_name)}_report.pdf",
            mime="application/pdf",
        )

        with st.form("email_form"):
            st.markdown("### 📩 Email Settings (Teacher Login)")
            from_email = st.text_input("Sender Email (Gmail)", placeholder="example@gmail.com")
            password = st.text_input("App Password", type="password", placeholder="Gmail app password")
            submitted = st.form_submit_button("Send Email")

            if submitted:
                to_email = student_df.iloc[0]["email"]
                subject = f"{selected_name} - Weekly Report"
                body = f"Hello {selected_name},\n\nYour weekly performance report is attached.\n\nBest regards!"

                if from_email and password:
                    result = send_email(from_email, password, to_email, subject, body)
                    if result:
                        st.success("Email sent!")
                else:
                    st.warning("Please enter email and password.")
    else:
        st.warning("No data found for selected student and subject.")
else:
    st.info("Please upload CSV file.")
