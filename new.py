import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import streamlit as st
import tempfile
import os

def calculate_performance(student_df):
    """Calculate average performance for a student"""
    avg_grade = student_df["grade"].mean()
    return avg_grade

def create_chart(grades_dict, student_name):
    """Create a bar chart for student grades"""
    fig, ax = plt.subplots(figsize=(10, 6))
    subjects = list(grades_dict.keys())
    grades = list(grades_dict.values())
    
    bars = ax.bar(subjects, grades, color='skyblue', edgecolor='navy', alpha=0.7)
    ax.set_ylim(0, 100)
    ax.set_title(f"{student_name} - Weekly Grades", fontsize=16, fontweight='bold')
    ax.set_ylabel("Grade", fontsize=12)
    ax.set_xlabel("Subject", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, grade in zip(bars, grades):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{grade}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save to BytesIO
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='PNG', dpi=300, bbox_inches='tight')
    plt.close(fig)
    img_bytes.seek(0)
    
    return img_bytes

def create_pdf(student_name, grades_dict, img_bytes):
    """Create PDF report with student data and chart"""
    pdf_bytes = BytesIO()
    doc = SimpleDocTemplate(pdf_bytes, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph(f"<b>Weekly Performance Report - {student_name}</b>", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 20))
    
    # Summary
    avg_grade = sum(grades_dict.values()) / len(grades_dict.values()) if grades_dict else 0
    summary_text = f"""
    <b>Performance Summary:</b><br/>
    Average Grade: {avg_grade:.2f}<br/>
    Number of Subjects: {len(grades_dict)}<br/>
    Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}<br/>
    """
    summary = Paragraph(summary_text, styles['Normal'])
    story.append(summary)
    story.append(Spacer(1, 20))
    
    # Subject details
    details_text = "<b>Subject Breakdown:</b><br/>"
    for subject, grade in grades_dict.items():
        details_text += f"• {subject}: {grade}<br/>"
    
    details = Paragraph(details_text, styles['Normal'])
    story.append(details)
    story.append(Spacer(1, 20))
    
    # Add chart image
    if img_bytes:
        # Save image to temporary file for ReportLab
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(img_bytes.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            img = Image(tmp_file_path, width=400, height=240)
            story.append(img)
        except:
            story.append(Paragraph("Chart could not be generated", styles['Normal']))
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    doc.build(story)
    pdf_bytes.seek(0)
    return pdf_bytes

def send_email(from_email, password, to_email, subject, body, pdf_attachment=None, smtp_server="smtp.gmail.com", smtp_port=587):
    """Send email with PDF attachment"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Add body to email
        msg.attach(MIMEText(body, 'plain'))
        
        # Add PDF attachment
        if pdf_attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(pdf_attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {to_email.split("@")[0]}_weekly_report.pdf'
            )
            msg.attach(part)
        
        # Create SMTP session
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable TLS encryption
        server.login(from_email, password)
        
        # Send email
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        
        return True
        
    except Exception as e:
        st.error(f"Email sending failed: {str(e)}")
        return False

def process_student_reports(df, from_email, password):
    """Main function to process all student reports"""
    success_count = 0
    error_count = 0
    students = df['name'].unique()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, student_name in enumerate(students):
        try:
            status_text.text(f'Processing {student_name}...')
            
            # Get student data
            student_subjects = df[df['name'] == student_name]['subject'].unique()
            grades_dict = {}
            
            for subject in student_subjects:
                student_df = df[(df['name'] == student_name) & (df['subject'] == subject)]
                if not student_df.empty:
                    avg_grade = calculate_performance(student_df)
                    grades_dict[subject] = round(avg_grade, 2)
            
            if not grades_dict:
                st.warning(f"No grades found for {student_name}")
                error_count += 1
                continue
            
            # Create chart
            img_bytes = create_chart(grades_dict, student_name)
            
            # Create PDF
            pdf_bytes = create_pdf(student_name, grades_dict, img_bytes)
            
            # Get email and send
            student_email = df[df['name'] == student_name].iloc[0]['email']
            subject_line = f"{student_name} - Weekly Performance Report"
            body = f"""Hi {student_name},

Please find attached your weekly performance report.

Summary:
- Average Grade: {sum(grades_dict.values()) / len(grades_dict.values()):.2f}
- Subjects Evaluated: {len(grades_dict)}

Keep up the great work!

Best regards,
Academic Team"""
            
            send_result = send_email(from_email, password, student_email, subject_line, body, pdf_bytes)
            
            if send_result:
                success_count += 1
                st.success(f"✅ Report sent to {student_name}")
            else:
                error_count += 1
                
        except Exception as e:
            error_count += 1
            st.error(f"Error processing {student_name}: {str(e)}")
        
        # Update progress
        progress_bar.progress((i + 1) / len(students))
    
    status_text.text('Processing complete!')
    
    # Final summary
    st.success(f"✅ Reports successfully sent: {success_count}")
    if error_count:
        st.warning(f"⚠️ Errors occurred for {error_count} students.")
    
    return success_count, error_count

# Example usage with Streamlit
def main():
    st.title("Student Performance Report Generator")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        # Email configuration
        st.subheader("Email Configuration")
        from_email = st.text_input("From Email Address")
        password = st.text_input("Email Password", type="password")
        
        if st.button("Generate and Send Reports"):
            if from_email and password:
                with st.spinner('Generating reports...'):
                    success_count, error_count = process_student_reports(df, from_email, password)
            else:
                st.error("Please provide email credentials")

if __name__ == "__main__":
    main()
