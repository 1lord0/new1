import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
from concurrent.futures import ThreadPoolExecutor

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
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'email_scheduler' not in st.session_state:
    st.session_state.email_scheduler = {
        'active': False,
        'send_time': '12:00',
        'send_day': 'Monday',
        'last_sent': None,
        'next_send': None,
        'email_settings': {}
    }

if 'email_logs' not in st.session_state:
    st.session_state.email_logs = []

if 'teacher_reports' not in st.session_state:
    st.session_state.teacher_reports = []

# Set matplotlib style for better looking charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def remove_accents(text):
    """Remove accents from Turkish characters for filename safety"""
    if not isinstance(text, str):
        text = str(text)
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def calculate_performance_metrics(student_df):
    """Calculate comprehensive performance metrics"""
    metrics = {}
    
    if len(student_df) >= 2:
        # Trend analysis
        weeks = student_df['week'].values
        grades = student_df['grade'].values
        
        # Calculate trend
        correlation = np.corrcoef(weeks, grades)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        
        # Performance change
        recent_avg = grades[-2:].mean() if len(grades) >= 2 else grades[-1]
        earlier_avg = grades[:-2].mean() if len(grades) > 2 else grades[0]
        change = recent_avg - earlier_avg
        
        # Consistency (inverse of standard deviation)
        consistency = max(0, 100 - grades.std())
        
        metrics.update({
            'trend': 'Yükseliş' if correlation > 0.1 else 'Düşüş' if correlation < -0.1 else 'Kararlı',
            'trend_strength': abs(correlation),
            'performance_change': change,
            'consistency_score': consistency,
            'improvement_rate': change / len(weeks) if len(weeks) > 1 else 0
        })
    
    # Basic stats
    metrics.update({
        'current_grade': student_df['grade'].iloc[-1] if not student_df.empty else 0,
        'average_grade': student_df['grade'].mean(),
        'max_grade': student_df['grade'].max(),
        'min_grade': student_df['grade'].min(),
        'total_weeks': len(student_df),
        'last_week': student_df['week'].max()
    })
    
    return metrics

def create_enhanced_performance_chart(student_df, selected_name, selected_subject):
    """Create enhanced performance chart with trend analysis"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Main performance chart
        weeks = student_df["week"].values
        grades = student_df["grade"].values
        
        # Plot line with markers
        ax1.plot(weeks, grades, marker="o", linewidth=3, markersize=8, label='Notlar')
        
        # Add trend line if enough data
        if len(weeks) >= 3:
            z = np.polyfit(weeks, grades, 1)
            p = np.poly1d(z)
            ax1.plot(weeks, p(weeks), "--", alpha=0.7, linewidth=2, label='Trend')
        
        # Highlight last grade
        ax1.scatter(weeks[-1], grades[-1], color='red', s=150, zorder=5, label='Son Not')
        
        # Add grade labels
        for i, (week, grade) in enumerate(zip(weeks, grades)):
            ax1.annotate(f'{grade:.0f}', (week, grade), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        ax1.set_xlabel("Hafta", fontsize=12)
        ax1.set_ylabel("Not", fontsize=12)
        ax1.set_title(f"{selected_name} - {selected_subject} Detaylı Performans Analizi", 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        ax1.legend()
        
        # Performance comparison chart
        if len(grades) >= 2:
            # Calculate moving averages
            if len(grades) >= 3:
                moving_avg = np.convolve(grades, np.ones(3)/3, mode='valid')
                moving_weeks = weeks[2:]
                ax2.plot(moving_weeks, moving_avg, marker="s", linewidth=2, 
                        label='3 Haftalık Ortalama', color='orange')
            
            # Show grade differences
            grade_diff = np.diff(grades)
            week_diff = weeks[1:]
            colors = ['green' if diff >= 0 else 'red' for diff in grade_diff]
            
            ax2.bar(week_diff, grade_diff, color=colors, alpha=0.7, 
                   label='Haftalık Değişim')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_xlabel("Hafta", fontsize=12)
            ax2.set_ylabel("Not Değişimi", fontsize=12)
            ax2.set_title("Haftalık Performans Değişimi", fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        
        # Save to bytes
        img_bytes = BytesIO()
        fig.savefig(img_bytes, format="PNG", dpi=300, bbox_inches='tight')
        img_bytes.seek(0)
        
        return fig, img_bytes
    except Exception as e:
        logger.error(f"Error creating enhanced performance chart: {e}")
        return None, None

def predict_future_performance(student_df, weeks_ahead=4):
    """Predict future performance with confidence intervals"""
    if not SKLEARN_AVAILABLE or len(student_df) < 3:
        return None
        
    try:
        X = student_df["week"].values.reshape(-1, 1)
        y = student_df["grade"].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R-squared for model reliability
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Predict future weeks
        last_week = student_df["week"].max()
        future_weeks = np.arange(last_week + 1, last_week + weeks_ahead + 1).reshape(-1, 1)
        future_predictions = model.predict(future_weeks)
        
        # Calculate prediction confidence based on R-squared and data variance
        residuals = y - y_pred
        residual_std = np.std(residuals)
        confidence_interval = 1.96 * residual_std  # 95% confidence interval
        
        predictions = []
        for i, (week, pred) in enumerate(zip(future_weeks.flatten(), future_predictions)):
            # Adjust confidence based on how far into future
            adjusted_confidence = confidence_interval * (1 + i * 0.2)
            
            predictions.append({
                'week': int(week),
                'predicted_grade': max(0, min(100, pred)),
                'confidence_lower': max(0, pred - adjusted_confidence),
                'confidence_upper': min(100, pred + adjusted_confidence),
                'model_reliability': r2
            })
        
        return predictions
    except Exception as e:
        logger.error(f"Error in future performance prediction: {e}")
        return None

def create_comparison_chart(student_df, class_avg_df=None):
    """Create chart comparing student performance to class average"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        weeks = student_df["week"].values
        grades = student_df["grade"].values
        
        ax.plot(weeks, grades, marker="o", linewidth=3, markersize=8, 
               label='Öğrenci Performansı', color='blue')
        
        if class_avg_df is not None and not class_avg_df.empty:
            class_weeks = class_avg_df["week"].values
            class_grades = class_avg_df["grade"].values
            ax.plot(class_weeks, class_grades, marker="s", linewidth=2, 
                   label='Sınıf Ortalaması', color='red', alpha=0.7)
            
            # Fill between to show above/below average
            ax.fill_between(weeks, grades, 
                           np.interp(weeks, class_weeks, class_grades),
                           alpha=0.2, color='green', 
                           where=(grades >= np.interp(weeks, class_weeks, class_grades)),
                           label='Ortalama Üstü')
            ax.fill_between(weeks, grades, 
                           np.interp(weeks, class_weeks, class_grades),
                           alpha=0.2, color='red',
                           where=(grades < np.interp(weeks, class_weeks, class_grades)),
                           label='Ortalama Altı')
        
        ax.set_xlabel("Hafta", fontsize=12)
        ax.set_ylabel("Not", fontsize=12)
        ax.set_title("Öğrenci vs Sınıf Ortalaması Karşılaştırması", 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 100)
        
        return fig
    except Exception as e:
        logger.error(f"Error creating comparison chart: {e}")
        return None

def generate_teacher_summary(df):
    """Generate comprehensive teacher summary report"""
    try:
        summary = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'total_students': df['name'].nunique(),
            'total_subjects': df['subject'].nunique(),
            'current_week': df['week'].max(),
            'subjects': {}
        }
        
        # Subject-wise analysis
        for subject in df['subject'].unique():
            subject_df = df[df['subject'] == subject]
            current_week_data = subject_df[subject_df['week'] == subject_df['week'].max()]
            
            # Performance metrics
            subject_metrics = {
                'student_count': subject_df['name'].nunique(),
                'average_grade': subject_df['grade'].mean(),
                'current_week_avg': current_week_data['grade'].mean(),
                'highest_grade': subject_df['grade'].max(),
                'lowest_grade': subject_df['grade'].min(),
                'grade_distribution': {
                    'A (90-100)': len(subject_df[subject_df['grade'] >= 90]),
                    'B (80-89)': len(subject_df[(subject_df['grade'] >= 80) & (subject_df['grade'] < 90)]),
                    'C (70-79)': len(subject_df[(subject_df['grade'] >= 70) & (subject_df['grade'] < 80)]),
                    'D (60-69)': len(subject_df[(subject_df['grade'] >= 60) & (subject_df['grade'] < 70)]),
                    'F (<60)': len(subject_df[subject_df['grade'] < 60])
                },
                'students_at_risk': [],
                'top_performers': [],
                'most_improved': []
            }
            
            # Identify students needing attention
            for student in subject_df['name'].unique():
                student_data = subject_df[subject_df['name'] == student].sort_values('week')
                
                if len(student_data) >= 2:
                    recent_avg = student_data['grade'].tail(2).mean()
                    overall_avg = student_data['grade'].mean()
                    trend = student_data['grade'].diff().tail(3).mean()
                    
                    # At risk: low grades or negative trend
                    if recent_avg < 60 or (trend < -5 and recent_avg < 75):
                        subject_metrics['students_at_risk'].append({
                            'name': student,
                            'recent_avg': recent_avg,
                            'trend': trend
                        })
                    
                    # Top performers
                    if recent_avg >= 90:
                        subject_metrics['top_performers'].append({
                            'name': student,
                            'recent_avg': recent_avg
                        })
                    
                    # Most improved
                    if trend > 5:
                        subject_metrics['most_improved'].append({
                            'name': student,
                            'improvement': trend
                        })
            
            summary['subjects'][subject] = subject_metrics
        
        return summary
    except Exception as e:
        logger.error(f"Error generating teacher summary: {e}")
        return None

def create_enhanced_pdf(student_name, student_df, plot_image_bytes, metrics, predictions=None, class_comparison=None):
    """Create comprehensive PDF report"""
    if not FPDF_AVAILABLE:
        st.error("PDF kütüphanesi yüklenmemiş. Lütfen 'pip install fpdf2' komutu ile yükleyin.")
        return None
        
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Font setup
        try:
            pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
            pdf.set_font('DejaVu', size=16)
        except (FileNotFoundError, RuntimeError):
            pdf.set_font("Arial", size=16)
            student_name = remove_accents(student_name)

        # Header
        pdf.cell(0, 15, f"{student_name} - Haftalik Detay Performans Raporu", ln=True, align="C")
        pdf.ln(5)
        
        # Report date
        pdf.set_font_size(10)
        pdf.cell(0, 8, f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}", ln=True, align="R")
        pdf.ln(5)

        # Performance Summary
        pdf.set_font_size(14)
        pdf.cell(0, 10, "PERFORMANS OZETI:", ln=True)
        pdf.set_font_size(10)
        pdf.ln(3)
        
        # Current status
        pdf.cell(0, 6, f"Guncel Not: {metrics['current_grade']:.1f}", ln=True)
        pdf.cell(0, 6, f"Genel Ortalama: {metrics['average_grade']:.1f}", ln=True)
        pdf.cell(0, 6, f"En Yuksek Not: {metrics['max_grade']:.0f}", ln=True)
        pdf.cell(0, 6, f"Trend: {metrics.get('trend', 'Belirsiz')}", ln=True)
        
        if 'performance_change' in metrics:
            change_text = f"+{metrics['performance_change']:.1f}" if metrics['performance_change'] >= 0 else f"{metrics['performance_change']:.1f}"
            pdf.cell(0, 6, f"Son Performans Degisimi: {change_text} puan", ln=True)
        
        pdf.ln(5)

        # Weekly grades table
        pdf.set_font_size(12)
        pdf.cell(0, 8, "HAFTALIK NOTLAR:", ln=True)
        pdf.set_font_size(10)
        pdf.ln(3)
        
        for _, row in student_df.iterrows():
            week_num = int(row['week'])
            grade = row['grade']
            
            # Color coding for grades
            if grade >= 90:
                grade_status = " (Mukemmel)"
            elif grade >= 80:
                grade_status = " (Iyi)"
            elif grade >= 70:
                grade_status = " (Orta)"
            elif grade >= 60:
                grade_status = " (Gecti)"
            else:
                grade_status = " (Kaldi)"
                
            pdf.cell(0, 6, f"Hafta {week_num}: {grade:.1f}{grade_status}", ln=True)

        pdf.ln(5)
        
        # Predictions
        if predictions:
            pdf.set_font_size(12)
            pdf.cell(0, 8, "GELECEK HAFTA TAHMINLERI:", ln=True)
            pdf.set_font_size(10)
            pdf.ln(3)
            
            for pred in predictions[:2]:  # Show only next 2 weeks
                pdf.cell(0, 6, f"Hafta {pred['week']}: {pred['predicted_grade']:.1f} (Guvenilirlik: {pred['model_reliability']:.2f})", ln=True)
            
            pdf.ln(5)

        # Add performance chart
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

        # Recommendations
        pdf.ln(85)
        pdf.set_font_size(12)
        pdf.cell(0, 8, "ONERILER:", ln=True)
        pdf.set_font_size(10)
        pdf.ln(3)
        
        # Generate personalized recommendations
        if metrics.get('trend') == 'Düşüş':
            pdf.cell(0, 6, "- Performansinizda dunus goruluyor. Ek calisma onerilir.", ln=True)
        elif metrics.get('trend') == 'Yükseliş':
            pdf.cell(0, 6, "- Harika! Performansiniz yukselis egiliminde.", ln=True)
        
        if metrics['average_grade'] < 70:
            pdf.cell(0, 6, "- Daha duzenli calisma programi olusturun.", ln=True)
            pdf.cell(0, 6, "- Ogretmeninizden ek yardim isteyin.", ln=True)
        
        if metrics.get('consistency_score', 50) < 70:
            pdf.cell(0, 6, "- Performansinizi daha tutarli hale getirmeye odaklanin.", ln=True)

        # Generate PDF bytes
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
        logger.error(f"Enhanced PDF creation error: {e}")
        st.error(f"PDF oluşturma hatası: {e}")
        return None

def send_enhanced_email(from_email, password, to_email, subject, student_name, metrics, predictions, pdf_bytes):
    """Send enhanced email with detailed performance information"""
    try:
        # Create enhanced email body
        trend_emoji = "📈" if metrics.get('trend') == 'Yükseliş' else "📉" if metrics.get('trend') == 'Düşüş' else "📊"
        
        body = f"""Merhaba {student_name},

{trend_emoji} HAFTALIK PERFORMANS RAPORU {trend_emoji}

📊 PERFORMANS ÖZETİ:
• Güncel Notunuz: {metrics['current_grade']:.1f}
• Genel Ortalamanız: {metrics['average_grade']:.1f}
• Performans Trendi: {metrics.get('trend', 'Belirsiz')}
• Toplam Hafta Sayısı: {metrics['total_weeks']}

"""

        # Add performance change information
        if 'performance_change' in metrics:
            change = metrics['performance_change']
            if change > 0:
                body += f"🎉 Harika! Son haftalarda {change:.1f} puan iyileşme gösterdiniz.\n\n"
            elif change < -2:
                body += f"⚠️ Dikkat! Son haftalarda {abs(change):.1f} puan düşüş var. Daha fazla çalışma öneriyoruz.\n\n"
            else:
                body += f"📊 Performansınız stabil kalıyor.\n\n"

        # Add predictions
        if predictions:
            body += "🔮 GELECEK HAFTA TAHMİNLERİ:\n"
            for pred in predictions[:2]:
                reliability = "Yüksek" if pred['model_reliability'] > 0.7 else "Orta" if pred['model_reliability'] > 0.4 else "Düşük"
                body += f"• Hafta {pred['week']}: {pred['predicted_grade']:.1f} (Güvenilirlik: {reliability})\n"
            body += "\n"

        # Add recommendations
        body += "💡 ÖNERİLER:\n"
        if metrics.get('trend') == 'Düşüş':
            body += "• Performansınızda düşüş gözlemleniyor. Ek çalışma yapmanızı öneriyoruz.\n"
        elif metrics.get('trend') == 'Yükseliş':
            body += "• Harika! Bu yükseliş trendini sürdürün.\n"
        
        if metrics['average_grade'] < 70:
            body += "• Daha düzenli çalışma programı oluşturun.\n"
            body += "• Öğretmeninizden ek yardım alın.\n"
        elif metrics['average_grade'] >= 90:
            body += "• Mükemmel performans! Bu seviyeyi koruyun.\n"

        body += f"""
📄 Detaylı rapor ekte bulunmaktadır.

Bu rapor otomatik olarak her hafta Pazartesi 12:00'da gönderilmektedir.

Başarılar dileriz! 🎓

---
Otomatik Öğrenci Takip Sistemi
Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}
"""

        # Send email
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # PDF attachment
        if pdf_bytes:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(pdf_bytes)
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={remove_accents(student_name)}_detay_rapor.pdf"
            )
            msg.attach(part)

        # Send email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        
        return True
    except Exception as e:
        logger.error(f"Enhanced email sending error: {e}")
        return False

def send_teacher_summary_email(from_email, password, teacher_email, summary):
    """Send comprehensive summary to teacher"""
    try:
        subject = f"Haftalık Öğretmen Özet Raporu - {datetime.now().strftime('%d.%m.%Y')}"
        
        body = f"""Sayın Öğretmenim,

📊 HAFTALIK SINIF PERFORMANS ÖZETİ

📅 Rapor Tarihi: {summary['report_date']}
👥 Toplam Öğrenci: {summary['total_students']}
📚 Toplam Ders: {summary['total_subjects']}
📍 Mevcut Hafta: {summary['current_week']}

"""

        # Subject-wise breakdown
        for subject, metrics in summary['subjects'].items():
            body += f"\n📖 {subject.upper()}:\n"
            body += f"• Öğrenci Sayısı: {metrics['student_count']}\n"
            body += f"• Genel Ortalama: {metrics['average_grade']:.1f}\n"
            body += f"• Bu Hafta Ortalama: {metrics['current_week_avg']:.1f}\n"
            body += f"• En Yüksek Not: {metrics['highest_grade']:.1f}\n"
            body += f"• En Düşük Not: {metrics['lowest_grade']:.1f}\n"
            
            # Grade distribution
            body += "\n📊 Not Dağılımı:\n"
            for grade_range, count in metrics['grade_distribution'].items():
                body += f"  {grade_range}: {count} öğrenci\n"
            
            # Students needing attention
            if metrics['students_at_risk']:
                body += "\n⚠️ Dikkat Gereken Öğrenciler:\n"
                for student in metrics['students_at_risk'][:5]:  # Top 5
                    body += f"  • {student['name']}: {student['recent_avg']:.1f} (Trend: {student['trend']:.1f})\n"
            
            # Top performers
            if metrics['top_performers']:
                body += "\n🌟 Başarılı Öğrenciler:\n"
                for student in metrics['top_performers'][:5]:  # Top 5
                    body += f"  • {student['name']}: {student['recent_avg']:.1f}\n"
            
            # Most improved
            if metrics['most_improved']:
                body += "\n📈 En Çok Gelişen Öğrenciler:\n"
                for student in metrics['most_improved'][:3]:  # Top 3
                    body += f"  • {student['name']}: +{student['improvement']:.1f} puan\n"
            
            body += "\n" + "="*50 + "\n"

        body += f"""
Bu rapor otomatik olarak her hafta Pazartesi gönderilmektedir.

Detaylı analiz için sistemde oturum açabilirsiniz.

İyi çalışmalar! 👨‍🏫

---
Otomatik Öğrenci Takip Sistemi
"""

        # Send email
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = teacher_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_email, password)
            server.sendmail(from_email, teacher_email, msg.as_string())
        
        return True
    except Exception as e:
        logger.error(f"Teacher summary email error: {e}")
        return False

def schedule_monday_reports(df, email_settings):
    """Schedule automatic Monday 12:00 AM reports"""
    def send_weekly_reports():
        try:
            logger.info("Starting scheduled weekly report generation...")
            
            # Generate teacher summary
            teacher_summary = generate_teacher_summary(df)
            
            # Send teacher summary first
            if teacher_summary and email_settings.get('teacher_email'):
                send_teacher_summary_email(
                    email_settings['from_email'],
                    email_settings['password'],
                    email_settings['teacher_email'],
                    teacher_summary
                )
                
                # Log teacher email
                st.session_state.teacher_reports.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'teacher_summary',
                    'status': 'success'
                })
            
            # Send student reports
from io import BytesIO
import matplotlib.pyplot as plt

# Performans hesaplama (örnek basit fonksiyon)
def calculate_performance(student_df):
    avg_grade = student_df["grade"].mean()
    return avg_grade

success_count = 0
error_count = 0

students = df['name'].unique()

for student_name in students:
    try:
        student_subjects = df[df['name'] == student_name]['subject'].unique()
        grades_dict = {}

        for subject in student_subjects:
            student_df = df[(df['name'] == student_name) & (df['subject'] == subject)]

            if not student_df.empty:
                avg_grade = calculate_performance(student_df)
                grades_dict[subject] = round(avg_grade, 2)

        # Bar chart (grafik)
        fig, ax = plt.subplots()
        ax.bar(grades_dict.keys(), grades_dict.values(), color='skyblue')
        ax.set_ylim(0, 100)
        ax.set_title(f"{student_name} - Weekly Grades")
        ax.set_ylabel("Grade")
        ax.set_xlabel("Subject")

        img_bytes = BytesIO()
        fig.savefig(img_bytes, format='PNG')
        plt.close(fig)
        img_bytes.seek(0)

        # PDF oluştur
        pdf_bytes = create_pdf(student_name, grades_dict, img_bytes)

        # E-posta bilgileri
        student_email = df[df['name'] == student_name].iloc[0]['email']
        subject_line = f"{student_name} - Weekly Report"
        body = f"Hi {student_name},\n\nAttached is your weekly performance report.\n\nBest regards."

        send_result = send_email(from_email, password, student_email, subject_line, body)

        if send_result:
            success_count += 1
        else:
            error_count += 1

    except Exception as e:
        error_count += 1
        st.error(f"Error for {student_name}: {e}")

st.success(f"✅ Reports sent: {success_count}")
if error_count:
    st.warning(f"⚠️ Errors occurred for {error_count} students.")
