import streamlit as st
from models.predict import MED117Predictor
from PIL import Image
import os
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
import time
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication


def get_db():
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client['herbal_detection']
        return db
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None


# ========== SIMPLE IMAGE VALIDATION ==========
def validate_image_basic(image_file):
    """Simple validation - checks brightness and contrast only"""
    try:
        image_file.seek(0)
        img = Image.open(image_file)
        img_array = np.array(img)
        
        if len(img_array.shape) == 3:
            img_gray = np.mean(img_array, axis=2)
        else:
            img_gray = img_array
        
        avg_brightness = np.mean(img_gray)
        std_brightness = np.std(img_gray)
        
        if avg_brightness < 10 or avg_brightness > 245:
            return False, "❌ Image quality issue (too dark/bright)"
        if std_brightness < 5:
            return False, "❌ Image has no texture/detail"
        if std_brightness > 100:
            return False, "❌ Image too noisy"
        
        return True, "✅ Basic checks passed"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


# ========== PDF GENERATION ==========
def create_detection_pdf(plant_name, confidence, plant_info, treatments, results):
    """Generate comprehensive PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#4ade80'),
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("🌿 Plant Detection Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Detection Result
    story.append(Paragraph(f"<b>Detection Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    result_data = [
        ['Plant Name', plant_name],
        ['Confidence', f'{confidence:.1f}%'],
    ]
    result_table = Table(result_data, colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#4ade80')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Plant Information
    if plant_info:
        story.append(Paragraph("<b>Plant Information</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        info_data = [
            ['Common Name', plant_info.get('common_name', 'N/A')],
            ['Scientific Name', plant_info.get('plant_name', 'N/A')],
            ['Family', plant_info.get('family', 'N/A')],
            ['Local Name (Assamese)', plant_info.get('assamese_name', 'Unknown')],
            ['Parts Used', plant_info.get('parts_used', 'N/A')],
        ]
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Description
        story.append(Paragraph("<b>Description:</b>", styles['Heading3']))
        story.append(Paragraph(plant_info.get('description', 'N/A'), styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Medicinal Uses
        story.append(Paragraph("<b>Medicinal Uses:</b>", styles['Heading3']))
        story.append(Paragraph(plant_info.get('medicinal_uses', 'N/A'), styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Treatments
    if treatments:
        story.append(Paragraph("<b>Treatment Options</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        for idx, treatment in enumerate(treatments, 1):
            story.append(Paragraph(f"<b>Treatment #{idx}</b>", styles['Heading3']))
            treat_data = [
                ['Method', treatment.get('treatment', 'N/A')],
                ['Application', treatment.get('application', 'N/A')],
                ['Frequency', treatment.get('frequency', 'N/A')],
            ]
            treat_table = Table(treat_data, colWidths=[1.5*inch, 4.5*inch])
            treat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#a78bfa')),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(treat_table)
            story.append(Spacer(1, 0.15*inch))
    
    # Top 5 Predictions
    story.append(Paragraph("<b>Top 5 Predictions</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    pred_data = [['Rank', 'Plant Name', 'Confidence']]
    for i, r in enumerate(results, 1):
        pred_data.append([f'#{i}', r.get('plant_name'), f"{r.get('confidence', 0):.1f}%"])
    
    pred_table = Table(pred_data, colWidths=[0.8*inch, 3.5*inch, 1.7*inch])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fbbf24')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(pred_table)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


# ========== EMAIL SENDING ==========
def send_email_with_pdf(recipient_email, plant_name, pdf_buffer):
    """Send email with PDF attachment"""
    try:
        sender_email = "bhargavbotta7@gmail.com"
        sender_password = "jxguvciaavzjdlzv"
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f'🌿 Plant Detection Report - {plant_name}'
        
        body = f"""
        <html>
        <body style='font-family: Arial, sans-serif;'>
            <h2 style='color: #4ade80;'>🌿 Plant Detection Report</h2>
            <p>Dear User,</p>
            <p>Please find attached your plant detection report for <b>{plant_name}</b>.</p>
            <p>Detection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <br>
            <p style='color: #666;'>This is an automated email from Herbal Plant Detection System.</p>
            <p style='color: #999; font-size: 12px;'>Sent from: {sender_email}</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        pdf_attachment = MIMEApplication(pdf_buffer.read(), _subtype='pdf')
        pdf_attachment.add_header('Content-Disposition', 'attachment', 
                                 filename=f'Plant_Detection_{plant_name}_{datetime.now().strftime("%Y%m%d")}.pdf')
        msg.attach(pdf_attachment)
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        return True, "✅ Email sent successfully!"
    except Exception as e:
        return False, f"❌ Email error: {str(e)}"


# ========== IMPROVED CSS ==========
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
* { font-family: 'Poppins', sans-serif !important; }
html, body, .main, .block-container, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1d3a 50%, #0f1529 100%) !important;
    background-attachment: fixed !important; color: #fff !important;
}
[data-testid="stSidebar"] { background: linear-gradient(135deg, #0f1429 0%, #1a1f3a 100%) !important; }
h1, h2, h3, h4, h5, h6 { color: #4ade80 !important; font-weight: 800 !important; text-shadow: 0 0 15px rgba(74, 222, 128, 0.3) !important; }
.stButton>button { background: linear-gradient(135deg, #4ade80 0%, #22c55e 50%, #16a34a 100%) !important; color: #000 !important; font-weight: 800 !important; border-radius: 14px !important; padding: 16px 32px !important; box-shadow: 0 8px 25px rgba(74, 222, 128, 0.35) !important; }

/* GREEN PDF DOWNLOAD BUTTON */
div.stDownloadButton > button {
    background: linear-gradient(135deg, #4ade80 0%, #22c55e 50%, #16a34a 100%) !important;
    color: #000 !important;
    font-weight: 800 !important;
    border-radius: 14px !important;
    padding: 16px 32px !important;
    box-shadow: 0 8px 25px rgba(74, 222, 128, 0.35) !important;
}

/* VISIBLE INPUT LABELS */
label[data-testid="stWidgetLabel"] {
    color: #4ade80 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    margin-bottom: 8px !important;
    text-shadow: 0 0 10px rgba(74, 222, 128, 0.3) !important;
}

input, .stTextInput input, .stPasswordInput input { 
    background: rgba(15, 20, 41, 0.95) !important; 
    color: #fff !important; 
    border: 2px solid rgba(74, 222, 128, 0.5) !important; 
    border-radius: 10px !important; 
    padding: 12px 16px !important;
    font-size: 15px !important;
}

input:focus, .stTextInput input:focus, .stPasswordInput input:focus {
    border: 2px solid #4ade80 !important;
    box-shadow: 0 0 15px rgba(74, 222, 128, 0.4) !important;
}

.title-box { background: linear-gradient(135deg, rgba(74, 222, 128, 0.15) 0%, rgba(34, 197, 94, 0.08) 100%) !important; border: 3px solid #4ade80 !important; border-radius: 20px !important; padding: 40px 30px !important; text-align: center !important; }
.result-card { background: linear-gradient(135deg, rgba(74, 222, 128, 0.15) 0%, rgba(34, 197, 94, 0.08) 100%) !important; border: 3px solid #4ade80 !important; border-radius: 18px !important; padding: 40px !important; text-align: center !important; }
.info-card { background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.08) 100%) !important; border: 2px solid #3b82f6 !important; border-radius: 14px !important; padding: 22px !important; margin-bottom: 15px !important; }
.treatment-card { background: linear-gradient(135deg, rgba(167, 139, 250, 0.15) 0%, rgba(139, 92, 246, 0.08) 100%) !important; border: 2px solid #a78bfa !important; border-radius: 12px !important; padding: 18px !important; margin: 10px 0 !important; }

/* IMPROVED USER CARD */
.user-card { 
    background: linear-gradient(135deg, rgba(74, 222, 128, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%) !important; 
    border: 2px solid rgba(74, 222, 128, 0.4) !important; 
    border-radius: 16px !important; 
    padding: 24px !important; 
    margin: 16px 0 !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
}

.history-card { background: linear-gradient(135deg, rgba(251, 191, 36, 0.15) 0%, rgba(245, 158, 11, 0.08) 100%) !important; border: 2px solid #fbbf24 !important; border-radius: 12px !important; padding: 18px !important; margin: 10px 0 !important; }

/* ADMIN TEXT VISIBILITY */
.admin-title {
    color: #ef4444 !important;
    font-weight: 700 !important;
    text-shadow: 0 0 20px rgba(239, 68, 68, 0.6) !important;
}

/* FIX EXPANDER SPACING */
.streamlit-expanderHeader {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #4ade80 !important;
}

/* CLEAN EXPANDER CONTENT */
.streamlit-expanderContent {
    padding: 20px !important;
    background: rgba(30, 33, 57, 0.5) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="🌿 Herbal Plants Disease Detection", page_icon="🌿", layout="wide")

if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'user_type' not in st.session_state:
    st.session_state['user_type'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "🔬 Detection"
if 'show_register' not in st.session_state:
    st.session_state['show_register'] = False
if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# ========== CSV FUNCTIONS ==========
def load_plant_info(plant_name):
    try:
        df = pd.read_csv("data/metadata/plant_info.csv")
        row = df[df['plant_name'].str.strip() == plant_name.strip()]
        if row.empty:
            row = df[df['plant_name'].str.contains(plant_name, case=False, na=False)]
        return row.iloc[0].to_dict() if not row.empty else None
    except:
        return None

def load_treatments(plant_name):
    try:
        df = pd.read_csv("data/metadata/treatments.csv")
        treatments = df[df['plant_name'].str.strip() == plant_name.strip()]
        if treatments.empty:
            treatments = df[df['plant_name'].str.contains(plant_name, case=False, na=False)]
        return treatments.to_dict('records') if not treatments.empty else []
    except:
        return []

# ========== USER FUNCTIONS ==========
def register_user(username, email, password, fullname, db):
    try:
        if db.users.find_one({"username": username}):
            return False, "❌ Username exists!"
        db.users.insert_one({"username": username, "email": email, "password": password, "fullname": fullname, "created_at": datetime.now()})
        return True, "✅ Success!"
    except:
        return False, "❌ Error!"

def login_user(username, password, db):
    try:
        user = db.users.find_one({"username": username, "password": password})
        return (True, user) if user else (False, None)
    except:
        return False, None

def save_detection(user_id, plant_name, confidence, image_name, db):
    try:
        db.history.insert_one({"user_id": user_id, "plant_name": plant_name, "confidence": confidence, "image_name": image_name, "timestamp": datetime.now()})
    except:
        pass

def get_user_history(user_id, db):
    try:
        return list(db.history.find({"user_id": user_id}).sort("timestamp", -1).limit(20))
    except:
        return []

def get_all_users(db):
    try:
        return list(db.users.find().sort("created_at", -1))
    except:
        return []

def delete_user(db, user_id):
    try:
        db.users.delete_one({"_id": user_id})
        db.history.delete_many({"user_id": str(user_id)})
        return True
    except:
        return False

def add_user_by_admin(db, username, email, password, fullname):
    try:
        if db.users.find_one({"username": username}):
            return False, "❌ Username exists!"
        db.users.insert_one({"username": username, "email": email, "password": password, "fullname": fullname, "created_at": datetime.now()})
        return True, "✅ User added!"
    except:
        return False, "❌ Error!"

# ========== LOGIN PAGE ==========
def show_login_page(db):
    st.markdown("<div style='text-align: center; padding: 40px;'><h1 style='color: #4ade80;'>🌿 Herbal Plants Disease Detection</h1></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2.5, 1])
    with col2:
        st.markdown("<div style='background: rgba(30, 33, 57, 0.8); padding: 40px; border-radius: 20px; border: 2px solid rgba(74, 222, 128, 0.3);'>", unsafe_allow_html=True)
        
        col_login, col_register, col_admin = st.columns(3)
        with col_login:
            if st.button("🔐 Login", use_container_width=True, key="btn_login"):
                st.session_state['show_register'] = False
                st.rerun()
        with col_register:
            if st.button("📝 Register", use_container_width=True, key="btn_register"):
                st.session_state['show_register'] = True
                st.rerun()
        with col_admin:
            if st.button("🛡️ Admin", use_container_width=True, key="btn_admin"):
                st.session_state['show_register'] = 'admin'
                st.rerun()
        
        st.markdown("<div style='margin: 25px 0;'></div>", unsafe_allow_html=True)
        
        if not st.session_state['show_register']:
            with st.form("login_form"):
                st.markdown("<h3 style='color: #4ade80; text-align: center; margin-bottom: 20px;'>👤 User Login</h3>", unsafe_allow_html=True)
                username = st.text_input("📧 Email Address", placeholder="your.email@example.com", key="login_email")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_pass")
                submit = st.form_submit_button("Continue", use_container_width=True)
                if submit:
                    if username and password:
                        success, user = login_user(username, password, db)
                        if success:
                            st.session_state['user'] = user
                            st.session_state['user_type'] = 'user'
                            st.success("✅ Login successful!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials!")
                    else:
                        st.error("❌ Fill all fields!")
        
        elif st.session_state['show_register'] == True:
            with st.form("register_form"):
                st.markdown("<h3 style='color: #4ade80; text-align: center; margin-bottom: 20px;'>📝 Create New Account</h3>", unsafe_allow_html=True)
                fullname = st.text_input("👤 Full Name", placeholder="John Doe", key="reg_fullname")
                username = st.text_input("🆔 Username", placeholder="johndoe123", key="reg_username")
                email = st.text_input("📧 Email Address", placeholder="your.email@example.com", key="reg_email")
                password = st.text_input("🔒 Password", type="password", placeholder="Minimum 6 characters", key="reg_pass")
                confirm = st.text_input("🔒 Confirm Password", type="password", placeholder="Re-enter password", key="reg_confirm")
                submit = st.form_submit_button("Create Account", use_container_width=True)
                if submit:
                    if all([fullname, username, email, password, confirm]):
                        if password != confirm:
                            st.error("❌ Passwords don't match!")
                        elif len(password) < 6:
                            st.error("❌ Password must be at least 6 characters!")
                        else:
                            success, msg = register_user(username, email, password, fullname, db)
                            if success:
                                st.success(msg)
                                time.sleep(1)
                                st.session_state['show_register'] = False
                                st.rerun()
                            else:
                                st.error(msg)
                    else:
                        st.error("❌ Fill all fields!")
        
        else:
            st.markdown("<h3 class='admin-title' style='text-align: center; margin-bottom: 20px;'>🛡️ Admin Access</h3>", unsafe_allow_html=True)
            with st.form("admin_form"):
                admin_user = st.text_input("🛡️ Admin Username", placeholder="admin", key="admin_user")
                admin_pass = st.text_input("🔒 Admin Password", type="password", placeholder="Enter admin password", key="admin_pass")
                submit = st.form_submit_button("🔐 Admin Login", use_container_width=True)
                if submit:
                    if admin_user == ADMIN_USERNAME and admin_pass == ADMIN_PASSWORD:
                        st.session_state['user'] = {"fullname": "Admin", "username": "admin", "_id": "admin"}
                        st.session_state['user_type'] = 'admin'
                        st.success("✅ Admin access granted!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("❌ Invalid admin credentials!")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ========== DETECTION PAGE ==========
def show_detection_page(db):
    st.markdown("<div class='title-box'><h1>🔬 Plant Disease Detection</h1><p>Upload your training data leaf image</p></div>", unsafe_allow_html=True)
    
    if not os.path.exists('models/med117_model.h5'):
        st.error("❌ Model not found!")
        return
    
    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=['jpg','jpeg','png'])
    
    if uploaded_file:
        st.image(Image.open(uploaded_file), width=400, caption="📸 Uploaded Image")
        
        if st.button("🚀 PREDICT PLANT DISEASE", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            try:
                status.info("🔍 Checking...")
                progress.progress(20)
                is_valid, msg = validate_image_basic(uploaded_file)
                if not is_valid:
                    progress.empty()
                    status.empty()
                    st.error(msg)
                    return
                
                status.info("🤖 AI analyzing...")
                progress.progress(50)
                predictor = MED117Predictor()
                uploaded_file.seek(0)
                results = predictor.predict(uploaded_file, top_k=5)
                plant_name = results[0].get('plant_name', 'Unknown')
                confidence = results[0]['confidence']
                
                if confidence < 70:
                    progress.empty()
                    status.empty()
                    st.error(f"❌ Low confidence ({confidence:.1f}%) - Not a valid plant image!")
                    st.warning("💡 **Please upload a clear leaf image from your training data**")
                    return
                
                status.info("💾 Saving...")
                progress.progress(90)
                save_detection(str(st.session_state['user'].get('_id')), plant_name, confidence, uploaded_file.name, db)
                
                st.session_state['last_result'] = {
                    'plant_name': plant_name,
                    'confidence': confidence,
                    'plant_info': load_plant_info(plant_name),
                    'treatments': load_treatments(plant_name),
                    'results': results
                }
                
                progress.progress(100)
                time.sleep(0.3)
                progress.empty()
                status.empty()
                st.success(f"✅ Valid plant detected! Confidence: {confidence:.1f}%")
                st.rerun()
                    
            except Exception as e:
                progress.empty()
                status.empty()
                st.error(f"❌ Error: {e}")
    
    # DISPLAY RESULTS
    if st.session_state.get('last_result'):
        result = st.session_state['last_result']
        plant_name = result['plant_name']
        confidence = result['confidence']
        plant_info = result['plant_info']
        treatments = result['treatments']
        results = result['results']
        
        st.markdown(f"""
        <div class='result-card'>
            <h2 style='color: #4ade80;'>✨ PREDICTION RESULT ✨</h2>
            <h1 style='color: #fff; font-size: 2.5rem;'>🪴 {plant_name}</h1>
            <h3 style='color: #4ade80; font-size: 1.8rem;'>📊 Accuracy: {confidence:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        col_act1, col_act2, col_act3 = st.columns(3)
        
        with col_act1:
            pdf_buffer = create_detection_pdf(plant_name, confidence, plant_info, treatments, results)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer,
                file_name=f"Plant_Report_{plant_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        with col_act2:
            if st.button("📧 Send via Email", use_container_width=True, key="btn_email_dialog"):
                st.session_state['show_email_dialog'] = True
        
        with col_act3:
            if st.button("💾 Save to History", use_container_width=True):
                save_detection(str(st.session_state['user'].get('_id')), plant_name, confidence, "saved_result", db)
                st.success("💾 Saved to history!")
        
        # EMAIL DIALOG
        if st.session_state.get('show_email_dialog'):
            with st.expander("📧 Send Report via Email", expanded=True):
                user_email = st.session_state['user'].get('email', '')
                email_to = st.text_input("📧 Recipient Email Address", value=user_email, placeholder="recipient@email.com")
                
                col_send, col_cancel = st.columns(2)
                with col_send:
                    if st.button("✅ Send Email", use_container_width=True):
                        if email_to:
                            with st.spinner("📤 Sending email..."):
                                pdf_buffer_email = create_detection_pdf(plant_name, confidence, plant_info, treatments, results)
                                success, msg = send_email_with_pdf(email_to, plant_name, pdf_buffer_email)
                                if success:
                                    st.success(msg)
                                    st.session_state['show_email_dialog'] = False
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error(msg)
                        else:
                            st.error("❌ Please enter recipient email!")
                
                with col_cancel:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state['show_email_dialog'] = False
                        st.rerun()
        
        # PLANT INFO
        if plant_info:
            st.markdown("---")
            st.markdown("### 🌿 Plant Information")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class='info-card'>
                    <h4 style='color: #4ade80;'>🏷️ Common Name</h4>
                    <p><strong>{plant_info.get('common_name', 'N/A')}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='info-card'>
                    <h4 style='color: #4ade80;'>🔬 Scientific Name</h4>
                    <p><em>{plant_info.get('plant_name', 'N/A')}</em></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='info-card'>
                    <h4 style='color: #4ade80;'>👨‍👩‍👧‍👦 Family</h4>
                    <p><strong>{plant_info.get('family', 'N/A')}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                assamese = plant_info.get('assamese_name', 'Unknown')
                if assamese and assamese != 'Unknown':
                    st.markdown(f"""
                    <div class='info-card'>
                        <h4 style='color: #4ade80;'>🌏 Local Name (Assamese)</h4>
                        <p><strong>{assamese}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='info-card'>
                <h4 style='color: #4ade80;'>📖 Description</h4>
                <p>{plant_info.get('description', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"""
                <div class='info-card'>
                    <h4 style='color: #4ade80;'>💊 Medicinal Uses</h4>
                    <p>{plant_info.get('medicinal_uses', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='info-card'>
                    <h4 style='color: #4ade80;'>🌱 Parts Used</h4>
                    <p><strong>{plant_info.get('parts_used', 'N/A')}</strong></p>
                </div>
                """, unsafe_allow_html=True)
        
        # TREATMENTS
        if treatments:
            st.markdown("---")
            st.markdown("### 💊 Treatment Options")
            for idx, treatment in enumerate(treatments, 1):
                st.markdown(f"""
                <div class='treatment-card'>
                    <h4 style='color: #a78bfa;'>💊 Treatment #{idx}</h4>
                    <p><strong>Method:</strong> {treatment.get('treatment', 'N/A')}</p>
                    <p><strong>Application:</strong> {treatment.get('application', 'N/A')}</p>
                    <p><strong>Frequency:</strong> {treatment.get('frequency', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # TOP 5
        st.markdown("---")
        st.markdown("### 📊 Top 5 Predictions")
        for i, r in enumerate(results, 1):
            st.write(f"**#{i}:** {r.get('plant_name')} - {r.get('confidence', 0):.1f}%")

# ========== HISTORY PAGE ==========
def show_history_page(db):
    st.markdown("<div class='title-box'><h1>📜 Detection History</h1><p>Your previous plant detections</p></div>", unsafe_allow_html=True)
    
    user_id = str(st.session_state['user'].get('_id'))
    history = get_user_history(user_id, db)
    
    if history:
        st.markdown(f"### 📊 Total Detections: {len(history)}")
        st.markdown("---")
        
        for idx, record in enumerate(history, 1):
            timestamp = record.get('timestamp', datetime.now())
            time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if isinstance(timestamp, datetime) else str(timestamp)
            
            st.markdown(f"""
            <div class='history-card'>
                <h4 style='color: #fbbf24;'>🌿 Detection #{idx}</h4>
                <p><strong>Plant:</strong> {record.get('plant_name', 'Unknown')}</p>
                <p><strong>Confidence:</strong> {record.get('confidence', 0):.1f}%</p>
                <p><strong>Image:</strong> {record.get('image_name', 'N/A')}</p>
                <p><strong>Time:</strong> {time_str}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📭 No detection history found. Upload an image to start detecting!")

# ========== CLEAN ADMIN PAGE ==========
def show_admin_page(db):
    st.markdown("<div class='title-box'><h1>🛡️ Admin Panel</h1><p>Manage All Users</p></div>", unsafe_allow_html=True)
    
    # ADD USER SECTION IN CLEAN EXPANDER
    with st.expander("➕ Add New User", expanded=False):
        with st.form("add_user"):
            st.markdown("### 👤 Create New User Account")
            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("👤 Full Name", placeholder="John Doe")
                new_user = st.text_input("🆔 Username", placeholder="johndoe")
            with col2:
                new_email = st.text_input("📧 Email", placeholder="user@example.com")
                new_pass = st.text_input("🔒 Password", type="password", placeholder="Minimum 6 characters")
            
            if st.form_submit_button("➕ Add User", use_container_width=True):
                if all([new_name, new_user, new_email, new_pass]):
                    success, msg = add_user_by_admin(db, new_user, new_email, new_pass, new_name)
                    if success:
                        st.success(msg)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error("❌ Please fill all fields!")
    
    st.markdown("---")
    
    # ALL USERS IN CLEAN BOXES
    st.markdown("### 👥 All Users")
    users = get_all_users(db)
    
    if users:
        st.markdown(f"**Total Users:** {len(users)}")
        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
        
        for idx, u in enumerate(users, 1):
            # USER CARD
            st.markdown(f"""
            <div class='user-card'>
                <h3 style='color: #4ade80; margin-bottom: 15px;'>👤 {u.get('fullname', 'N/A')}</h3>
                <p style='font-size: 14px; margin: 8px 0;'><strong>Username:</strong> {u.get('username', 'N/A')}</p>
                <p style='font-size: 14px; margin: 8px 0;'><strong>Email:</strong> {u.get('email', 'N/A')}</p>
                <p style='font-size: 14px; margin: 8px 0;'><strong>Created:</strong> {str(u.get('created_at', 'N/A'))[:19]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # DELETE BUTTON
            if st.button(f"🗑️ Delete User", key=f"del_{u['_id']}", use_container_width=True):
                if delete_user(db, u['_id']):
                    st.success("✅ User deleted successfully!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Failed to delete user!")
            
            st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    else:
        st.info("👥 No users found in the database.")

# ========== MAIN ==========
def main():
    db = get_db()
    if st.session_state['user'] is None:
        show_login_page(db)
    else:
        with st.sidebar:
            st.markdown(f"<div style='padding: 20px; text-align: center; background: #1e2139; border-radius: 12px; border: 2px solid #4ade80;'><h3 style='color: #4ade80;'>👤 {st.session_state['user'].get('fullname')}</h3></div>", unsafe_allow_html=True)
            
            if st.session_state['user_type'] == 'admin':
                if st.button("🛡️ Admin Panel", use_container_width=True):
                    st.session_state['current_page'] = "🛡️ Admin"
                    st.rerun()
            else:
                if st.button("🔬 Detection", use_container_width=True):
                    st.session_state['current_page'] = "🔬 Detection"
                    st.session_state['last_result'] = None
                    st.rerun()
                if st.button("📜 History", use_container_width=True):
                    st.session_state['current_page'] = "📜 History"
                    st.rerun()
            
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state['user'] = None
                st.session_state['last_result'] = None
                st.rerun()
        
        if st.session_state['user_type'] == 'admin':
            show_admin_page(db)
        else:
            if st.session_state['current_page'] == "🔬 Detection":
                show_detection_page(db)
            elif st.session_state['current_page'] == "📜 History":
                show_history_page(db)

if __name__ == "__main__":
    main()
