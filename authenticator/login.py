import streamlit as st
from sqlalchemy.orm import Session
from models.models import User

def login(db: Session):
    st.title("Login")
    feedback = st.empty()
    email = st.text_input("Email").strip()
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        feedback.write("Checking credentials...")
        try:
            if not email:
                feedback.error("Email cannot be empty")
                return False
            user = db.query(User).filter_by(email=email).first()
            if user:
                feedback.write(f"Debug: User found: ID={user.id}, Email={user.email}")
                if user.password == password:
                    st.session_state.user = user
                    st.session_state.user_id = user.id
                    st.session_state.user_email = user.email
                    st.session_state.page = "dashboard"
                    st.session_state.session_marker = "active"
                    feedback.success(f"Welcome, {user.name}!")
                    feedback.write(f"Debug: Login successful, user_id={user.id}")
                    st.rerun()
                    return True
                else:
                    feedback.error("Invalid password")
                    return False
            else:
                feedback.error("No user found with this email")
                feedback.write(f"Debug:No user found for email='{email}'")
                return False
        except Exception as e:
            feedback.error(f"Database error: {str(e)}")
            return False
    return False

