import streamlit as st
from sqlalchemy.orm import Session
from models.models import User

def login(db: Session):
    st.title("Login")
    feedback = st.empty()
    email = st.text_input("Email").strip()
    password = st.text_input("Password", type="password")
    feedback.write(f"Debug: Email entered: '{email}'")
    feedback.write(f"Debug: Session state before login: {list(st.session_state.keys())}")
    try:
        test_user = db.query(User).filter_by(email="valentinafurxhi@gmail.com").first()
        feedback.write(f"Debug: Hardcoded query result: {test_user.email if test_user else 'None'}")
        all_users = db.query(User).all()
        feedback.write(f"Debug: All users: {[u.email for u in all_users]}")
    except Exception as e:
        feedback.error(f"Debug: Database error: {str(e)}")

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
                    feedback.write(f"Debug: Session state after login: {list(st.session_state.keys())}")
                    st.rerun()
                    return True
                else:
                    feedback.error("Invalid password")
                    return False
            else:
                feedback.error("No user found with this email")
                feedback.write(f"Debug: No user found for email='{email}'")
                return False
        except Exception as e:
            feedback.error(f"Database error: {str(e)}")
            return False
    return False

