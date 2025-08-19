import streamlit as st
from sqlalchemy.orm import Session
from models.models import User

def signup(db: Session):
    st.title("Sign Up")

    name = st.text_input("Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    role = st.selectbox("Role", ["patient", "expert", "admin"])

    if st.button("Sign Up"):
        if db.query(User).filter_by(email=email).first():
            st.error("Email already exists")
            st.write("Debug: Signup failed - email exists")
            return False

        try:
            new_user = User(name=name, email=email, password=password, role=role)
            db.add(new_user)
            db.commit()
            db.refresh(new_user)

            st.session_state.user = new_user
            st.session_state.user_id = new_user.id
            st.session_state.user_email = new_user.email
            st.session_state.page = "dashboard"

            st.success("Sign up successful!")
            st.write(f"Debug: Signup successful, user_id={new_user.id}")
            st.rerun()
            return True

        except Exception as e:
            st.error("Error during signup")
            st.write(f"Debug: Signup error - {str(e)}")
            db.rollback()
            return False

    st.write("Debug: Signup form displayed")
    return False
