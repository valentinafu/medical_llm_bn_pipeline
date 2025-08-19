# import streamlit as st
# from sqlalchemy.orm import Session
# from db.database import SessionLocal
# from authenticator.login import login
# from authenticator.signup import signup
# from models.models import User
# from views.admin_view import admin_view
# from views.patient_view import patient_view
#
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
#
# def main():
#     st.set_page_config(page_title="Medical Inference App", page_icon="🏥")
#
#     # Initialize session state
#     if "page" not in st.session_state:
#         st.session_state.page = "auth"
#     feedback = st.empty()  # Placeholder for session debug
#     feedback.write(f"Debug: Page state = {st.session_state.page}")
#     feedback.write(f"Debug: Session state keys: {list(st.session_state.keys())}")
#
#     # Check session persistence
#     if "session_marker" in st.session_state:
#         feedback.write(f"Debug: Session marker found: {st.session_state.session_marker}")
#     else:
#         feedback.write("Debug: No session marker - session may have cleared")
#
#     # Get database session
#     db: Session = next(get_db())
#
#     # Restore session if user_id exists
#     if "user_id" in st.session_state and "user" not in st.session_state:
#         feedback.write(f"Debug: Attempting to restore session for user_id={st.session_state.user_id}")
#         try:
#             user = db.query(User).filter_by(id=st.session_state.user_id).first()
#             if user:
#                 st.session_state.user = user
#                 st.session_state.user_email = user.email
#                 st.session_state.page = "dashboard"
#                 feedback.write(f"Debug: Session restored, user={user.name}, role={user.role}")
#             else:
#                 feedback.write("Debug: No user found for user_id, clearing session")
#                 st.session_state.clear()
#                 st.session_state.page = "auth"
#         except Exception as e:
#             feedback.error(f"Debug: Database error during restoration: {str(e)}")
#             st.session_state.clear()
#             st.session_state.page = "auth"
#
#     # Handle authentication or dashboard
#     if "user" not in st.session_state or st.session_state.page == "auth":
#         auth_mode = st.radio("Choose action", ["Login", "Sign Up"], key="auth_mode")
#         feedback.write(f"Debug: Auth mode = {auth_mode}")
#         if auth_mode == "Login":
#             login(db)
#         else:
#             if signup(db):
#                 st.session_state.page = "auth"
#                 st.rerun()
#     else:
#         st.success(f"Logged in as {st.session_state.user.name} ({st.session_state.user.role})")
#         feedback.write(f"Debug: Displaying dashboard for user={st.session_state.user.name}")
#         if st.button("Logout"):
#             st.session_state.clear()
#             st.session_state.page = "auth"
#             feedback.write("Debug: Logged out")
#             st.rerun()
#
#         if st.session_state.user.role == "patient":
#             patient_view()
#         elif st.session_state.user.role == "admin":
#             admin_view()
#
# if __name__ == "__main__":
#     main()

import streamlit as st
from sqlalchemy.orm import Session
from db.database import SessionLocal
from authenticator.login import login
from authenticator.signup import signup
from models.models import User
from views.admin_view import admin_view
from views.patient_view import patient_view


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def main():
    st.set_page_config(page_title="Medical Inference App", page_icon="🏥")

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "auth"

    # Debug info (remove in production)
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.write(f"Page state: {st.session_state.page}")
        st.sidebar.write(f"Session keys: {list(st.session_state.keys())}")
        if "user" in st.session_state:
            st.sidebar.write(f"User: {st.session_state.user.name} ({st.session_state.user.role})")

    # Get database session
    db: Session = next(get_db())

    # Restore session if user_id exists but user object is missing
    if "user_id" in st.session_state and "user" not in st.session_state:
        try:
            user = db.query(User).filter_by(id=st.session_state.user_id).first()
            if user:
                st.session_state.user = user
                st.session_state.user_email = user.email
                st.session_state.page = "dashboard"
            else:
                # User not found, clear session
                st.session_state.clear()
                st.session_state.page = "auth"
        except Exception as e:
            st.error(f"Session restoration error: {str(e)}")
            st.session_state.clear()
            st.session_state.page = "auth"

    # Main app logic
    if "user" not in st.session_state or st.session_state.page == "auth":
        # Show authentication page
        st.title("🏥 Medical Inference App")

        auth_mode = st.radio("Choose action", ["Login", "Sign Up"], key="auth_mode")

        if auth_mode == "Login":
            if login(db):
                # Login successful, redirect based on role
                user = st.session_state.user
                if user.role == "patient":
                    st.session_state.page = "patient"
                elif user.role == "admin":
                    st.session_state.page = "admin"
                else:
                    st.session_state.page = "dashboard"  # fallback
                st.rerun()
        else:
            if signup(db):
                st.success("Account created successfully! Please log in.")
                st.session_state.page = "auth"
                st.rerun()
    else:
        # User is logged in - show appropriate view
        user = st.session_state.user

        # Header with user info and logout
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title(f"🏥 Medical App - {user.role.title()} Portal")
        with col2:
            if st.button("🚪 Logout", type="secondary"):
                st.session_state.clear()
                st.session_state.page = "auth"
                st.rerun()

        st.caption(f"Welcome back, **{user.name}** ({user.email})")
        st.divider()

        # Role-based view routing
        if user.role == "patient":
            patient_view()
        elif user.role == "admin":
            admin_view()
        else:
            st.error(f"Unknown user role: {user.role}")
            st.info("Please contact support.")


if __name__ == "__main__":
    main()