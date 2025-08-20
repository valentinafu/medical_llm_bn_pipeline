""""
This module implements the **Patient View** of the Medical Inference App.
It provides an interactive Streamlit interface where patients can report their symptoms,
receive probabilistic health assessments from them, and view their assessment history.
Purpose : Provides patients with an AI-assisted tool for symptom-based health assessments,
         backed by Bayesian Networks stored in Neo4j and SQLAlchemy.
Key features:
1. Evaluation Management:
   - _get_or_create_eval_for_category(): Ensures an Evaluation object exists for a patient condition.
   - get_bn_json_for_category(): Retrieves stored Bayesian Network JSON (if available).
   - run_analysis(): Runs Bayesian inference for a condition using patient's symptoms ,
     updates the Evaluation, and displays results.

2. Database & Neo4j Connectivity:
   - get_connection(): Connects to Neo4j (via Neo4jUploader).
   - load_categories(): Fetches available categories/conditions from Neo4j.
   - load_symptoms_for_category(): Retrieves symptoms related to selected category.

3. User Interface:
   - patient_view(): Main entrypoint for patients.
        * Tab 1: "New Assessment", lets users select a condition, answer symptom questions,
          and generates a Bayesian inference-based assessment.
        * Tab 2: "History" shows past assessments with predictions and reported symptoms.
   - display_assesment_results(): Displays probabilities in human friendly form (low, moderate, high likelihood).
   - show_patient_history(): Lists up to 10 most recent assessments for the patient.

4. Utilities:
   - clean normalization helper `_norm()` for string processing.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from db.database import SessionLocal
from models.models import Evaluation
from neo4jUploader import Neo4jUploader
from baysian import BayesianNetworkBuilder
# load environment variables
load_dotenv()
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")


# find existing evaluation or create new one for user and category
def _get_or_create_eval_for_category(db, user_id: int, category: str, evidence_dict: dict) -> Evaluation:
    ev = (
        db.query(Evaluation)
        .filter(Evaluation.user_id == user_id, Evaluation.category == category)
        .order_by(Evaluation.timestamp.desc())
        .first()
    )

    if ev:
        ev.timestamp = datetime.utcnow()
        ev.symptoms = json.dumps(evidence_dict)
        ev.target_node = category
        ev.status = "pending"
    else:
        ev = Evaluation(
            user_id=user_id,
            timestamp=datetime.utcnow(),
            symptoms=json.dumps(evidence_dict),
            prediction="",
            target_node=category,
            status="pending",
            category=category
        )

    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev


# build Bayesian Network from JSON for a category in evaluation.
def get_bn_json_for_category(db, category: str) -> Optional[dict]:
    q = db.query(Evaluation).filter(Evaluation.category == category)

    if hasattr(Evaluation, "llm_response"):
        q = q.filter(Evaluation.llm_response.isnot(None))
    elif hasattr(Evaluation, "output_from_llm"):
        q = q.filter(Evaluation.output_from_llm.isnot(None))
    ev_bn = q.order_by(Evaluation.timestamp.desc()).first()
    if not ev_bn:
        return None

    raw = getattr(ev_bn, "llm_response", None) or getattr(ev_bn, "output_from_llm", None)
    try:
        return json.loads(raw) if raw else None
    except json.JSONDecodeError as e:
        st.error(f"Model data corrupted for {category}: {e}")
        return None


@st.cache_resource
# Connection with Neo4J
def get_connection():
    try:
        return Neo4jUploader(uri=uri, user=user, password=password)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


def load_categories(conn) -> List[str]:
    if not conn:
        return []
    try:
        cats = conn.get_categories() or []
        return sorted(set(cats), key=lambda x: x.lower())
    except Exception as e:
        st.error(f"Failed to load categories : {e}")
        return []


def load_symptoms_for_category(conn, category: str) -> List[str]:
    if not conn:
        return []
    try:
        return conn.get_symptoms(category) or []
    except Exception as e:
        st.error(f"Failed to load symptoms for {category}: {e}")
        return []


def display_assesment_results(probabilities: Dict[str, float]) -> bool:
    if not probabilities:
        st.warning("Unable to generate prediction at this time.")
        return False

    st.markdown("### Assessment Results")

    show_treatment_suggestions = False

    for state, prob in probabilities.items():
        percentage = round(prob * 100, 1)

        if state.lower() == "present":
            if percentage > 70:
                st.error(f"🔴**High likelihood** of condition: {percentage}%")
                show_treatment_suggestions = True
            elif percentage > 40:
                st.warning(f"🟡**Moderate likelihood** of condition: {percentage}%")
                show_treatment_suggestions = True
            else:
                st.success(f"🟢**Low likelihood** of condition: {percentage}%")
        else:
            if percentage > 70:
                st.success(f"Low risk indicated: {percentage}% chance condition is absent")

    st.info(
        "💡**Remember**: This is an AI assessment tool. Always consult with healthcare professionals for medical advice.")

    return show_treatment_suggestions


def run_analysis(db, user_id: int, category: str, symptoms: Dict[str, int]):
    try:
        evaluation = _get_or_create_eval_for_category(db, user_id, category, symptoms)
        bn_json = get_bn_json_for_category(db, category)
        if not bn_json:
            evaluation.status = "no_model"
            db.commit()
            return None, None

        builder = BayesianNetworkBuilder.from_llm_response(bn_json)
        probs, info = builder.infer_with_evidence_filtering(category, symptoms)
        display_assesment_results(probs)
        evaluation.status = "ok"
        db.commit()
        return probs, info

    except Exception as e:
        evaluation.status = "failed"
        db.commit(evaluation)
        st.warning("No model is available yet for this condition or analysis failed...", icon="⚠️")
        with st.expander("Error details"):
            st.exception(e)
        return None, None


def show_patient_history(user_id: int):
    st.markdown("### Your Assessment History")

    db = SessionLocal()
    try:
        evaluations = (
            db.query(Evaluation)
            .filter(Evaluation.user_id == user_id)
            .order_by(Evaluation.timestamp.desc())
            .limit(10)
            .all()
        )

        if not evaluations:
            st.info("No previous assessments found.")
            return

        for eval in evaluations:
            with st.expander(f"{eval.category} - {eval.timestamp.strftime('%Y-%m-%d %H:%M')}"):
                if eval.prediction:
                    try:
                        pred_data = json.loads(eval.prediction)
                        for state, prob in pred_data.items():
                            percentage = round(prob * 100, 1)
                            if state.lower() == "present":
                                if percentage > 70:
                                    st.error(f"High likelihood: {percentage}%")
                                elif percentage > 40:
                                    st.warning(f"Moderate likelihood: {percentage}%")
                                else:
                                    st.success(f"Low likelihood: {percentage}%")
                    except Exception as e:
                        st.text(eval.prediction)

                if eval.symptoms:
                    try:
                        symptoms_data = json.loads(eval.symptoms)
                        positive_symptoms = [k for k, v in symptoms_data.items() if v == 1]
                        if positive_symptoms:
                            st.write("**Reported symptoms:**", ", ".join(positive_symptoms))
                    except:
                        pass

    except Exception as e:
        st.error(f"Failed to load history: {e}")
    finally:
        db.close()


def patient_view():
    st.title("Health Assessment Tool")
    user = st.session_state.get('user')
    user_id = getattr(user, "id", None) if user else None
    if not user_id:
        st.error("Please log in to continue.")
        return

    st.markdown(f"## Welcome, {user.name}!")

    tab1, tab2 = st.tabs(["New Assessment", "History"])

    with tab1:
        st.markdown("Tell us about your symptoms and we'll provide an assessment.")
        conn = get_connection()
        if not conn:
            st.error("Service temporarily unavailable. Please try again later.")
            return

        categories = load_categories(conn)
        if not categories:
            st.warning("No assessments available at this time.")
            return

        selected_category = st.selectbox(
            "What condition would you like to assess?",
            options=categories,
            help="Select the medical condition you're concerned about"
        )

        if not selected_category:
            return

        graph_symptoms = load_symptoms_for_category(conn, selected_category)
        if not graph_symptoms:
            st.warning(f"Assessment not available for {selected_category}.")
            return

        st.markdown(f"### Questions about {selected_category}")
        st.markdown("Please answer yes or no to each question:")

        symptoms_data = {}

        with st.form("assessment_form"):
            for symptom in graph_symptoms:
                display_name = symptom.replace('_', ' ').replace('-', ' ').title()

                answer = st.radio(
                    f"Do you have {display_name.lower()}?",
                    options=["No", "Yes"],
                    key=f"q_{symptom}",
                    horizontal=True
                )

                symptoms_data[symptom] = 1 if answer == "Yes" else 0

            submitted = st.form_submit_button("🔍 Get Assessment", type="primary")
        if submitted:
            positive_count = sum(symptoms_data.values())

            if positive_count > 0:
                st.info(f"Analyzing your {positive_count} symptoms you reported...")
            else:
                st.info("Analyzing your responses...")

            with st.spinner("Processing..."):
                db = SessionLocal()
                try:
                    result = run_analysis(db, user_id, selected_category, symptoms_data)
                    if result and len(result) == 2:
                        probabilities, builder = result
                    else:
                        st.warning("No model is available for this condition or analysis failed.")
                except Exception as e:
                    st.error("Assessment failed. Please try again.")
                    if st.checkbox("Show error details", key="show_errors"):
                        st.exception(e)
                finally:
                    db.close()

    with tab2:
        show_patient_history(user_id)


import unicodedata, re


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").strip()
    return s.casefold()
