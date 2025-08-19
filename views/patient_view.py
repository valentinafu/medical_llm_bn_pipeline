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
from treatments import BNTreatmentInference

# Load environment variables
load_dotenv()
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")


def _get_or_create_eval_for_category(db, user_id: int, category: str, evidence_dict: dict) -> Evaluation:
    """Get existing evaluation or create new one for user+category."""
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


def _get_bn_json_for_category(db, category: str) -> Optional[dict]:
    """Retrieve saved Bayesian Network JSON for a category."""
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
def get_connection():
    """Create cached Neo4j connection."""
    try:
        return Neo4jUploader(uri=uri, user=user, password=password)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


def load_categories(conn) -> List[str]:
    """Load available categories from Neo4j."""
    if not conn:
        return []
    try:
        cats = conn.get_categories() or []
        return sorted(set(cats), key=lambda x: x.lower())
    except Exception as e:
        st.error(f"Failed to load categories: {e}")
        return []


def load_symptoms(conn, category: str) -> List[str]:
    """Load symptoms for a given category."""
    if not conn:
        return []
    try:
        return conn.get_symptoms(category) or []
    except Exception as e:
        st.error(f"Failed to load symptoms for {category}: {e}")
        return []


def display_simple_results(probabilities: Dict[str, float]) -> bool:
    """Display clean, simple prediction results for patients. Returns True if risk is high enough for treatment suggestions."""
    if not probabilities:
        st.warning("Unable to generate prediction at this time.")
        return False

    st.markdown("### Assessment Results")

    show_treatment_suggestions = False

    for state, prob in probabilities.items():
        percentage = round(prob * 100, 1)

        if state.lower() == "present":
            if percentage > 70:
                st.error(f"🔴 **High likelihood** of condition: {percentage}%")
                show_treatment_suggestions = True
            elif percentage > 40:
                st.warning(f"🟡 **Moderate likelihood** of condition: {percentage}%")
                show_treatment_suggestions = True
            else:
                st.success(f"🟢 **Low likelihood** of condition: {percentage}%")
        else:
            # Show absence probability in a subtle way
            if percentage > 70:
                st.success(f"✅ Low risk indicated: {percentage}% chance condition is absent")

    # Simple disclaimer
    st.info(
        "💡**Remember**: This is an AI assessment tool. Always consult with healthcare professionals for medical advice.")

    return show_treatment_suggestions

def run_analysis(db, user_id: int, category: str, symptoms: Dict[str, int]):
    try:
        evaluation = _get_or_create_eval_for_category(db, user_id, category, symptoms)
        bn_json = _get_bn_json_for_category(db, category)
        if not bn_json:
            evaluation.status = "no_model"
            db.commit()
            return None, None

        # Build BN from JSON so nodes/CPDs exist
        builder = BayesianNetworkBuilder.from_llm_response(bn_json)

        # Pull Neo4j triples (scoped to your category!)
        # triples, nodes_std, alias_map, canon_map = get_connection().get_causal_triples(category=category)

        # Auto-derive aliases so raw Neo4j/evidence labels map to BN node strings
        # auto_alias = _auto_aliases_from_triples(builder, triples, extra_labels=symptoms.keys())
        # for raw, real in auto_alias.items():
        #     # avoid no-op alias (same normalized target)
        #     if builder._normalize_node_name(raw) != builder._normalize_node_name(real):
        #         builder.add_node_alias(raw, real)

        probs, info = builder.infer_with_evidence_filtering(category, symptoms)

        evaluation.status = "ok"
        db.commit()
        return probs, info

    except Exception as e:
        evaluation.status = "failed"
        db.commit(evaluation)
        st.warning("No model available yet for this condition or analysis failed.", icon="⚠️")
        with st.expander("Error details"):
            st.exception(e)
        return None, None

def _auto_aliases_from_triples(builder, triples, extra_labels=None):
    """
    Map raw labels from Neo4j (and any extra labels like evidence keys)
    to the EXACT node strings in the built model.
    """
    name_map = {builder._normalize_node_name(n): n for n in builder.get_available_nodes()}
    raw_labels = set()
    for s, _, t in triples:
        raw_labels.add(s); raw_labels.add(t)
    for lab in (extra_labels or []):
        raw_labels.add(lab)

    alias = {}
    for raw in raw_labels:
        key = builder._normalize_node_name(raw)
        if key in name_map:
            alias[raw] = name_map[key]
    return alias

def add_treatment_reasoning_to_patient_view(bn_builder: BayesianNetworkBuilder,
                                            condition: str, evidence: Dict[str, any]):
    """
    Add treatment reasoning to the patient dashboard.
    Call this after showing prediction results.
    """
    st.markdown("---")

    # Debug info
    st.markdown("### 🧠 AI Treatment Reasoning (Experimental)")
    st.info("This experimental feature uses the Bayesian Network to identify which symptoms, "
            "if addressed, might most effectively reduce your risk.")

    # Debug: Check if builder is valid
    if bn_builder is None:
        st.error("❌ No Bayesian Network available for treatment reasoning")
        return

    st.success("✅ Bayesian Network loaded successfully")

    # Use a unique key for the button to avoid conflicts
    if st.button("🔬 Generate Treatment Insights", key="treatment_insights_btn", type="primary"):
        st.write("Button clicked! Processing...")  # Debug message

        try:
            with st.spinner("Analyzing treatment possibilities..."):
                st.write(f"Creating treatment reasoner for condition: {condition}")  # Debug
                treatment_reasoner = BNTreatmentInference(bn_builder)

                st.write("Generating suggestions...")  # Debug
                suggestions = treatment_reasoner.generate_treatment_suggestions(condition, evidence)

                st.write(f"Got suggestions: {suggestions}")  # Debug

        except Exception as e:
            st.error(f"Error generating treatment insights: {str(e)}")
            st.exception(e)


def show_patient_history(user_id: int):
    """Display patient's assessment history."""
    st.markdown("### 📋 Your Assessment History")

    db = SessionLocal()
    try:
        evaluations = (
            db.query(Evaluation)
            .filter(Evaluation.user_id == user_id)
            .filter(Evaluation.status == "completed")
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
                        for state, prob in  pred_data.items():
                            percentage = round(prob * 100, 1)
                            if state.lower() == "present":
                                if percentage > 70:
                                    st.error(f"High likelihood: {percentage}%")
                                elif percentage > 40:
                                    st.warning(f"Moderate likelihood: {percentage}%")
                                else:
                                    st.success(f"Low likelihood: {percentage}%")
                    except:
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
    """Simple patient dashboard focused on core functionality."""
    st.title("🏥 Health Assessment Tool")

    # Check authentication
    user = st.session_state.get('user')
    user_id = getattr(user, "id", None) if user else None

    if not user_id:
        st.error("Please log in to continue.")
        return

    # Simple welcome
    st.markdown(f"## Welcome, {user.name}! 👋")

    # Tabs for different sections
    tab1, tab2 = st.tabs(["🔍 New Assessment", "📋 History"])

    with tab1:
        st.markdown("Tell us about your symptoms and we'll provide an assessment.")

        # Get connection and data
        conn = get_connection()
        if not conn:
            st.error("Service temporarily unavailable. Please try again later.")
            return

        categories = load_categories(conn)
        if not categories:
            st.warning("No assessments available at this time.")
            return

        # Category selection
        selected_category = st.selectbox(
            "What condition would you like to assess?",
            options=categories,
            help="Select the medical condition you're concerned about"
        )

        if not selected_category:
            return

        # Load symptoms
        graph_symptoms = load_symptoms(conn, selected_category)
        if not graph_symptoms:
            st.warning(f"Assessment not available for {selected_category}.")
            return

        # Simple symptom form
        st.markdown(f"### Questions about {selected_category}")
        st.markdown("Please answer yes or no to each question:")

        symptoms_data = {}

        with st.form("assessment_form"):
            for symptom in graph_symptoms:
                # Clean up symptom name for display
                display_name = symptom.replace('_', ' ').replace('-', ' ').title()

                answer = st.radio(
                    f"Do you have {display_name.lower()}?",
                    options=["No", "Yes"],
                    key=f"q_{symptom}",
                    horizontal=True
                )

                symptoms_data[symptom] = 1 if answer == "Yes" else 0

            # Submit
            submitted = st.form_submit_button("🔍 Get Assessment", type="primary")

        # Process results
        if submitted:
            positive_count = sum(symptoms_data.values())

            if positive_count > 0:
                st.info(f"Analyzing your {positive_count} reported symptoms...")
            else:
                st.info("Analyzing your responses...")

            with st.spinner("Processing..."):
                db = SessionLocal()
                try:
                    result = run_analysis(db, user_id, selected_category, symptoms_data)
                    if result and len(result) == 2:
                        probabilities, builder = result
                        if probabilities and builder:
                            # Display results
                            show_treatments = display_simple_results(probabilities)
                            if show_treatments:
                                try:
                                    add_treatment_reasoning_to_patient_view(builder, selected_category, symptoms_data)
                                except Exception as e:
                                    st.info("Treatment reasoning is not available for this condition yet.")
                        else:
                            st.warning("Unable to complete assessment. Please try again or contact support.")
                    else:
                        st.warning("No model available for this condition or analysis failed.")

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
    # normalize for duplicate detection (no mutation of the model!)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").strip()
    return s.casefold()

# def bn_healthcheck(model, bn_json=None, evidence_dict=None, target=None):
#     print("=== NODES IN MODEL ===")
#     nodes = list(model.model.nodes())
#     for n in sorted(nodes):
#         codes = " ".join(f"U+{ord(c):04X}" for c in n)
#         print(f"- {repr(n)}  (len={len(n)}  codes={codes})")
#     print(f"Total nodes: {len(nodes)}\n")
#
#     # Duplicates that differ only by case/spacing/unicode
#     print("=== NEAR-DUPLICATE LABELS (by normalized form) ===")
#     buckets = {}
#     for n in nodes:
#         buckets.setdefault(_norm(n), []).append(n)
#     dupes = [v for v in buckets.values() if len(v) > 1]
#     if dupes:
#         for v in dupes: print("•", v)
#     else:
#         print("None")
#     print()
#
#     # Zero-width / NBSP check
#     zw = re.compile(r"[\u200B-\u200D\uFEFF\u00A0]")
#     bad = [n for n in nodes if zw.search(n)]
#     print("=== HIDDEN-CHAR NODES (ZW/NBSP) ===")
#     if bad:
#         for n in bad: print("•", repr(n))
#     else:
#         print("None")
#     print()
#
#     if bn_json:
#         expected = set(bn_json["nodes"])
#         print("=== DIFF: JSON nodes vs model nodes ===")
#         print("Missing in MODEL:", sorted(expected - set(nodes)))
#         print("Extra in MODEL :", sorted(set(nodes) - expected))
#         # Also check that all names used in edges/cpds exist in nodes
#         used = set()
#         for e in bn_json.get("edges", []):
#             used.add(e["from"]); used.add(e["to"])
#         for c in bn_json.get("cpds", []):
#             used.add(c["node"]); used.update(c.get("given", []))
#         print("Referenced but NOT in JSON 'nodes':", sorted(used - expected))
#         print()
#
#     # CPD state-name sanity + show present/absent order
#     print("=== CPD STATE ORDER ===")
#     for cpd in model.model.get_cpds():
#         var = cpd.variable
#         sn = cpd.state_names.get(var)
#         print(f"{var}: states={sn}")
#         for p in cpd.variables[1:]:
#             print(f"  parent {p}: states={cpd.state_names.get(p)}")
#     print()
#
#     # Basic model check
#     print("=== CHECK MODEL ===")
#     try:
#         ok = model.model.check_model()
#         print("check_model():", ok)
#     except Exception as e:
#         print("check_model() raised:", e)
#     print()
#
#     # Prior/posterior quick look for target (optional)
#     if target:
#         try:
#             inf = VariableElimination(model)
#             prior = inf.query([target])[target]
#             print(f"Prior P({target}):", {sn: float(prior.values[i]) for i, sn in enumerate(model.get_cpds(target).state_names[target])})
#             if evidence_dict:
#                 post = inf.query([target], evidence=evidence_dict)[target]
#                 print(f"Posterior P({target} | evidence):", {sn: float(post.values[i]) for i, sn in enumerate(model.get_cpds(target).state_names[target])})
#         except Exception as e:
#             print("Inference error:", e)
