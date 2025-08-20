import json
import os
from datetime import datetime
from dotenv import load_dotenv
from models.models import Evaluation, User
from neo4jUploader import Neo4jUploader
from open_ai_connector import extract_triples, generate_bn_structure_and_probabilities_from_llm
from baysian import BayesianNetworkBuilder
from db.database import SessionLocal
from web_scraper import scrape_page
import re
from contextlib import contextmanager
from bs4 import BeautifulSoup

# This module implements the **Admin Panel** of the Medical Inference App, providing
# workflows to extract knowledge from clinical/medical text/URLs, save the results as triple it in Neo4j, and
# construct Bayesian Networks (BNs) for medical inference.
#
# Key features:
# 1. Role Enforcement:
#    - require_admin_role():  access to admin role only views.
#
# 2. Triple processing helpers:
#    - lc_triples(): lowercases triples for consistency.
#    - _noisy_or_prob(): calculates noisy-OR probabilities for CPDs.
#    - _stable_ns(): generates unique, stable widget keys for Streamlit forms.
#    - _is_noisy_or(): checks if a CPD uses a noisy-OR model or not.
#
# 3. Neo4j and DB Integration:
#    - get_conn(): connects to Neo4j using environment variables.
#    - create_or_get_evaluation_for_category(): ensures an Evaluation object exists for a condition.
#    - load_or_generate_bn_json(): retrieves or regenerates BN JSON from LLM.
#
# 4. Bayesian Network Construction:
#    - build_bn_for_evaluation_no_evidence(): generates Bayesian Network from Neo4j triples,
#      runs LLM to produce structure or CPDs, saves results, and visualizes the network.
#    - update_cpd_in_full_json(): updates Bayesian Network JSON with edited CPDs.
#
# 5. CPT Editing (Streamlit UI):
#    - display_cpds_dropdown(): lets users interactively edit CPDs (Noisy-OR or Tabular Cases)
#      with sliders/inputs and preview updated probabilities.
#
# 6. Admin Views:
#    - admin_view(): main admin panel — allows category selection, text/URL extraction,
#      triple upload to Neo4j, BN creation, and CPD editing.
#    - render_evaluation_panel(): shows details for a selected evaluation,
#      displays the Bayesian Network graph, and enables CPT updates.
#
# 7. Category Inference:
#    - infer_category(), infer_category_from_url(), extract_title_from_html(),
#      clean_category(): heuristics to guess clean category names from user input,
#      URLs, or HTML titles.
#
# Purpose: Provides admins/experts with the ability to extract structured knowledge
# from clinical content, organize data Neo4j, and build/edit Bayesian Networks
# to support medical inference and decision-making.



db = SessionLocal()

def require_admin_role():
    user = st.session_state.get('user')
    if not user or user.role != "admin":
        st.error("Access Denied")
        st.info("This page requires admin privileges.")
        st.stop()
    return user


# ----------------- Helper Functions -----------------

def _lc(x):
    return str(x).lower().strip()


def lc_triples(triples):
    return [(_lc(a), _lc(r), _lc(b)) for (a, r, b) in triples]


def _noisy_or_prob(x_bits, weights_map, leak=0.0, given_order=None):
    given_order = given_order or list(weights_map.keys())
    prod = 1.0 - float(leak)
    for parent, bit in zip(given_order, x_bits):
        if bit:
            w = float(weights_map.get(parent, 0.0))
            prod *= (1.0 - w)
    p1 = 1.0 - prod
    return max(0.0, min(1.0, p1))


def _stable_ns(node: str, given: list | None = None, tag: str = "") -> str:
    given = given or []
    base = f"{node}|{','.join(given)}|{tag}"
    return f"{node}__{tag}__{hashlib.md5(base.encode()).hexdigest()[:8]}"


def _is_noisy_or(cpd: dict) -> bool:
    return isinstance(cpd.get("noisy_or_weights"), dict)


def get_conn() -> Neo4jUploader:
    load_dotenv()
    return Neo4jUploader(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD")
    )


def _get_saved_llm_json(evaluation: Evaluation):
    return getattr(evaluation, "llm_response", None) or getattr(evaluation, "output_from_llm", None)


def _set_saved_llm_json(evaluation: Evaluation, response_dict: dict):
    payload = json.dumps(response_dict, indent=2)
    if hasattr(evaluation, "llm_response"):
        evaluation.llm_response = payload
    elif hasattr(evaluation, "output_from_llm"):
        evaluation.output_from_llm = payload
    else:
        evaluation.llm_response = payload


def create_or_get_evaluation_for_category(db, category: str) -> Evaluation:
    ev = (
        db.query(Evaluation)
        .filter(Evaluation.category == category)
        .order_by(Evaluation.timestamp.desc())
        .first()
    )
    if not ev:
        ev = Evaluation(
            category=category,
            symptoms=json.dumps({}),
            status="draft",
            timestamp=datetime.utcnow()
        )
        db.add(ev)
        db.commit()
        db.refresh(ev)
    return ev


def build_bn_for_evaluation_no_evidence(evaluation: Evaluation):
    conn = get_conn()
    db = SessionLocal()
    try:
        category = evaluation.category
        target_node_lc = _lc(category)

        st.subheader(f"Bayesian Network for: {category}")
        with st.spinner("Building Bayesian Network…"):
            triples = conn.get_causal_triples(category)
            if not triples:
                st.warning(f"No triples are found for given '{category}'.")
                return

            triples_lc = lc_triples(triples)
            builder = BayesianNetworkBuilder()

            saved = _get_saved_llm_json(evaluation)
            if saved:
                try:
                    response = json.loads(saved)
                    st.success("Using saved model structure....")
                except Exception:
                    st.info("Regenerating model structure.")
                    llm_raw = generate_bn_structure_and_probabilities_from_llm(target_node_lc, triples_lc, {})
                    response = builder.clean_response(llm_raw)
                    _set_saved_llm_json(evaluation, response)
                    evaluation.status = "llm"
                    db.add(evaluation)
                    db.commit()
            else:
                st.info("Generating new model structure...")
                llm_raw = generate_bn_structure_and_probabilities_from_llm(target_node_lc, triples_lc, {})
                response = builder.clean_response(llm_raw)
                _set_saved_llm_json(evaluation, response)
                evaluation.status = "llm"
                db.add(evaluation)
                db.commit()

            builder = builder.from_llm_response(response)
            builder.build_structure()

            # Store CPDs
            st.session_state["cpds"] = response.get("cpds", [])

            evaluation.target_node = category
            db.add(evaluation)
            db.commit()

            # visualization of the BN
            fig = builder.visualize(target_node=category)
            st.pyplot(fig)
            st.success("Bayesian network created and evaluation updated.")

            # Handle CPT cases
            st.markdown("## Edit Conditional Probability Tables")
            if not st.session_state["cpds"]:
                st.info("No CPDs found to edit.")
            else:
                updated_cpd, run_update = display_cpds_dropdown(st.session_state["cpds"], key_prefix="")

                if run_update:
                    try:
                        full_json = json.loads(_get_saved_llm_json(evaluation))
                        full_json_updated = update_cpd_in_full_json(full_json, updated_cpd)

                        # Persist updated BN JSON
                        _set_saved_llm_json(evaluation, full_json_updated)
                        db.add(evaluation)
                        db.commit()

                        # Rebuild with updated CPTs
                        builder_updated = BayesianNetworkBuilder()
                        builder_updated = builder_updated.from_llm_response(full_json_updated)
                        builder_updated.build_structure()

                        st.success("CPT updated and saved. Rebuilding graph…")

                        # Draw refreshed graph
                        fig2 = builder_updated.visualize(category)
                        st.pyplot(fig2)

                        # Refresh session CPDs
                        st.session_state["cpds"] = full_json_updated.get("cpds", [])

                    except Exception as e:
                        st.error(f"Error updating CPT and rebuilding graph: {e}")

    except Exception as e:
        st.error(f"Failed to build BN: {e}")
        if st.checkbox("Show technical details", key="bn_build_error"):
            st.exception(e)
    finally:
        conn.close()
        db.close()


# "Display categories/condition that we can create Bayesian Network
def _render_category_actions():
    st.subheader("Categories")
    try:
        conn = get_conn()
        categories = conn.get_categories()
        conn.close()

        if not categories:
            st.info("No categories/condition found yet. Add URL and the name above to create one.")
            return
        # Simple layout showing categories with only create buttons
        for cat in categories:
            left, right = st.columns([4, 2])
            left.write(f"**{cat}**")
            if right.button("Create Bayesian Network", key=f"cbd_{cat}"):
                db = SessionLocal()
                try:
                    evaluation = create_or_get_evaluation_for_category(db, cat)
                    st.session_state["selected_evaluation"] = evaluation.id
                finally:
                    db.close()
                build_bn_for_evaluation_no_evidence(evaluation)

    except Exception as e:
        st.error(f"Failed to load categories... {e}")


import re, hashlib
import pandas as pd
import streamlit as st


def _uniq_key(prefix: str, *parts) -> str:
    txt = "|".join(str(p) for p in parts if p is not None)
    h = hashlib.md5(txt.encode("utf-8")).hexdigest()[:8]
    head = re.sub(r"[^0-9a-zA-Z_]+", "_", f"{prefix}_{parts[0] if parts else ''}")[:60]
    return f"{head}__{h}"


def display_cpds_dropdown(
        cpds,
        button_label: str = "🔁 Update and Recalculate Inference",
        key_prefix: str = "cpd_panel"
):
    # helper that always appends a unique hash
    def K(*parts):
        return _uniq_key(key_prefix, *parts)

    st.title("Explore and edit Conditional Probability Tables (CPTs)")

    node_names = [cpd["node"] for cpd in cpds]
    selected_node = st.selectbox(
        "Select a node to edit its CPT",
        node_names,
        key=K("node_selector")
    )

    cpd = next(c for c in cpds if c["node"] == selected_node)
    given = cpd.get("given", []) or []
    st.markdown(f"# Node: `{selected_node}`")
    st.markdown(f"**Given (parents):** {', '.join(given) if given else 'None (prior)'}")

    # stable namespace for the selected node
    ns = (selected_node, tuple(given), "cpd")

    # --- Handle noisy OR cases ---
    if _is_noisy_or(cpd):
        weights = cpd.get("noisy_or_weights", {}) or {}
        for p in given:
            weights.setdefault(p, 0.1)
        leak = float(cpd.get("leak", 0.0))

        cols = st.columns(2)
        with cols[0]:
            leak = st.number_input(
                "Leak (baseline P(child=present) with no parents present)",
                min_value=0.0, max_value=1.0, step=0.001, value=leak,
                key=K("leak", ns)
            )

        st.markdown("#### Parent Weights")
        for i, parent in enumerate(given):
            w_default = float(weights.get(parent, 0.1))
            w = st.slider(
                f"{parent}",
                min_value=0.0, max_value=1.0, step=0.01,
                value=w_default,
                key=K("w_slider", ns, i, parent)
            )
            weights[parent] = float(w)

        cpd["noisy_or_weights"] = weights
        cpd["leak"] = float(leak)

        run_update = st.button(button_label, key=K("recalc_noisyor", ns))
        return cpd, run_update

    # --- Handle tabular cases  ---
    probs = cpd["probabilities"]
    condition_order = cpd.get("condition_order")
    if isinstance(probs.get("present"), list) and not condition_order:
        condition_order = [{} for _ in range(len(probs["present"]))]

    updated_present, updated_absent = [], []

    if isinstance(probs.get("present"), list):
        for i, condition in enumerate(condition_order):
            with st.expander(
                    f"Condition {i + 1}: {condition}",
                    expanded=False,
                    key=K("expander", ns, i)
            ):
                col1, col2 = st.columns(2)
                with col1:
                    default_present = float(probs["present"][i]) if i < len(probs["present"]) else 0.5
                    present = st.number_input(
                        f"P(Present) | {condition}",
                        min_value=0.0, max_value=1.0, step=0.01,
                        value=default_present,
                        key=K("present_input", ns, i, str(condition))
                    )
                absent = round(1.0 - present, 6)
                col2.markdown(f"**P(Absent) | {condition}:** `{absent}`")
                updated_present.append(present)
                updated_absent.append(absent)
    else:
        # prior row
        st.markdown("### Edit Prior Probabilities")
        col1, col2 = st.columns(2)
        with col1:
            present = st.number_input(
                "P(Present)",
                min_value=0.0, max_value=1.0, step=0.01,
                value=float(probs.get("present", 0.5)),
                key=K("prior_present_input", ns)
            )
        absent = round(1.0 - present, 6)
        col2.markdown(f"**P(Absent):** `{absent}`")
        updated_present, updated_absent = present, absent

    # update values
    cpd["probabilities"]["present"] = updated_present
    cpd["probabilities"]["absent"] = updated_absent

    # preview new values
    st.markdown("### Updated CPT Preview")
    if isinstance(updated_present, list):
        rows = []
        for i, condition in enumerate(condition_order):
            row = {**condition}
            row["P(Present)"] = round(updated_present[i], 6)
            row["P(Absent)"] = round(updated_absent[i], 6)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.dataframe(pd.DataFrame([{
            "P(Present)": round(updated_present, 6),
            "P(Absent)": round(updated_absent, 6)
        }]), use_container_width=True)

    run_update = st.button(button_label, key=K("recalc_tabular", ns))
    return cpd, run_update


def update_cpd_in_full_json(full_json, updated_cpd):
    updated_probabilities_json = dict(full_json)
    updated_probabilities_json["cpds"] = list(full_json.get("cpds", []))  # shallow copy list

    found = False
    for i, cpd in enumerate(updated_probabilities_json["cpds"]):
        if cpd.get("node") == updated_cpd.get("node"):
            updated_probabilities_json["cpds"][i] = updated_cpd
            found = True
            break

    if not found:
        updated_probabilities_json["cpds"].append(updated_cpd)

    return updated_probabilities_json


def admin_view():
    st.header("Admin Panel")
    st.caption("Paste content, extract to Neo4j, then create or edit Bayesian networks per category.")

    sel_ev = st.session_state.get("selected_evaluation")
    if sel_ev:
        with st.expander("Selected Evaluation", expanded=True):
            if st.button("Clear selection"):
                st.session_state.pop("selected_evaluation", None)
            else:
                render_evaluation_panel(sel_ev)

    _render_category_actions()

    st.markdown("---")
    st.subheader("Add the link you want to extract medical information")
    colA, colB = st.columns([3, 2])
    with colA:
        category = st.text_input("Category, Condition you are extracting", max_chars=100)
    with colB:
        st.caption("")

    with st.form("extract_upload_form"):
        input_text = st.text_area("Paste raw text or a public URL")
        text = ""
        if input_text.startswith("http"):
            try:
                r = scrape_page(input_text)
                text = r.text if hasattr(r, "text") else r
            except Exception as e:
                st.error(f"URL fetch failed: {e}")
        else:
            text = input_text

        submit = st.form_submit_button("Extract and build Knowledge Graph")
    if submit and text and category:
        user = db.query(User).filter_by(id=st.session_state.user_id).first()
        eval = get_or_create_evaluation(user.id, category);
        st.text_area("Preview of Extracted Text", text[:1500], height=240)
        with st.spinner("Extracting triples and uploading them.."):
            triples = extract_triples(text)
            if category:
                conn = get_conn()
                try:
                    for subj, rel, obj in triples:
                        conn.insert_triple(
                            subj,
                            rel,
                            obj)

                    conn.attach_centers_for_page(
                        triples,
                        category=category,
                        attach_all_nodes=False
                    )
                    st.success(f"Uploaded {len(triples)} triples to Neo4j!")
                    with st.expander("Display uploaded triples"):
                        for t in triples:
                            st.write(f"- {t}")
                finally:
                    conn.close()
            else:
                st.warning("No triples extracted.")
    elif submit and not category:
        st.error("Please enter a category before uploading.")


def render_evaluation_panel(evaluation_id: int):
    try:
        evaluation = db.query(Evaluation).get(evaluation_id)
        if not evaluation:
            st.error("Evaluation not found.");
            return

        st.markdown(f"### Evaluation for **{evaluation.category}**")
        st.caption(f"ID: {evaluation.id} • Status: {evaluation.status} • Timestamp: {evaluation.timestamp}")

        conn = get_conn()
        try:
            triples = conn.get_causal_triples(evaluation.category)
        finally:
            conn.close()

        if not triples:
            st.warning("No triples found for this category.");
            return

        target_node_lc = _lc(evaluation.category)

        # 1) Load/generate BN JSON without evidence and persist if needed
        response = load_or_generate_bn_json(db, evaluation, triples, target_node_lc)

        # 2) Build + draw
        builder = BayesianNetworkBuilder().from_llm_response(response)
        builder.build_structure()
        st.markdown("#### Bayesian Network")
        fig = builder.visualize(target_node=evaluation.category)
        st.pyplot(fig)

        # 3) CPT editor
        st.session_state["cpds"] = response.get("cpds", [])
        if not st.session_state["cpds"]:
            st.info("No CPDs available to edit.")
        else:
            updated_cpd, run_update = display_cpds_dropdown(st.session_state["cpds"])
            if run_update:
                try:
                    full_json = json.loads(_get_saved_llm_json(evaluation))
                    full_json_updated = update_cpd_in_full_json(full_json, updated_cpd)
                    _set_saved_llm_json(evaluation, full_json_updated)
                    db.add(evaluation)
                    db.commit()

                    builder = BayesianNetworkBuilder().from_llm_response(full_json_updated)
                    builder.build_structure()
                    st.success("CPT saved. Rebuilding graph…")
                    fig2 = builder.visualize(target_node=evaluation.category)
                    st.pyplot(fig2)

                    st.session_state["cpds"] = full_json_updated.get("cpds", [])
                except Exception as e:
                    st.error(f"Failed to save CPT & rebuild: {e}")

    finally:
        db.close()


# ---------- helpers methods ----------

def clean_category(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r"\s*-\s*NHS.*$", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\b(overview|symptoms|causes|treatment|treatments|prevention|diagnosis)\b", "", raw,
                 flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw.title()


def infer_category_from_url(url: str) -> str:
    try:
        path = re.sub(r"https?://", "", url).split("/", 1)[-1]  # strip domain
        segments = [s for s in path.split("/") if s]
        if not segments:
            return ""
        # Prefer the first segment after 'conditions' if present; else last non-generic segment
        if segments[0].lower() == "conditions" and len(segments) > 1:
            cand = segments[1]
        else:
            # skip generic trailing segments like 'symptoms', 'treatment'
            generic = {"overview", "symptoms", "causes", "treatment", "treatments", "prevention", "diagnosis"}
            cand = next((s for s in segments if s.lower() not in generic), segments[-1])
        cand = cand.replace("-", " ").replace("_", " ")
        return clean_category(cand)
    except Exception:
        return ""


def extract_title_from_html(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        t = soup.title.string if soup.title and soup.title.string else ""
        return clean_category(t)
    except Exception:
        return ""


def infer_category(category_input: str, url: str | None, html: str | None) -> str:
    # priority: explicit input > <title> > url segment
    if category_input and category_input.strip():
        return clean_category(category_input.strip())
    title_guess = extract_title_from_html(html or "") if html else ""
    if title_guess:
        return title_guess
    if url and url.strip().startswith(("http://", "https://")):
        guess = infer_category_from_url(url.strip())
        if guess:
            return guess
    return ""


@contextmanager
def _db() -> SessionLocal:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# If there is a selected evaluation, update its category otherwise create a new one.
def get_or_create_evaluation(user_id: int, category: str, selected_eval=None):
    with _db() as db:
        if selected_eval:
            ev = db.query(Evaluation).get(selected_eval.id)
            if ev:
                ev.category = category
                ev.status = getattr(Evaluation, "status", "pending") or "pending"
                return ev.id
        # else create a new one evaluation
        ev = Evaluation(
            user_id=user_id,
            category=category,
            status="pending",
        )
        db.add(ev)
        db.flush()
        return ev.id


def load_or_generate_bn_json(db, evaluation, triples, target_node_lc):
    try:
        saved = _get_saved_llm_json(evaluation)
        if saved:
            return json.loads(saved)
    except Exception:
        pass

    # generate with NO evidence
    llm_response = generate_bn_structure_and_probabilities_from_llm(
        target_node_lc,
        lc_triples(triples),
        {}
    )
    response = BayesianNetworkBuilder().clean_response(llm_response)

    _set_saved_llm_json(evaluation, response)
    evaluation.status = "llm"
    db.add(evaluation)
    db.commit()
    return response


def render_evaluation_panel(evaluation_id: int):
    db = SessionLocal()
    try:
        evaluation = db.query(Evaluation).get(evaluation_id)
        if not evaluation:
            st.error("Evaluation not found...");
            return

        st.markdown(f"### Evaluation for **{evaluation.category}**")
        st.caption(f"ID: {evaluation.id} • Status: {evaluation.status} • Timestamp: {evaluation.timestamp}")

        conn = get_conn()
        try:
            triples = conn.get_causal_triples(evaluation.category)
        finally:
            conn.close()

        if not triples:
            st.warning("No triples found for this category.");
            return

        target_node_lc = _lc(evaluation.category)  # turn lowercase category

        response = load_or_generate_bn_json(db, evaluation, triples, target_node_lc)

        # 2) Build and display
        builder = BayesianNetworkBuilder().from_llm_response(response)
        builder.build_structure()
        st.markdown("#### Bayesian Network")
        fig = builder.visualize(target_node=evaluation.category)
        st.pyplot(fig)

        # 3) CPT editor
        st.session_state["cpds"] = response.get("cpds", [])
        if not st.session_state["cpds"]:
            st.info("No CPDs available to edit.")
        else:
            updated_cpd, run_update = display_cpds_dropdown(st.session_state["cpds"])
            if run_update:
                try:
                    full_json = json.loads(_get_saved_llm_json(evaluation))
                    full_json_updated = update_cpd_in_full_json(full_json, updated_cpd)
                    _set_saved_llm_json(evaluation, full_json_updated)
                    db.add(evaluation)
                    db.commit()

                    builder = BayesianNetworkBuilder().from_llm_response(full_json_updated)
                    builder.build_structure()
                    st.success("CPT saved. Rebuilding graph…")
                    fig2 = builder.visualize(target_node=evaluation.category)
                    st.pyplot(fig2)

                    st.session_state["cpds"] = full_json_updated.get("cpds", [])
                except Exception as e:
                    st.error(f"Failed to save CPT and rebuild: {e}")

    finally:
        db.close()
