"""
This module defines the `BayesianNetworkBuilder` class and helper functions for
building, validating, and running inference on Bayesian Networks from
clinical knowledge graphs or LLM generated JSON responses.

Key features:
1. Node and text handling:
   - Canonicalizes and normalizes node names for consistent graph building.
   - Maps aliases and raw text to display-safe node labels.

2. Parsing and cleaning:
   - clean_response(response): Safely parses and cleans JSON-like responses from LLMs.
   - Handles formatting quirks (e.g., code fences, trailing commas).

3. Model construction:
   - from_llm_response(llm_output): Builds a Bayesian Network structure directly from JSON containing
     nodes, edges, and CPDs (including both tabular and Noisy-OR CPDs).
   - build_structure_normal(triples): Builds Bayesian Network structure directly from causal triples.
   - Automatically adds priors for nodes without CPDs.

4. Probability calculations:
   - build_tabular_cpd_from_definition_full(): Converts structured probability
     definitions into `pgmpy` TabularCPD objects.
   - generate_noisy_or_cpd(): Constructs Noisy-OR CPDs from parent weights.
   - self_sync_model_parents_with_cpd(): Aligns Bayesian Network graph edges with CPD parents.

5. Evidence and inference:
   - filter_evidence_to_available_nodes(): Filters evidence to valid model nodes.
   - enforce_absent_present(): Enforces that evidence values to be only 'present' or 'absent'.
   - infer_with_evidence_filtering(): Runs probabilistic inference with audited evidence
     using variable elimination, returning posterior probabilities and audit info.

6. Visualization:
   - visualize(): Plots the Bayesian Network using `networkx` + `matplotlib` with
     selectable layouts (spring, shell, circular, kamada).
   - safe_figsize(): Dynamically adjusts figure size based on node count.

Purpose: Provides a pipeline to transform extracted triples or LLM outputs
         into a functional Bayesian Network that supports inference,
         visualization, and integration into clinical decision support systems.
"""

import json
import math
import re
import unicodedata
from typing import Dict, List, Tuple, Set, Optional, Any
from itertools import product

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class BayesianNetworkBuilder:
    def __init__(self, causal_relations=None):
        self.model = BayesianNetwork()
        self.causal_relations = causal_relations or {"CAUSES", "CAN_LEAD_TO", "INCLUDES"}
        self.node_aliases: Dict[str, str] = {}
        self._canon: Dict[str, str] = {}

    def canon_key(self, s: str) -> str:

        if s is None:
            return ""
        s = unicodedata.normalize("NFKC", str(s))
        s = s.replace("’", "'").replace("‘", "'")
        s = s.replace("–", "-").replace("—", "-")
        s = s.replace("\u00A0", " ")
        s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s.casefold()

    def to_display(self, raw: str) -> str:
        key = self.canon_key(raw)
        if key in self._canon:
            return self._canon[key]
        display = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(raw))).strip()
        self._canon[key] = display
        return display

    def normalize_node_name(self, name: str) -> str:
        s = unicodedata.normalize("NFKC", str(name)).casefold()
        s = s.replace("\u00A0", " ")
        s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s

    def name_map(self) -> Dict[str, str]:
        m = {self.normalize_node_name(n): n for n in self.get_available_nodes()}
        for alias, real in (self.node_aliases or {}).items():
            m[self.normalize_node_name(alias)] = real
        return m

    def clean_response(self, response: Any) -> dict:
        if isinstance(response, dict):
            return response
        txt = (response or "").strip()
        if not txt:
            raise ValueError("Empty or invalid LLM response.")
        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s*```$", "", txt)
        txt = re.sub(r",\s*([\]}])", r"\1", txt)
        try:
            return json.loads(txt)
        except json.JSONDecodeError as e:
            preview = txt[max(e.pos - 50, 0): e.pos + 50]
            raise ValueError(f"Invalid JSON from LLM: {e}\nContext: …{preview}…")

    def get_available_nodes(self) -> List[str]:
        return list(self.model.nodes())

    def get_symptoms(
            self,
            category: str,
            symptom_rels: Optional[Set[str]] = None,
    ) -> List[str]:
        if not category or not category.strip():
            return []

        # Default relation whitelist (extend to match your dataset)
        if symptom_rels is None:
            symptom_rels = {
                "HAS_SYMPTOM", "SYMPTOMS_INCLUDE", "INCLUDES", "INCLUDE",
                "HAS_SIGN", "PRESENTS_WITH"
            }

        rels = sorted({self.sanitize_rel(r) for r in symptom_rels})
        cat = category.strip()

        cypher = """
        // Normalize seed and pass relation whitelist
        WITH toLower($cat) AS seed, $rels AS rels

        // --- A) Inside the Category hub (target scope) ---
        MATCH (hub:Category) WHERE toLower(hub.name) = seed

        // A1: category member ->(symptom rel)-> symptom
        MATCH (hub)<-[:PART_OF]-(a:Entity)-[r]->(s:Entity)
        WHERE type(r) IN rels
        WITH seed, rels, collect(DISTINCT s.name) AS A1

        // A2: symptom ->(symptom rel)-> category member
        OPTIONAL MATCH (hub)<-[:PART_OF]-(s2:Entity)-[r2]->(a2:Entity)
        WHERE type(r2) IN rels
        WITH seed, rels, A1 + collect(DISTINCT s2.name) AS A

        // --- B) Heading-style nodes (e.g., 'Symptoms of <category>') ---
        WITH seed, rels, A
        OPTIONAL MATCH (h:Entity)
        WHERE toLower(h.name) IN ['symptoms of ' + seed, 'signs of ' + seed]
        OPTIONAL MATCH (h)-[rh]->(s3:Entity)
        WHERE type(rh) IN rels
        WITH seed, rels, A + collect(DISTINCT s3.name) AS AB

        // --- C) Fallback: direct neighbors of an Entity named exactly like the category ---
        OPTIONAL MATCH (e:Entity) WHERE toLower(e.name) = seed
        OPTIONAL MATCH (e)-[re]->(s4:Entity)
        WHERE type(re) IN rels
        WITH AB + collect(DISTINCT s4.name) AS ABC

        // Dedupe and return
        UNWIND ABC AS nm
        WITH DISTINCT nm WHERE nm IS NOT NULL AND trim(nm) <> ''
        RETURN nm AS symptom
        ORDER BY toLower(symptom)
        """

        with self.session() as s:
            rows = s.run(cypher, {"cat": cat, "rels": rels}).data()

        return [r["symptom"] for r in rows]

    def get_available_evidence_nodes(self) -> List[str]:
        return [cpd.variable for cpd in (self.model.get_cpds() or [])]

    # ---------- builder ----------

    @classmethod
    def from_llm_response(cls, llm_output: dict):
        builder = cls()

        nodes_in = llm_output.get("nodes", []) or []
        edges_in = llm_output.get("edges", []) or []
        cpds_raw = llm_output.get("cpds", []) or []
        cpds_list = cpds_raw if isinstance(cpds_raw, list) else [cpds_raw]

        # 1) build nodes
        for n in nodes_in:
            d = builder.to_display(n)
            if not builder.model.has_node(d):
                builder.model.add_node(d)

        # 2) build edges
        for e in edges_in:
            if not isinstance(e, dict) or "from" not in e or "to" not in e:
                continue
            u = builder.to_display(e["from"])
            v = builder.to_display(e["to"])
            if not builder.model.has_node(u): builder.model.add_node(u)
            if not builder.model.has_node(v): builder.model.add_node(v)
            if not builder.model.has_edge(u, v):
                builder.model.add_edge(u, v)

        # 3) Ensure CPD-implied parents exist
        for cpd_def in cpds_list:
            if not isinstance(cpd_def, dict): continue
            child = builder.to_display(cpd_def.get("node"))
            given = [builder.to_display(p) for p in (cpd_def.get("given", []) or [])]
            if not child: continue
            builder.model.add_node(child)
            for p in given:
                builder.model.add_node(p)
                if not builder.model.has_edge(p, child):
                    builder.model.add_edge(p, child)

        # 4) Build CPDs
        for cpd_def in cpds_list:
            if not isinstance(cpd_def, dict): continue
            try:
                node = builder.to_display(cpd_def.get("node"))
                if not node: continue
                given = [builder.to_display(p) for p in (cpd_def.get("given", []) or [])]

                if "noisy_or_weights" in cpd_def:
                    raw_w = cpd_def.get("noisy_or_weights", {}) or {}
                    cleaned_w = {builder.to_display(k): float(v) for k, v in raw_w.items()}
                    weights = {p: float(cleaned_w.get(p, 0.1)) for p in given}
                    leak = float(cpd_def.get("noisy_or_leak", 0.03))
                    cpd = generate_noisy_or_cpd(node=node, parents=given, weights=weights, base_prob=leak)
                else:
                    cpd_def2 = {"node": node, "given": given, "probabilities": cpd_def.get("probabilities", {}) or {}}
                    cpd = build_tabular_cpd_from_definition_full(cpd_def2)

                self_sync_model_parents_with_cpd(builder.model, cpd)
                builder.model.add_cpds(cpd)

            except Exception as e:
                st.warning(f"Error building CPD for {cpd_def.get('node', '?')}: {e}")

        # 5) Priors without CPD
        all_nodes = set(builder.model.nodes())
        with_cpds = {cpd.variable for cpd in (builder.model.get_cpds() or [])}
        for n in sorted(all_nodes - with_cpds):
            if not builder.model.get_parents(n):
                try:
                    prior = TabularCPD(
                        variable=n, variable_card=2,
                        values=[[0.9], [0.1]],
                        state_names={n: ["absent", "present"]}
                    )
                    builder.model.add_cpds(prior)
                except Exception as e:
                    st.warning(f"Could not add prior for '{n}': {e}")

        # 6) validate the model
        builder.model.check_model()
        if len(builder.model.nodes()) == 0:
            raise RuntimeError("BN has zero nodes after build — check the JSON keys ('nodes', 'cpds', ...).")
        return builder

    def filter_evidence_to_available_nodes(self, evidence: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        name_map = self.name_map()
        filtered, ignored = {}, []
        for k, v in (evidence or {}).items():
            nk = self.normalize_node_name(k)
            real = name_map.get(nk)
            if real is None:
                ignored.append(k)
                continue
            vv = (v.strip().lower() if isinstance(v, str) else v)
            if real in filtered:
                pv = (filtered[real].strip().lower() if isinstance(filtered[real], str) else filtered[real])
                filtered[real] = "present" if (vv == "present" or pv == "present") else "absent"
            else:
                filtered[real] = vv
        return filtered, ignored

    def _map_01_to_absent_present(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (evidence or {}).items():
            if isinstance(v, int):
                out[k] = "present" if v == 1 else ("absent" if v == 0 else v)
            elif isinstance(v, str):
                s = v.strip()
                out[k] = "present" if s == "1" else ("absent" if s == "0" else v)
            else:
                out[k] = v
        return out

    def build_structure(self):
        print("nothing")

    def build_structure_normal(
            self,
            triples: List[Tuple[str, str, str]],
            allowed_relations: Optional[set[str]] = None,
            avoid_cycles: bool = True,
    ) -> "BayesianNetworkBuilder":
        rel_whitelist = (
            {str(r or "").strip().upper().replace(" ", "_") for r in allowed_relations}
            if allowed_relations else None
        )

        for idx, triple in enumerate(triples):
            try:
                if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                    continue
                s_raw, rel_raw, t_raw = triple[0], triple[1], triple[2]

                s = self._as_str(s_raw)
                t = self._as_str(t_raw)
                rel_norm = self._as_str(rel_raw).upper().replace(" ", "_")

                if not s or not t:
                    continue
                if rel_whitelist is not None and rel_norm not in rel_whitelist:
                    continue

                if not self.model.has_node(s):
                    self.model.add_node(s)
                if not self.model.has_node(t):
                    self.model.add_node(t)
                if avoid_cycles:
                    if not self.model.has_edge(s, t):
                        self.model.add_edge(s, t)
                        import networkx as nx
                        if not nx.is_directed_acyclic_graph(nx.DiGraph(self.model.edges())):
                            self.model.remove_edge(s, t)
                else:
                    if not self.model.has_edge(s, t):
                        self.model.add_edge(s, t)

            except Exception as e:
                try:
                    import streamlit as st
                    st.warning(f"skipping triple #{idx} due to error: {e} | triple={triple}")
                except Exception:
                    print(f"[WARN] skipping triple #{idx} due to error: {e} | triple={triple}")

        return self

    # Strict values: ONLY 'present' or 'absent' (case-insensitive).
    def enforce_absent_present(self, evidence: Dict[str, Any]) -> Dict[str, str]:
        cleaned, bad = {}, {}
        for k, v in (evidence or {}).items():
            if isinstance(v, str) and v.strip().lower() in {"present", "absent"}:
                cleaned[k] = v.strip().lower()
            else:
                bad[k] = v
        if bad:
            raise ValueError(
                "Evidence values must be the strings 'present' or 'absent' only. "
                f"Offending entries: {bad}"
            )
        return cleaned

    def infer_with_evidence_filtering(
            self,
            query_variable: str,
            evidence: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if not self.model.check_model():
            raise ValueError("Model is invalid. Check CPDs and structure.")

        ev = self._map_01_to_absent_present(evidence or {})

        mapped, ignored_input = self.filter_evidence_to_available_nodes(ev)

        strict_evidence = self.enforce_absent_present(mapped)

        used, ignored_by_audit = {}, []
        for var, val in strict_evidence.items():
            cpd = self.model.get_cpds(var)
            if not cpd:
                ignored_by_audit.append(var);
                continue
            states = list(cpd.state_names.get(var, []))
            state_map = {s.lower(): s for s in states}
            vv = val.lower()
            if vv in state_map:
                used[var] = state_map[vv]
            else:
                ignored_by_audit.append(var)

        name_map = self.name_map()
        qk = self.normalize_node_name(query_variable)
        if qk not in name_map:
            raise ValueError(f"Query node not found: '{query_variable}' (normalized '{qk}').")
        query_node = name_map[qk]

        used.pop(query_node)
        ve = VariableElimination(self.model)
        try:
            posterior = ve.query(variables=[query_node], evidence=used or None, show_progress=False)
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

        if hasattr(posterior, "values") and hasattr(posterior, "state_names"):
            states = posterior.state_names.get(query_node, ["absent", "present"])
            vals = posterior.values.flatten()
            probs = {states[i]: float(vals[i]) for i in range(len(states))}
        else:
            probs = {}

        info = {
            "query_node": query_node,
            "normalized_query": qk,
            "used_evidence": used,
            "ignored_evidence_input": ignored_input,
            "ignored_by_audit": ignored_by_audit,
            "available_nodes": self.get_available_nodes(),
            "strict_mode": True,
            "converted_01": True,
        }
        return probs, info

    def visualize(self, figsize=(10, 6), layout="circular", target_node=""):
        if not self.model.nodes():
            raise ValueError("Model has no nodes to visualize.")
        graph = nx.DiGraph(self.model.edges())
        if not nx.is_directed_acyclic_graph(graph):
            st.warning("Graph contains cycles - visualization may not be optimal.")

        layout_func = {
            "spring": nx.spring_layout,
            "shell": nx.shell_layout,
            "circular": nx.circular_layout,
            "kamada": nx.kamada_kawai_layout,
        }.get(layout, nx.circular_layout)

        try:
            pos = layout_func(graph)
        except Exception:
            pos = nx.shell_layout(graph)

        fs = safe_figsize(figsize, len(list(self.model.nodes())))
        fig, ax = plt.subplots(figsize=fs)
        nx.draw(
            graph, pos, with_labels=True,
            node_color="lightcoral", node_size=1000,
            font_size=8, font_weight="bold", edge_color="gray", ax=ax
        )
        ax.set_title("Bayesian Network Structure for " + target_node)
        return fig

    def _as_str(self, x) -> str:
        if x is None:
            return ""
        if isinstance(x, (list, tuple)):
            return " ".join(map(str, x)).strip()
        return str(x).strip()


def build_tabular_cpd_from_definition_full(cpd_def: dict) -> TabularCPD:
    node = cpd_def["node"]
    given = cpd_def.get("given", []) or []
    probs = cpd_def.get("probabilities", {}) or {}

    if not given:
        p_present = float(probs["present"])
        p_absent = float(probs["absent"])
        return TabularCPD(
            variable=node, variable_card=2,
            values=[[round(p_absent, 4)], [round(p_present, 4)]],
            state_names={node: ["absent", "present"]}
        )

    parent_states = ["present", "absent"]
    combos = list(product(parent_states, repeat=len(given)))
    present, absent = [], []
    for combo in combos:
        key = ", ".join(combo)
        k1 = f"present | {key}"
        k0 = f"absent | {key}"
        p1 = float(probs.get(k1, 0.0))
        p0 = float(probs.get(k0, 1.0 - p1))
        present.append(round(p1, 4))
        absent.append(round(p0, 4))

    return TabularCPD(
        variable=node, variable_card=2,
        values=[absent, present],
        evidence=given, evidence_card=[2] * len(given),
        state_names={node: ["absent", "present"], **{p: ["absent", "present"] for p in given}}
    )


def generate_noisy_or_cpd(node: str, parents: List[str], weights: Dict[str, float], base_prob: float = 0.01
                          ) -> TabularCPD:
    present_probs, absent_probs = [], []
    for combo in product([0, 1], repeat=len(parents)):
        active = [weights[p] for p, v in dict(zip(parents, combo)).items() if int(v) == 1]
        p_present = 1 - np.prod([1 - w for w in active]) if active else float(base_prob)
        p_present = min(max(p_present, 0.0), 1.0)
        p_absent = 1.0 - p_present
        present_probs.append(round(p_present, 4))
        absent_probs.append(round(p_absent, 4))

    values = [absent_probs, present_probs]
    state_names = {node: ["absent", "present"], **{p: ["absent", "present"] for p in parents}}
    return TabularCPD(
        variable=node, variable_card=2,
        values=values,
        evidence=parents, evidence_card=[2] * len(parents),
        state_names=state_names
    )


def self_sync_model_parents_with_cpd(model: BayesianNetwork, cpd: TabularCPD) -> None:
    node = cpd.variable
    cpd_parents = list(getattr(cpd, "evidence", None) or getattr(cpd, "variables", [])[1:])
    for p in cpd_parents:
        if not model.has_node(p):
            model.add_node(p)
        if not model.has_edge(p, node):
            model.add_edge(p, node)
    for p in list(model.get_parents(node)):
        if p not in cpd_parents:
            model.remove_edge(p, node)


def safe_figsize(figsize, n_nodes: int) -> tuple[float, float]:
    if n_nodes is None:
        n_nodes = 0
    w_default = min(20.0, max(6.0, 0.35 * max(1, n_nodes)))
    h_default = min(14.0, max(4.0, 0.28 * max(1, n_nodes)))
    default = (w_default, h_default)

    if figsize is None:
        return default

    try:
        w, h = figsize  # must be iterable of length 2
        w = float(w)
        h = float(h)
        if not (math.isfinite(w) and math.isfinite(h) and w > 0 and h > 0):
            return default
        return (w, h)
    except Exception:
        return default
