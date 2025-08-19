# import unicodedata, re
#
# import pgmpy
# import streamlit as st
# import matplotlib
# matplotlib.use("Agg")
#
# import matplotlib.pyplot as plt
# import networkx as nx
# from itertools import product
# from typing import Dict, List, Tuple, Optional, Any
#
# import json
# import re
# import numpy as np
# from pgmpy.models import BayesianNetwork
# from pgmpy.factors.discrete import TabularCPD
# from pgmpy.inference import VariableElimination
#
#
# class BayesianNetworkBuilder:
#     """Bayesian Network Builder with parent/CPD sync, leak support, and healthcheck."""
#
#     def __init__(self, causal_relations=None):
#         self.model = BayesianNetwork()
#         self.causal_relations = causal_relations or {"CAUSES", "CAN_LEAD_TO", "INCLUDES"}
#         self.node_aliases: Dict[str, str] = {}
#         self._canon: Dict[str, str] = {}
#
#     def _canon_key(self, s: str) -> str:
#         """Canonical form for matching (never shown to users)."""
#         if s is None:
#             return ""
#         s = unicodedata.normalize("NFKC", str(s))
#         # unify punctuation/whitespace
#         s = s.replace("’", "'").replace("‘", "'")
#         s = s.replace("–", "-").replace("—", "-")
#         s = s.replace("\u00A0", " ")  # NBSP -> space
#         s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)  # zero-width chars
#         s = re.sub(r"\s+", " ", s).strip()
#         return s.casefold()  # robust lowercase
#
#     def _to_display(self, raw: str) -> str:
#         """
#         Resolve a raw label to the single display label we use for the model.
#         First one we see wins; subsequent variants map to the same display string.
#         """
#         key = self._canon_key(raw)
#         if key in self._canon:
#             return self._canon[key]
#         # first time seeing this label: freeze display form as the cleaned original
#         display = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(raw))).strip()
#         self._canon[key] = display
#         return display
#
#     # ---------- Utilities ----------
#
#     def _normalize_node_name(self, name: str) -> str:
#         """Robust canonical key for matching names."""
#         s = unicodedata.normalize("NFKC", str(name)).casefold()  # unicode fold + lowercase
#         s = s.replace("\u00A0", " ")  # NBSP -> space
#         s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)  # zero-width chars
#         s = re.sub(r"\s+", " ", s).strip()  # collapse spaces
#         # remove all non-alphanumerics so 'Shortness-of breath' == 'shortness of breath'
#         s = re.sub(r"[^a-z0-9]+", "", s)
#         return s
#
#
#     def add_node_alias(self, alias: str, actual_node: str):
#         self.node_aliases[self._normalize_node_name(alias)] = actual_node
#
#     def clean_response(self, response: Any) -> dict:
#         """Clean and parse an LLM response into JSON dict."""
#         if isinstance(response, dict):
#             return response
#         if not response or not str(response).strip():
#             raise ValueError("Empty or invalid LLM response.")
#         txt = str(response).strip()
#         txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
#         txt = re.sub(r"\s*```$", "", txt)
#         txt = re.sub(r",\s*([\]}])", r"\1", txt)  # trailing commas
#         try:
#             return json.loads(txt)
#         except json.JSONDecodeError as e:
#             preview = txt[max(e.pos - 50, 0): e.pos + 50]
#             raise ValueError(f"Invalid JSON from LLM: {e}\nContext: …{preview}…")
#
#     # ---------- Model info ----------
#     def get_available_nodes(self) -> List[str]:
#         return list(self.model.nodes())
#
#     def get_available_evidence_nodes(self) -> List[str]:
#         return [cpd.variable for cpd in (self.model.get_cpds() or [])]
#
#     # ---------- Evidence filtering ----------
#     def filter_evidence_to_available_nodes(self, evidence: Dict[str, Any]):
#         # same combined map as above
#         name_map = {self._normalize_node_name(n): n for n in self.get_available_nodes()}
#         for a, real in (self.node_aliases or {}).items():
#             name_map[self._normalize_node_name(a)] = real
#
#         filtered, ignored = {}, []
#         for k, v in (evidence or {}).items():
#             nk = self._normalize_node_name(k)
#             real = name_map.get(nk)
#             if real is None:
#                 ignored.append(k)
#                 continue
#             # merge rule (if duplicates): any 'present' wins
#             if real in filtered and str(v).strip().lower() != "present":
#                 pass
#             else:
#                 filtered[real] = v
#         return filtered, ignored
#
#     @classmethod
#     def from_llm_response(cls, llm_output: dict):
#         builder = cls()
#
#         nodes_in = llm_output.get("nodes", []) or []
#         edges_in = llm_output.get("edges", []) or []
#         cpds_raw = llm_output.get("cpds", [])
#         cpds_list = cpds_raw if isinstance(cpds_raw, list) else ([cpds_raw] if isinstance(cpds_raw, dict) else [])
#
#         # 1) Nodes
#         for n in nodes_in:
#             d = builder._to_display(n)
#             if not builder.model.has_node(d):
#                 builder.model.add_node(d)
#
#         # 2) Edges
#         for e in edges_in or []:
#             if "from" not in e or "to" not in e:
#                 continue
#             u = builder._to_display(e["from"])
#             v = builder._to_display(e["to"])
#             if not builder.model.has_node(u): builder.model.add_node(u)
#             if not builder.model.has_node(v): builder.model.add_node(v)
#             if not builder.model.has_edge(u, v):
#                 builder.model.add_edge(u, v)
#
#         # 3) Ensure CPD-implied parents exist
#         for cpd_def in cpds_list:
#             child = builder._to_display(cpd_def.get("node"))
#             given = [builder._to_display(p) for p in (cpd_def.get("given", []) or [])]
#             if not child:
#                 continue
#             builder.model.add_node(child)
#             for p in given:
#                 builder.model.add_node(p)
#                 if not builder.model.has_edge(p, child):
#                     builder.model.add_edge(p, child)
#
#         # 4) Build CPDs (use display labels!)
#         for cpd_def in cpds_list:
#             try:
#                 node = builder._to_display(cpd_def.get("node"))
#                 if not node:
#                     continue
#                 given = [builder._to_display(p) for p in (cpd_def.get("given", []) or [])]
#
#                 if "noisy_or_weights" in cpd_def:
#                     raw_w = cpd_def.get("noisy_or_weights", {}) or {}
#                     cleaned_w = {builder._to_display(k): float(v) for k, v in raw_w.items()}
#                     weights = {p: float(cleaned_w.get(p, 0.1)) for p in given}
#                     leak = float(cpd_def.get("noisy_or_leak", 0.03))
#                     cpd = generate_noisy_or_cpd(node=node, parents=given, weights=weights, base_prob=leak)
#                 else:
#                     # also rewrite names inside the CPD definition so helper sees the same labels
#                     cpd_def2 = {
#                         "node": node,
#                         "given": given,
#                         "probabilities": cpd_def.get("probabilities", {})
#                     }
#                     cpd = build_tabular_cpd_from_definition_full(cpd_def2)
#
#                 self = builder  # shortcut for call
#                 builder._sync_model_parents_with_cpd(builder.model, cpd)
#                 builder.model.add_cpds(cpd)
#             except Exception as e:
#                 st.warning(f"Error building CPD for {cpd_def.get('node', '?')}: {e}")
#
#         # 5) Priors for true roots without CPD
#         all_nodes = set(builder.model.nodes())
#         with_cpds = {cpd.variable for cpd in (builder.model.get_cpds() or [])}
#         for n in sorted(all_nodes - with_cpds):
#             if not builder.model.get_parents(n):
#                 prior = TabularCPD(variable=n, variable_card=2,
#                                    values=[[0.9], [0.1]],
#                                    state_names={n: ["absent", "present"]})
#                 builder.model.add_cpds(prior)
#
#         # 6) Validate
#         builder.model.check_model()
#         return builder
#
#     # @classmethod
#     # def from_llm_response(cls, llm_output: dict):
#     #     """
#     #     Build BN from LLM JSON:
#     #     - Adds nodes/edges.
#     #     - Ensures edges implied by CPDs exist.
#     #     - Converts Noisy-OR (supports 'noisy_or_leak') to TabularCPD.
#     #     - Aligns graph parents to CPD parents.
#     #     - Adds priors only for true roots missing CPDs.
#     #     """
#     #     builder = cls()
#     #
#     #     nodes_in = llm_output.get("nodes", []) or []
#     #     edges_in = llm_output.get("edges", []) or []
#     #     cpds_raw = llm_output.get("cpds", [])
#     #     cpds_list = cpds_raw if isinstance(cpds_raw, list) else ([cpds_raw] if isinstance(cpds_raw, dict) else [])
#     #
#     #     # 1) Nodes
#     #     for n in nodes_in:
#     #         try:
#     #             builder.model.add_node(n)
#     #         except Exception as ex:
#     #             st.warning(f"Skipping node '{n}': {ex}")
#     #
#     #     # 2) Edges (best effort)
#     #     try:
#     #         edges = [(e["from"], e["to"]) for e in edges_in if "from" in e and "to" in e]
#     #         for u, v in edges:
#     #             if not builder.model.has_node(u): builder.model.add_node(u)
#     #             if not builder.model.has_node(v): builder.model.add_node(v)
#     #             builder.model.add_edge(u, v)
#     #     except Exception as ex:
#     #         st.warning(f"Edges block malformed, skipping explicit edges: {ex}")
#     #
#     #     # 3) Ensure CPD-implied parents exist
#     #     for cpd_def in cpds_list:
#     #         child = cpd_def.get("node")
#     #         given = [str(p).strip() for p in (cpd_def.get("given", []) or [])]
#     #         if not child:
#     #             continue
#     #         builder.model.add_node(child)
#     #         for p in given:
#     #             builder.model.add_node(p)
#     #             if not builder.model.has_edge(p, child):
#     #                 builder.model.add_edge(p, child)
#     #
#     #     # 4) Build CPDs, syncing parents per CPD
#     #     for cpd_def in cpds_list:
#     #         try:
#     #             node = cpd_def.get("node")
#     #             if not node:
#     #                 st.warning("Skipping CPD with missing 'node'")
#     #                 continue
#     #             given = [str(p).strip() for p in (cpd_def.get("given", []) or [])]
#     #
#     #             if "noisy_or_weights" in cpd_def:
#     #                 raw_w = cpd_def.get("noisy_or_weights", {}) or {}
#     #                 cleaned_w = {str(k).strip(): float(v) for k, v in raw_w.items()}
#     #                 weights = {p: float(cleaned_w.get(p, 0.1)) for p in given}
#     #                 leak = float(cpd_def.get("noisy_or_leak", 0.03))  # leak support
#     #                 cpd = generate_noisy_or_cpd(
#     #                     node=node,
#     #                     parents=given,
#     #                     weights=weights,
#     #                     base_prob=leak,
#     #                 )
#     #             else:
#     #                 cpd = build_tabular_cpd_from_definition_full(cpd_def)
#     #
#     #             if not isinstance(cpd, TabularCPD):
#     #                 st.warning(f"Skipping invalid CPD (not TabularCPD) for '{node}'")
#     #                 continue
#     #
#     #             # Align graph to CPD parents and add CPD
#     #             builder._sync_model_parents_with_cpd(builder.model, cpd)
#     #             try:
#     #                 builder.model.add_cpds(cpd)
#     #             except Exception as ex:
#     #                 # Retry after forced sync
#     #                 builder._sync_model_parents_with_cpd(builder.model, cpd)
#     #                 builder.model.add_cpds(cpd)
#     #
#     #         except Exception as e:
#     #             st.warning(f"Error building CPD for {cpd_def.get('node','?')}: {e}")
#     #
#     #     # 5) Add priors for true roots without CPD
#     #     all_nodes = set(builder.model.nodes())
#     #     cpd_nodes = {cpd.variable for cpd in (builder.model.get_cpds() or [])}
#     #     missing = all_nodes - cpd_nodes
#     #     for n in sorted(missing):
#     #         if not builder.model.get_parents(n):  # only true roots
#     #             try:
#     #                 prior = TabularCPD(
#     #                     variable=n, variable_card=2,
#     #                     values=[[0.9], [0.1]],
#     #                     state_names={n: ["absent", "present"]}
#     #                 )
#     #                 builder.model.add_cpds(prior)
#     #             except Exception as e:
#     #                 st.warning(f"Could not add prior for '{n}': {e}")
#     #
#     #     # 6) Validate
#     #     try:
#     #         builder.model.check_model()
#     #     except Exception as e:
#     #         st.warning(f"Model validation warning: {e}")
#     #
#     #     return builder
#
#     def _enforce_absent_present(self, evidence: Dict[str, Any]) -> Dict[str, str]:
#         """
#         Strict value check:
#           - Accept ONLY strings 'present' or 'absent' (case-insensitive).
#           - Do not coerce booleans/ints/yes/no/etc.
#           - Raise a clear error listing the offending entries.
#         """
#         cleaned: Dict[str, str] = {}
#         bad: Dict[str, Any] = {}
#         for k, v in (evidence or {}).items():
#             if isinstance(v, str) and v.strip().lower() in {"present", "absent"}:
#                 cleaned[k] = v.strip().lower()
#             else:
#                 bad[k] = v
#         if bad:
#             raise ValueError(
#                 "Evidence values must be the strings 'present' or 'absent' only. "
#                 f"Offending entries: {bad}"
#             )
#         return cleaned
#
#     def _map_01_to_absent_present(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Convert ONLY 0/1 (or "0"/"1") to 'absent'/'present'.
#         Everything else is left as-is (strict mode will enforce later).
#         """
#         out: Dict[str, Any] = {}
#         for k, v in (evidence or {}).items():
#             if isinstance(v, int):
#                 if v == 0:
#                     out[k] = "absent"
#                 elif v == 1:
#                     out[k] = "present"
#                 else:
#                     out[k] = v
#             elif isinstance(v, str):
#                 s = v.strip()
#                 if s == "0":
#                     out[k] = "absent"
#                 elif s == "1":
#                     out[k] = "present"
#                 else:
#                     out[k] = v
#             else:
#                 out[k] = v
#         return out
#
#     # ---------- Inference ----------
#     def infer_with_evidence_filtering(
#             self,
#             query_variable: str,
#             evidence: Dict[str, Any] | None = None
#     ) -> Tuple[Dict[str, float], Dict[str, Any]]:
#         """
#         Inference pipeline (robust names, strict values):
#           1) Convert only 0/1 -> 'absent'/'present'.
#           2) Map evidence keys to model nodes using the SAME normalizer as the model (+aliases).
#              - If multiple inputs map to the same node, 'present' wins.
#           3) Enforce evidence values are exactly 'present'/'absent' (case-insensitive).
#           4) Audit against CPDs: drop nodes without CPDs or with invalid state labels.
#           5) Resolve query node by the same normalizer and run VE.
#         Returns: (posterior_probabilities, info_dict)
#         """
#         if not self.model.check_model():
#             raise ValueError("Model is invalid. Check CPDs and structure.")
#
#         # ---------- small local helpers ----------
#         def map01(v: Any) -> Any:
#             if isinstance(v, int):
#                 return "present" if v == 1 else ("absent" if v == 0 else v)
#             if isinstance(v, str):
#                 sv = v.strip()
#                 if sv == "1": return "present"
#                 if sv == "0": return "absent"
#             return v
#
#         def enforce_absent_present(ev: Dict[str, Any]) -> Dict[str, str]:
#             cleaned, bad = {}, {}
#             for k, v in (ev or {}).items():
#                 if isinstance(v, str) and v.strip().lower() in {"present", "absent"}:
#                     cleaned[k] = v.strip().lower()
#                 else:
#                     bad[k] = v
#             if bad:
#                 raise ValueError(
#                     "Evidence values must be the strings 'present' or 'absent' only. "
#                     f"Offending entries: {bad}"
#                 )
#             return cleaned
#
#         # ---------- (1) 0/1 -> strings, no other coercions ----------
#         ev = {k: map01(v) for k, v in (evidence or {}).items()}
#
#         # ---------- (2) Build one name map (normalizer + aliases) ----------
#         name_map = {self._normalize_node_name(n): n for n in self.get_available_nodes()}
#         for alias, real in (self.node_aliases or {}).items():
#             name_map[self._normalize_node_name(alias)] = real
#
#         # Map & merge evidence keys to real nodes
#         filtered, ignored_input = {}, []
#         for raw_k, v in ev.items():
#             nk = self._normalize_node_name(raw_k)
#             real = name_map.get(nk)
#             if real is None:
#                 ignored_input.append(raw_k)
#                 continue
#             # merge rule: any 'present' dominates
#             vv = (v.strip().lower() if isinstance(v, str) else v)
#             if real in filtered:
#                 pv = (filtered[real].strip().lower() if isinstance(filtered[real], str) else filtered[real])
#                 filtered[real] = "present" if (vv == "present" or pv == "present") else "absent"
#             else:
#                 filtered[real] = vv
#
#         # ---------- (3) Strict values: must be 'present'/'absent' ----------
#         strict_evidence = enforce_absent_present(filtered)
#
#         # ---------- (4) Audit vs CPDs: drop nodes without CPDs / bad labels ----------
#         used, ignored_by_audit = {}, []
#         for var, val in strict_evidence.items():
#             cpd = self.model.get_cpds(var)
#             if not cpd:
#                 ignored_by_audit.append(var);
#                 continue
#             states = list(cpd.state_names.get(var, []))
#             state_map = {s.lower(): s for s in states}
#             vv = val.lower()
#             if vv in state_map:
#                 used[var] = state_map[vv]  # map to exact CPD label
#             else:
#                 ignored_by_audit.append(var)
#
#         # ---------- Resolve query node via same normalizer ----------
#         qk = self._normalize_node_name(query_variable)
#         if qk not in name_map:
#             sample = list(name_map.keys())[:10]
#             raise ValueError(
#                 f"Query node not found: '{query_variable}'. Normalized: '{qk}'. "
#                 f"Example normalized keys: {sample}"
#             )
#         query_node = name_map[qk]
#
#         # ---------- (5) Variable Elimination ----------
#         ve = pgmpy.inference.VariableElimination(self.model) if hasattr(pgmpy, "inference") else VariableElimination(
#             self.model)
#         try:
#             posterior = ve.query(variables=[query_node], evidence=used or None, show_progress=False)
#         except Exception as e:
#             raise RuntimeError(f"Inference failed: {e}")
#
#         # Extract probabilities
#         if hasattr(posterior, "values") and hasattr(posterior, "state_names"):
#             states = posterior.state_names.get(query_node, ["absent", "present"])
#             vals = posterior.values.flatten()
#             probabilities = {states[i]: float(vals[i]) for i in range(len(states))}
#         else:
#             probabilities = {}
#
#         info = {
#             "used_evidence": used,
#             "ignored_evidence_input": ignored_input,
#             "ignored_by_audit": ignored_by_audit,
#             "query_node": query_node,
#             "normalized_query": qk,
#             "available_nodes": list(self.get_available_nodes()),
#             "strict_mode": True,
#             "converted_01": True
#         }
#         return probabilities, info
#
#     def healthcheck(self, bn_json: Optional[dict] = None,
#                     evidence: Optional[dict] = None,
#                     target: Optional[str] = None) -> Dict[str, Any]:
#         """Light diagnostics: name mismatches, state orders, model validity, optional priors/posteriors."""
#         nodes = set(self.model.nodes())
#         diag = {
#             "missing_in_model": [],
#             "extra_in_model": [],
#             "unknown_evidence": [],
#             "bad_state_order": [],
#             "check_model_ok": None,
#         }
#
#         if bn_json:
#             json_nodes = set(bn_json.get("nodes", []))
#             diag["missing_in_model"] = sorted(json_nodes - nodes)
#             diag["extra_in_model"] = sorted(nodes - json_nodes)
#
#             # referenced names in edges/cpds that aren't in 'nodes'
#             used_names = set()
#             for e in bn_json.get("edges", []) or []:
#                 used_names.add(e.get("from")); used_names.add(e.get("to"))
#             for c in bn_json.get("cpds", []) or []:
#                 used_names.add(c.get("node"))
#                 for g in c.get("given", []) or []:
#                     used_names.add(g)
#             ref_not_listed = [x for x in used_names if x and x not in json_nodes]
#             if ref_not_listed:
#                 st.warning(f"References not in JSON 'nodes': {ref_not_listed}")
#
#         if evidence:
#             diag["unknown_evidence"] = sorted([k for k in evidence if k not in nodes])
#
#         for cpd in self.model.get_cpds() or []:
#             sn = (cpd.state_names or {}).get(cpd.variable)
#             if sn != ["absent", "present"]:
#                 diag["bad_state_order"].append({"node": cpd.variable, "state_names": sn})
#
#         try:
#             diag["check_model_ok"] = bool(self.model.check_model())
#         except Exception as e:
#             diag["check_model_ok"] = False
#             st.warning(f"check_model() raised: {e}")
#
#         if target and target in nodes:
#             ve = VariableElimination(self.model)
#             prior = ve.query([target])[target]
#             st.write(f"Prior P({target}) = {dict(zip(prior.state_names[target], map(float, prior.values)))}")
#             if evidence:
#                 used, _, _ = audit_evidence(self.model, evidence)
#                 post = ve.query([target], evidence=used)[target]
#                 st.write(f"Posterior P({target} | evidence) = {dict(zip(post.state_names[target], map(float, post.values)))}")
#
#         return diag
#
#     # ---------- Viz / misc ----------
#     def get_model_summary(self) -> Dict[str, Any]:
#         return {
#             "nodes": list(self.model.nodes()),
#             "edges": list(self.model.edges()),
#             "num_nodes": len(self.model.nodes()),
#             "num_edges": len(self.model.edges()),
#             "cpds_count": len(self.model.get_cpds() or []),
#             "is_valid": self.model.check_model() if self.model.nodes() else False,
#         }
#
#     def build_structure(self):
#         """No-op (structure is already maintained while adding CPDs/edges)."""
#         return
#
#     def validate_model(self) -> bool:
#         return bool(self.model.check_model())
#
#     def visualize(self, figsize=(10, 6), layout="circular"):
#         if not self.model.nodes():
#             raise ValueError("Model has no nodes to visualize.")
#         graph = nx.DiGraph(self.model.edges())
#         if not nx.is_directed_acyclic_graph(graph):
#             st.warning("Graph contains cycles - visualization may not be optimal.")
#
#         layout_func = {
#             "spring": nx.spring_layout,
#             "shell": nx.shell_layout,
#             "circular": nx.circular_layout,
#             "kamada": nx.kamada_kawai_layout,
#         }.get(layout, nx.circular_layout)
#
#         try:
#             pos = layout_func(graph)
#         except Exception:
#             pos = nx.shell_layout(graph)
#
#         fig, ax = plt.subplots(figsize=figsize)
#         nx.draw(
#             graph, pos, with_labels=True,
#             node_color="lightcoral", node_size=1000,
#             font_size=8, font_weight="bold", edge_color="gray", ax=ax
#         )
#         ax.set_title("Bayesian Network Structure")
#         return fig
#
#     # ---------- Parent/CPD sync ----------
#     @staticmethod
#     def _sync_model_parents_with_cpd(model, cpd: TabularCPD) -> None:
#         """
#         Ensure the graph's parents for cpd.variable exactly match CPD.evidence.
#         Adds missing edges, removes extra parents not listed in CPD.
#         """
#         node = cpd.variable
#         cpd_parents = list(getattr(cpd, "evidence", None) or getattr(cpd, "variables", [])[1:])
#
#         for p in cpd_parents:
#             if not model.has_node(p):
#                 model.add_node(p)
#             if not model.has_edge(p, node):
#                 model.add_edge(p, node)
#
#         for p in list(model.get_parents(node)):
#             if p not in cpd_parents:
#                 model.remove_edge(p, node)
#
#
# # ---------- Helpers (module-level) ----------
#
# def build_tabular_cpd_from_definition_full(cpd_def: dict) -> TabularCPD:
#     """
#     Build a TabularCPD from {"node": str, "given": [..], "probabilities": {...}}.
#     Assumes two states: absent/present. Row 0 = absent, Row 1 = present.
#     """
#     node = cpd_def["node"]
#     given = cpd_def.get("given", []) or []
#     probs = cpd_def.get("probabilities", {}) or {}
#
#     if not given:
#         p_present = float(probs["present"])
#         p_absent = float(probs["absent"])
#         return TabularCPD(
#             variable=node, variable_card=2,
#             values=[[round(p_absent, 4)], [round(p_present, 4)]],
#             state_names={node: ["absent", "present"]}
#         )
#
#     parent_states = ["present", "absent"]
#     combos = list(product(parent_states, repeat=len(given)))
#
#     present, absent = [], []
#     for combo in combos:
#         key = ", ".join(combo)
#         k1 = f"present | {key}"
#         k0 = f"absent | {key}"
#         p1 = float(probs.get(k1, 0.0))
#         p0 = float(probs.get(k0, 1.0 - p1))
#         present.append(round(p1, 4))
#         absent.append(round(p0, 4))
#
#     return TabularCPD(
#         variable=node, variable_card=2,
#         values=[absent, present],
#         evidence=given, evidence_card=[2] * len(given),
#         state_names={node: ["absent", "present"], **{p: ["absent", "present"] for p in given}}
#     )
#
#
# def generate_noisy_or_cpd(node: str, parents: List[str], weights: Dict[str, float], base_prob: float = 0.01
#                           ) -> TabularCPD:
#     """
#     Convert Noisy-OR (with optional leak 'base_prob') into TabularCPD.
#     weights: parent -> activation prob in (0,1)
#     base_prob: leak when all parents are absent.
#     """
#     present_probs, absent_probs = [], []
#     for combo in product([0, 1], repeat=len(parents)):
#         active = [weights[p] for p, v in dict(zip(parents, combo)).items() if int(v) == 1]
#         p_present = 1 - np.prod([1 - w for w in active]) if active else float(base_prob)
#         p_present = min(max(p_present, 0.0), 1.0)
#         p_absent = 1.0 - p_present
#         present_probs.append(round(p_present, 4))
#         absent_probs.append(round(p_absent, 4))
#
#     values = [absent_probs, present_probs]
#     state_names = {node: ["absent", "present"], **{p: ["absent", "present"] for p in parents}}
#     return TabularCPD(
#         variable=node, variable_card=2,
#         values=values,
#         evidence=parents, evidence_card=[2] * len(parents),
#         state_names=state_names
#     )
#
#
# def audit_evidence(model: BayesianNetwork, evidence_dict: Dict[str, Any]):
#     """
#     Returns (used_evidence, ignored_keys, per_node_state_info)
#     - used_evidence: {RealName: ValidStateLabel}
#     - ignored_keys: [keys not in model or without CPD]
#     - info: {RealName: [state labels in CPD order]}
#     """
#     def _lc(x): return str(x).strip().lower()
#     node_map: Dict[str, str] = {}
#     for n in model.nodes():
#         k = _lc(n)
#         if k in node_map and node_map[k] != n:
#             raise ValueError(f"Case-collision: {node_map[k]} vs {n}")
#         node_map[k] = n
#
#     used, ignored, info = {}, [], {}
#     for k, v in (evidence_dict or {}).items():
#         rn = node_map.get(_lc(k))
#         if not rn:
#             ignored.append(k); continue
#         cpd = model.get_cpds(rn)
#         if not cpd:
#             ignored.append(k); continue
#         states = list(cpd.state_names.get(rn, []))
#         info[rn] = states
#
#         # normalize value
#         if isinstance(v, (int, bool)):
#             sv = "present" if v in (1, True) else "absent"
#         else:
#             sv = str(v).strip().lower()
#             sv = {"1": "present", "true": "present", "yes": "present", "present": "present",
#                   "0": "absent", "false": "absent", "no": "absent", "absent": "absent"}.get(sv, sv)
#
#         states_map = { _lc(s): s for s in states }
#         if sv in states_map:
#             used[rn] = states_map[sv]
#         elif v in states:
#             used[rn] = v
#         else:
#             ignored.append(k)
#     return used, ignored, info


import json
import math
import re
import unicodedata
from typing import Dict, List, Tuple, Optional, Any
from itertools import product

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pyvis.network import Network


class BayesianNetworkBuilder:
    """
    Bayesian Network Builder with:
      - Canonical name mapping (+ optional aliases)
      - Stable display labels for nodes
      - CPD/parent sync (incl. Noisy-OR w/ leak)
      - Strict evidence ('present'/'absent' only; 0/1 auto-mapped)
      - Healthcheck (name mismatches, state order, model validity)
    """

    def __init__(self, causal_relations=None):
        self.model = BayesianNetwork()
        self.causal_relations = causal_relations or {"CAUSES", "CAN_LEAD_TO", "INCLUDES"}
        self.node_aliases: Dict[str, str] = {}
        self._canon: Dict[str, str] = {}  # normalized_key -> display label

    # ---------- Canonicalization / label control ----------

    def _canon_key(self, s: str) -> str:
        """Robust canonical key (for internal matching only)."""
        if s is None:
            return ""
        s = unicodedata.normalize("NFKC", str(s))
        s = s.replace("’", "'").replace("‘", "'")
        s = s.replace("–", "-").replace("—", "-")
        s = s.replace("\u00A0", " ")
        s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s.casefold()

    def _to_display(self, raw: str) -> str:
        """
        Resolve a raw label to the single display label we use in the model.
        First one we see wins; later variants map to the same display string.
        """
        key = self._canon_key(raw)
        if key in self._canon:
            return self._canon[key]
        display = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(raw))).strip()
        self._canon[key] = display
        return display

    def _normalize_node_name(self, name: str) -> str:
        """
        Canonical key for lookups. Lowercase + remove non-alphanumerics so
        'Shortness-of   breath' == 'shortness of breath'.
        """
        s = unicodedata.normalize("NFKC", str(name)).casefold()
        s = s.replace("\u00A0", " ")
        s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s

    def _name_map(self) -> Dict[str, str]:
        """Normalized-name -> display-name map for ALL nodes, plus aliases."""
        m = {self._normalize_node_name(n): n for n in self.get_available_nodes()}
        for alias, real in (self.node_aliases or {}).items():
            m[self._normalize_node_name(alias)] = real
        return m

    def add_node_alias(self, alias: str, actual_node: str):
        self.node_aliases[self._normalize_node_name(alias)] = actual_node

    # ---------- Parsing ----------

    def clean_response(self, response: Any) -> dict:
        """Parse an LLM response (string or dict) into a JSON dict."""
        if isinstance(response, dict):
            return response
        txt = (response or "").strip()
        if not txt:
            raise ValueError("Empty or invalid LLM response.")
        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s*```$", "", txt)
        txt = re.sub(r",\s*([\]}])", r"\1", txt)  # trailing commas
        try:
            return json.loads(txt)
        except json.JSONDecodeError as e:
            preview = txt[max(e.pos - 50, 0): e.pos + 50]
            raise ValueError(f"Invalid JSON from LLM: {e}\nContext: …{preview}…")

    # ---------- Model info ----------

    def get_available_nodes(self) -> List[str]:
        return list(self.model.nodes())



    from typing import Optional, Set, List

    def get_symptoms(
            self,
            category: str,
            symptom_rels: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Return a sorted list of symptom names scoped to the given `category`.

        Scope / heuristics:
          1) Prefer nodes attached to (:Category {name: category}) via [:PART_OF].
          2) Also read from heading-style nodes like 'Symptoms of <category>' / 'Signs of <category>'.
          3) Final fallback: an (:Entity) whose name equals the category.

        `symptom_rels` is a whitelist of relation types treated as symptom links.
        Defaults cover common patterns in your data (adjust as needed).
        """
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

    # ---------- Builder ----------

    @classmethod
    def from_llm_response(cls, llm_output: dict):
        """
        Build BN from LLM JSON:
          - Adds nodes/edges using stable display labels
          - Ensures CPD-implied parents exist
          - Converts Noisy-OR (supports 'noisy_or_leak') to TabularCPD
          - Adds priors only for true roots missing CPDs
        """
        builder = cls()

        nodes_in = llm_output.get("nodes", []) or []
        edges_in = llm_output.get("edges", []) or []
        cpds_raw = llm_output.get("cpds", []) or []
        cpds_list = cpds_raw if isinstance(cpds_raw, list) else [cpds_raw]

        # 1) Nodes
        for n in nodes_in:
            d = builder._to_display(n)
            if not builder.model.has_node(d):
                builder.model.add_node(d)

        # 2) Edges
        for e in edges_in:
            if not isinstance(e, dict) or "from" not in e or "to" not in e:
                continue
            u = builder._to_display(e["from"])
            v = builder._to_display(e["to"])
            if not builder.model.has_node(u): builder.model.add_node(u)
            if not builder.model.has_node(v): builder.model.add_node(v)
            if not builder.model.has_edge(u, v):
                builder.model.add_edge(u, v)

        # 3) Ensure CPD-implied parents exist
        for cpd_def in cpds_list:
            if not isinstance(cpd_def, dict): continue
            child = builder._to_display(cpd_def.get("node"))
            given = [builder._to_display(p) for p in (cpd_def.get("given", []) or [])]
            if not child: continue
            builder.model.add_node(child)
            for p in given:
                builder.model.add_node(p)
                if not builder.model.has_edge(p, child):
                    builder.model.add_edge(p, child)

        # 4) Build CPDs (use display labels)
        for cpd_def in cpds_list:
            if not isinstance(cpd_def, dict): continue
            try:
                node = builder._to_display(cpd_def.get("node"))
                if not node: continue
                given = [builder._to_display(p) for p in (cpd_def.get("given", []) or [])]

                if "noisy_or_weights" in cpd_def:
                    raw_w = cpd_def.get("noisy_or_weights", {}) or {}
                    cleaned_w = {builder._to_display(k): float(v) for k, v in raw_w.items()}
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

        # 5) Priors for true roots without CPD
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

        # 6) Validate
        builder.model.check_model()
        if len(builder.model.nodes()) == 0:
            raise RuntimeError("BN has zero nodes after build — check the JSON keys ('nodes', 'cpds', ...).")
        return builder

    # ---------- Evidence handling & inference ----------

    def filter_evidence_to_available_nodes(self, evidence: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Map raw evidence keys to real node names using the canonical map (+aliases).
        If duplicates map to the same node, 'present' wins.
        """
        name_map = self._name_map()
        filtered, ignored = {}, []
        for k, v in (evidence or {}).items():
            nk = self._normalize_node_name(k)
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
        """Only convert 0/1 (or '0'/'1') to 'absent'/'present'."""
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
        """
        Simple + safe:
          - Uses subject -> object as-is (after coercion to strings).
          - Optional filter by allowed_relations (strings like "CAUSES").
          - Skips duplicate edges; optionally skips edges that would create cycles.
        """
        # normalize whitelist once
        rel_whitelist = (
            {str(r or "").strip().upper().replace(" ", "_") for r in allowed_relations}
            if allowed_relations else None
        )

        for idx, triple in enumerate(triples):
            try:
                # Unpack flexibly; triple may be list/tuple
                if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                    continue
                s_raw, rel_raw, t_raw = triple[0], triple[1], triple[2]

                # Coerce everything to strings (handles lists/tuples)
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

                # Optionally avoid cycles
                if avoid_cycles:
                    # tentatively add and check
                    if not self.model.has_edge(s, t):
                        self.model.add_edge(s, t)
                        import networkx as nx
                        if not nx.is_directed_acyclic_graph(nx.DiGraph(self.model.edges())):
                            # revert if it creates a cycle
                            self.model.remove_edge(s, t)
                    # else edge already present
                else:
                    if not self.model.has_edge(s, t):
                        self.model.add_edge(s, t)

            except Exception as e:
                # Helpful debug in Streamlit or console
                try:
                    import streamlit as st
                    st.warning(f"Skipping triple #{idx} due to error: {e} | triple={triple}")
                except Exception:
                    print(f"[WARN] Skipping triple #{idx} due to error: {e} | triple={triple}")

        return self

    def _enforce_absent_present(self, evidence: Dict[str, Any]) -> Dict[str, str]:
        """Strict values: ONLY 'present' or 'absent' (case-insensitive)."""
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
        """
        Inference pipeline:
          1) Convert only 0/1 -> 'absent'/'present'.
          2) Map/merge evidence keys to model nodes via canonical map (+aliases).
          3) Enforce strict 'present'/'absent'.
          4) Drop evidence for nodes without CPDs or with invalid state labels.
          5) Resolve query via the same canonicalization and run VE.
        """
        if not self.model.check_model():
            raise ValueError("Model is invalid. Check CPDs and structure.")

        # (1) 0/1 -> strings
        ev = self._map_01_to_absent_present(evidence or {})

        # (2) map & merge keys
        mapped, ignored_input = self.filter_evidence_to_available_nodes(ev)

        # (3) strict values
        strict_evidence = self._enforce_absent_present(mapped)

        # (4) audit vs CPDs: only keep nodes with CPDs & valid state labels
        used, ignored_by_audit = {}, []
        for var, val in strict_evidence.items():
            cpd = self.model.get_cpds(var)
            if not cpd:
                ignored_by_audit.append(var); continue
            states = list(cpd.state_names.get(var, []))
            state_map = {s.lower(): s for s in states}
            vv = val.lower()
            if vv in state_map:
                used[var] = state_map[vv]
            else:
                ignored_by_audit.append(var)

        # (5) resolve query node
        name_map = self._name_map()
        qk = self._normalize_node_name(query_variable)
        if qk not in name_map:
            raise ValueError(f"Query node not found: '{query_variable}' (normalized '{qk}').")
        query_node = name_map[qk]

        # VE
        ve = VariableElimination(self.model)
        try:
            posterior = ve.query(variables=[query_node], evidence=used or None, show_progress=False)
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

        # Extract probs
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

    # ---------- Healthcheck / summary / viz ----------

    def healthcheck(self, bn_json: Optional[dict] = None, evidence: Optional[dict] = None) -> Dict[str, Any]:
        nodes = set(self.model.nodes())
        diag = {
            "missing_in_model": [],
            "extra_in_model": [],
            "unknown_evidence": [],
            "bad_state_order": [],
            "check_model_ok": None,
        }

        if bn_json:
            json_nodes = set(bn_json.get("nodes", []) or [])
            diag["missing_in_model"] = sorted(json_nodes - nodes)
            diag["extra_in_model"] = sorted(nodes - json_nodes)

            # warn if names used in edges/CPDs aren't listed in "nodes"
            used_names = set()
            for e in (bn_json.get("edges", []) or []):
                used_names.add(e.get("from")); used_names.add(e.get("to"))
            for c in (bn_json.get("cpds", []) or []):
                used_names.add(c.get("node"))
                for g in (c.get("given", []) or []):
                    used_names.add(g)
            ref_not_listed = [x for x in used_names if x and x not in json_nodes]
            if ref_not_listed:
                st.warning(f"References not in JSON 'nodes': {ref_not_listed}")

        if evidence:
            nm = self._name_map()
            diag["unknown_evidence"] = sorted([
                k for k in evidence if self._normalize_node_name(k) not in nm
            ])

        for cpd in (self.model.get_cpds() or []):
            sn = (cpd.state_names or {}).get(cpd.variable)
            if sn != ["absent", "present"]:
                diag["bad_state_order"].append({"node": cpd.variable, "state_names": sn})

        try:
            diag["check_model_ok"] = bool(self.model.check_model())
        except Exception as e:
            diag["check_model_ok"] = False
            st.warning(f"check_model() raised: {e}")

        return diag

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.model.nodes()),
            "edges": list(self.model.edges()),
            "num_nodes": len(self.model.nodes()),
            "num_edges": len(self.model.edges()),
            "cpds_count": len(self.model.get_cpds() or []),
            "is_valid": self.model.check_model() if self.model.nodes() else False,
        }

    def validate_model(self) -> bool:
        return bool(self.model.check_model())

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

        fs = _safe_figsize(figsize, len(list(self.model.nodes())))
        fig, ax = plt.subplots(figsize= fs)
        nx.draw(
            graph, pos, with_labels=True,
            node_color="lightcoral", node_size=1000,
            font_size=8, font_weight="bold", edge_color="gray", ax=ax
        )
        ax.set_title("Bayesian Network Structure for "+ target_node)
        return fig

    def _as_str(self, x) -> str:
        """Coerce labels to strings; flatten lists/tuples."""
        if x is None:
            return ""
        if isinstance(x, (list, tuple)):
            return " ".join(map(str, x)).strip()
        return str(x).strip()


    def _bucket_relation(self, rel: str) -> str:
        """Map a relation label to a semantic bucket for coloring."""
        u = (rel or "").upper()
        symptoms = {"SYMPTOMS", "SYMPTOMS_INCLUDE", "HAS_SYMPTOM", "INCLUDES"}
        treatments = {"TREATS", "INDICATED_FOR", "MANAGES", "RELIEVES", "TREATED_BY", "RECOMMENDED"}
        causes = {
            "CAUSES", "CAN_LEAD_TO", "LEADS_TO", "RESULTS_IN", "TRIGGERS",
            "MAY_CAUSE", "MAY_BE_LINKED_TO", "MAY_BE_ASSOCIATED_WITH",
            "INCREASES_RISK_OF", "IS_THOUGHT_TO_HAPPEN_WHEN"
        }
        if u in symptoms: return "symptom"
        if u in treatments: return "treatment"
        if u in causes: return "cause"
        return "other"

    # def st_visualize_bn(
    #         self,
    #         target_node: str,
    #         triples: list[tuple[str, str, str]] | None = None,
    #         height: int = 700,
    #         width: str = "100%",
    # ):
    #     """
    #     Streamlit visualization of the BN:
    #       - target_node in red
    #       - nodes directly connected to target colored by relation bucket derived from `triples`
    #       - other nodes lightblue
    #       - edges shown as in the BN (directed)
    #     Parameters
    #     ----------
    #     target_node : str
    #         The node you want to highlight (must match the label in the BN).
    #     triples : list[(src, rel, dst)] | None
    #         KG triples you used to build the BN. Used to color neighbors of target
    #         by relation type. If None, we’ll color only the target in red and
    #         leave everything else in lightblue.
    #     """
    #     # Build a quick lookup of neighbor colors, based on relation types that touch target
    #     neighbor_color: dict[str, str] = {}
    #     edge_color: dict[tuple[str, str], str] = {}
    #
    #     if triples:
    #         for s, r, t in triples:
    #             # Normalize textual labels a bit (trim)
    #             s1 = (s or "").strip()
    #             t1 = (t or "").strip()
    #             bucket = self._bucket_relation(r)
    #
    #             # If target -> neighbor or neighbor -> target, color that neighbor and the edge
    #             if s1 == target_node:
    #                 if bucket == "symptom":
    #                     neighbor_color[t1] = "#7bd389"  # green-ish
    #                 elif bucket == "treatment":
    #                     neighbor_color[t1] = "#f5a742"  # orange-ish
    #                 elif bucket == "cause":
    #                     neighbor_color[t1] = "#e5c100"  # gold-ish
    #                 else:
    #                     neighbor_color[t1] = "#add8e6"  # lightblue
    #                 edge_color[(s1, t1)] = neighbor_color[t1]
    #
    #             if t1 == target_node:
    #                 if bucket == "symptom":
    #                     neighbor_color[s1] = "#7bd389"
    #                 elif bucket == "treatment":
    #                     neighbor_color[s1] = "#f5a742"
    #                 elif bucket == "cause":
    #                     neighbor_color[s1] = "#e5c100"
    #                 else:
    #                     neighbor_color[s1] = "#add8e6"
    #                 edge_color[(s1, t1)] = neighbor_color[s1]
    #
    #     # Create PyVis network
    #     net = Network(height=f"{height}px", width=width, directed=True, notebook=False)
    #     net.barnes_hut()
    #
    #     # Add nodes with colors
    #     for node in self.model.nodes():
    #         if node == target_node:
    #             net.add_node(node, label=node, color="#ff4d4f", shape="ellipse")  # red
    #         elif node in neighbor_color:
    #             net.add_node(node, label=node, color=neighbor_color[node], shape="ellipse")
    #         else:
    #             net.add_node(node, label=node, color="#add8e6", shape="ellipse")  # lightblue
    #
    #     # Add edges; color those that touch target and we recognized via triples
    #     for u, v in self.model.edges():
    #         col = edge_color.get((u, v)) or edge_color.get((v, u))  # cover both directions
    #         if col:
    #             net.add_edge(u, v, color=col, arrows="to")
    #         else:
    #             net.add_edge(u, v, color="#999999", arrows="to")  # default gray
    #
    #     # Render to HTML string and embed in Streamlit
    #     html_str = net.generate_html()
    #     st.components.v1.html(html_str, height=height, scrolling=True)

    # ---------- Parent/CPD sync ----------
    @staticmethod
    def _sync_model_parents_with_cpd(model, cpd: TabularCPD) -> None:
        """(Unused inside class now) Kept for compatibility if you call it externally."""
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


# ---------- Helpers (module-level) ----------

def build_tabular_cpd_from_definition_full(cpd_def: dict) -> TabularCPD:
    """
    Build TabularCPD from {"node": str, "given": [..], "probabilities": {...}}.
    Two states: absent/present. Row0=absent, Row1=present.
    """
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
    """
    Convert Noisy-OR (with optional leak 'base_prob') into TabularCPD.
    weights: parent -> activation prob in (0,1)
    base_prob: leak when all parents are absent.
    """
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
    """Ensure graph parents match CPD.evidence (used by from_llm_response)."""
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

def _safe_figsize(figsize, n_nodes: int) -> tuple[float, float]:
    """Return a valid (w,h) for matplotlib, defaulting based on graph size."""
    # derive a default that scales with node count
    if n_nodes is None:
        n_nodes = 0
    w_default = min(20.0, max(6.0, 0.35 * max(1, n_nodes)))
    h_default = min(14.0, max(4.0, 0.28 * max(1, n_nodes)))
    default = (w_default, h_default)

    # if not provided, use default
    if figsize is None:
        return default

    # try to coerce to (float, float)
    try:
        w, h = figsize  # must be iterable of length 2
        w = float(w)
        h = float(h)
        if not (math.isfinite(w) and math.isfinite(h) and w > 0 and h > 0):
            return default
        return (w, h)
    except Exception:
        return default


from typing import Optional, Set, List




