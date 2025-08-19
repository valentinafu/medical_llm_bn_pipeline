from openai import OpenAI
import re
import json

client = OpenAI(
    api_key="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")
model = "gpt-4.1"


def extract_triples(text):
    print("Extracting triples")
    prompt = f"""
    Extract medical triples like (Subject, Relation, Object)
    Text:
    \"\"\"{text}\"\"\"
    Return only triples.
    """
    print("Extracting triples")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    print('response', response)
    output = response.choices[0].message.content
    print('output', output)
    triples = []
    for line in output.strip().split("\n"):
        start_idx = line.find("(")
        end_idx = line.find(")", start_idx)
        if start_idx != -1 and end_idx != -1:
            triple_text = line[start_idx + 1:end_idx]
            parts = [p.strip() for p in triple_text.split(",")]
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def map_symptoms_to_dataset_columns_from_all(symptoms: list, diagnoses: list, all_dataset_columns: list) -> list:
    prompt = (
        f"Symptoms reported: {', '.join(symptoms)}\n"
        f"Diagnoses from the knowledge graph: {', '.join(diagnoses)}\n"
        f"Dataset columns: {', '.join(all_dataset_columns)}\n\n"
        f"Map the symptoms and diagnoses to the exact column names that are medically equivalent.\n"
        f"Always include 'age' and 'gender' if present in dataset.\n"
        f"Return a JSON dictionary like:\n"
        f'{{ "Coronary heart disease": "ihd", "age": "age" }}\n'
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        mapping = json.loads(response.choices[0].message.content)
        test = {k: v for k, v in mapping.items() if v in all_dataset_columns}
        return test
    except Exception as e:
        print(f"LLM mapping failed: {e}")
        return []


def map_probabilities_from_knowledge_graph(triples: list) -> dict:
    formatted_triples = "\n".join([f"- {src} {rel} {tgt}" for src, rel, tgt in triples])

    prompt = (
        f"You are helping build a Bayesian Network from medical knowledge.\n\n"
        f"Causal triples:\n{formatted_triples}\n\n"
        f"For each variable, provide:\n"
        f"- Prior probability as [P(No), P(Yes)] if it has no parents\n"
        f"- Conditional probability table (CPT) if it has parents\n\n"
        f"Return only a JSON dictionary like this:\n"
        f'{{\n'
        f'  "Smoking": [0.8, 0.2],\n'
        f'  "Lung Cancer|Smoking": [[0.9, 0.3], [0.1, 0.7]]\n'
        f'}}'
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        output = response.choices[0].message.content.strip()

        # Extract JSON
        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No valid JSON found in response")

        json_str = output[json_start:json_end]

        # Clean JSON: replace single quotes with double quotes and format keys
        json_str = re.sub(r"'", '"', json_str)
        json_str = re.sub(r'([{,]\s*)([A-Za-z0-9 _|]+)(\s*):', r'\1"\2"\3:', json_str)

        # Parse JSON
        probabilities = json.loads(json_str)
        return probabilities

    except json.JSONDecodeError as jde:
        print(f"JSON parsing failed: {jde}\nRaw output:\n{output}")
        return {}
    except Exception as e:
        print(f"LLM probability mapping failed: {e}")
        return {}



def generate_bn_structure_from_llm3(target_condition, causes, symptoms):
    def fmt(name, items):
        return f"- {name}:\n" + "\n".join([f"  - {x}" for x in items])

    prompt = f"""
You are a careful clinical modeler building a Bayesian Network (BN) for **{target_condition}**. 
Return **ONLY valid JSON** per the schema below.

INPUT
{fmt(f"Candidate causes of {target_condition}", causes)}

- Target condition: {target_condition}

{fmt("Observed symptoms", symptoms)}

OBJECTIVE
Produce a clinically grounded BN that supports inference and treatment reasoning.

HARD STRUCTURE RULES
1) Causal flow: risk_factors/mechanisms → {target_condition} → symptoms/complications → treatments/interventions.
2) Forbidden edges: 
   - treatment → anything (except escalation edges from symptom severity to a specific treatment; see #7)
   - symptom → cause/mechanism
   - treatment → treatment (unless modeling protocol escalation; avoid by default)
3) Acyclic graph only. All nodes reachable from or to the target condition (no isolated nodes).
4) Parent cap: Max 5 parents per node. If you exceed, introduce a mechanism node or use Noisy-OR.
5) Every symptom must be a descendant of {target_condition}. Every treatment must be a child of {target_condition} or a child of a severe symptom/complication.
6) Mechanism links are encouraged when physiologically standard (e.g., Prematurity → Weak lower esophageal sphincter → {target_condition}).
7) Treatment escalation (allowed): severe symptoms/complications may → specific treatments (e.g., Poor weight gain → Surgery). Do **not** let treatments feed back into causes or the target.

VARIABLE STATES
12) All variables binary with states exactly: "present", "absent".

CPD RULES
13) If no parents: prior must be numeric and realistic.
14) If 1–5 parents: provide a complete conditional table. Include `"condition_order"` that matches `given`.
    Example:
    {{
      "node": "Symptom A",
      "given": ["Parent1","Parent2"],
      "condition_order": ["present","absent"],   // for each parent, use same two-state order
      "probabilities": {{
        "present | present, absent": 0.70,
        "absent  | present, absent": 0.30,
        "present | absent, absent": 0.10,
        "absent  | absent, absent": 0.90
      }}
    }}
15) If >5 parents: use Noisy-OR with weights in [0.10, 0.30] and include an explicit `"noisy_or_leak"` in [0.01, 0.10].
    Example:
    {{
      "node": "{target_condition}",
      "given": ["P1","P2","P3","P4","P5","P6"],
      "noisy_or_weights": {{"P1":0.20,"P2":0.25,"P3":0.15}},
      "noisy_or_leak": 0.03
    }}
16) Probability hygiene: every distribution sums to 1 (within ±0.01), all values ∈ [0,1], no placeholders.

CALIBRATION HINTS (not hard constraints)
17) Prevalence priors for common risk factors ~0.05–0.30. 
18) Sensitivity of hallmark symptoms given condition often 0.60–0.90; specificity reflected by lower rates when {target_condition} is absent.
19) Treatment nodes model **recommendation/usage**, not efficacy outcomes.

SCHEMA (STRICT)
Return exactly:
{{
  "nodes": ["..."],                                // unique names
  "edges": [{{"from":"Parent","to":"Child"}}],     // parent→child
  "node_types": {{"Node": "risk_factor|mechanism|condition|symptom|complication|treatment"}},
  "cpds": [ /* CPD objects as per rules 13–15 */ ],
  "metadata": {{
    "target_condition": "{target_condition}",
    "merged_from": [{{"canonical":"Node","aliases":["Syn1","Syn2"]}}]
  }},
  "validation_report": {{
    "forbidden_edges": [],         // list any detected; must be empty
    "cycles_detected": false,
    "orphans": [],                 // nodes with no path to/from target
    "parent_cap_violations": [],   // nodes >5 parents
    "n_nodes": 0,
    "notes": "Brief QA comments"
  }}
}}

VALIDATION YOU MUST PASS BEFORE RETURNING
- No forbidden edges; no cycles.
- Every node in edges appears in nodes.
- {target_condition} exists and is type "condition".
- Symptom/treatment reachability rules satisfied.
- Parent cap respected or Noisy-OR used with leak.
- No duplicate names (case/space normalized).
- All tables normalized; values in [0,1].

Generate realistic numbers, NHS-style pediatric context. 
Return **JSON only**—no prose, no markdown.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a rule-following medical Bayesian modeler. Always obey schema and constraints exactly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content if response and response.choices else None




def generate_bn_structure_from_llm_work2(target_condition, causes, symptoms):
    def format_list(name, items):
        return f"- {name}:\n" + "\n".join([f"  - {item}" for item in items])

    prompt = f"""
You are a pediatric clinical reasoning expert and Bayesian modeler.

TASK
Build a clinically sound Bayesian Network (BN) about **{target_condition}** and return ONLY valid JSON.

INPUT
{format_list(f"Candidate causes of {target_condition}", causes)}

- Target condition: {target_condition}

{format_list("Observed symptoms", symptoms)}

GLOBAL MODELING PRINCIPLES
1) **Causal flow** must be: Risk factors/underlying mechanisms → {target_condition} → Symptoms/complications → Treatments/interventions.
2) **Never** draw edges from treatments to {target_condition} or to causes. Treatments are responses, not causes.
3) Keep node names concise and clinical (e.g., "Cow's milk allergy" not "Allergy to cow's milk").
4) Avoid duplicates and near-duplicates (merge “powder to thicken formula”, “thickening powder helps”, “pre-thickened formula” into one treatment concept).
5) You may add well-known pediatric risk factors **only if clinically standard** (e.g., prematurity, overfeeding, weak lower esophageal sphincter, cow’s milk allergy). Do not add speculative nodes.
6) Distinguish **states** (risk_factor/condition/symptom) from **interventions** (treatment). Interventions may be children of {target_condition} or of severe symptoms (i.e., recommended because symptoms are present), but must not be parents of them.
7) You may include intermediate mechanism nodes (e.g., “Weak lower esophageal sphincter”) when helpful.

VARIABLES & STATES
- All variables are binary with states exactly: "present", "absent".

EDGES CONSTRAINTS (hard rules)
- Allowed parents of {target_condition}: risk factors / mechanisms only.
- Allowed children of {target_condition}: symptoms, complications, treatments.
- Disallowed: treatment → {target_condition}, treatment → cause/mechanism, treatment → unrelated treatment.

CPDs (probabilities)
A) Nodes with no parents: supply a prior with numeric probabilities summing to 1.
   {{
     "node": "Obesity",
     "given": [],
     "probabilities": {{"present": 0.2, "absent": 0.8}}
   }}

B) Nodes with 1–5 parents: provide a **complete** table whose keys follow the exact parent order in `given`.
   {{
     "node": "Symptom A",
     "given": ["Condition X", "Condition Y"],
     "probabilities": {{
       "present | present, absent": 0.75, "absent | present, absent": 0.25,
       "present | absent, absent": 0.10, "absent | absent, absent": 0.90
     }}
   }}

C) Nodes with >5 parents: use a **noisy_or_weights** dictionary with weights in [0.1, 0.3].
   {{
     "node": "{target_condition}",
     "given": ["Parent1","Parent2","Parent3","Parent4","Parent5","Parent6"],
     "noisy_or_weights": {{"Parent1": 0.2, "Parent2": 0.25, "Parent3": 0.15}}
   }}

VALIDATION CHECKS (you must satisfy these)
- All probabilities are numeric in [0,1] and rows sum to 1.
- Every node in `edges` appears in `nodes`.
- No forbidden edge directions (see constraints above).
- No duplicate node names after case/whitespace normalization.

OUTPUT FORMAT (strict)
Return ONLY this JSON object:
{{
  "nodes": ["...","..."],                     // list of unique node names (strings)
  "edges": [{{"from": "Parent", "to": "Child"}}],  // parent→child
  "cpds": [/* CPD objects as specified above */],
  "node_types": {{                           // OPTIONAL but recommended
    "NodeName": "risk_factor" | "condition" | "symptom" | "treatment" | "mechanism"
  }}
}}

Generate realistic, clinically grounded numbers (NHS-style pediatric context). Do not include explanations or markdown—**JSON only**.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful, rule-following medical Bayesian modeler. Always obey schema and constraints exactly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    content = response.choices[0].message.content if response and response.choices else None
    return content

def generate_bn_structure_from_llm(target_condition, causes, symptoms):
    def format_list(name, items):
        return f"- {name}:\n" + "\n".join([f"  - {item}" for item in items])

    prompt = f"""
    You are a clinical reasoning expert and Bayesian modeler.

    Your task is to build a Bayesian Network (BN) in JSON format to support medical inference.

    {format_list(f"Causes of {target_condition}", causes)}

    - Target condition: {target_condition}

    {format_list("Symptoms observed in patients", symptoms)}

    **Rules for the Bayesian Network structure:**

    1. Causes (e.g., risk factors, underlying conditions) must lead to the **target condition** (e.g., cause → {target_condition}).
    2. The **target condition** must lead to its **symptoms** (e.g., {target_condition} → Shortness of breath).
    3. Do NOT create causal edges from **treatments to the condition**. Treatments may depend on the condition, not the other way around.
    4. Optional: treatments can be children of the target condition (e.g., {target_condition} → Treatment).
    5. Optional: include causal links between causes or symptoms if they are medically meaningful (e.g., Obesity → High blood pressure).
    6. You may add helpful **intermediate nodes** to represent mechanisms, complications, or latent processes.
    7. Symptoms may influence or amplify each other (e.g., Shortness of breath → Fatigue).
    8. Ensure all causal directions follow the medical timeline:
       - Causes occur **before** the condition
       - The condition leads to **symptoms**
       - Treatments are **decisions made after** diagnosis
    9. Use only **binary variables** ("present", "absent") for all nodes.
    10.Make sure all symptoms are included.
    10. For each CPD:
       - Use realistic **numeric probabilities**. Avoid symbolic placeholders like "P(...)".

       - If the node has **no parents**:
         Use a flat prior:
         ```json
         {{
           "node": "Obesity",
           "given": [],
           "probabilities": {{
             "present": 0.2,
             "absent": 0.8
           }}
         }}
         ```

       - If the node has **1 to 5 parents**:
         Use an expanded conditional probability table covering all parent combinations. Each key must reflect the condition order defined in the `given` list:
         ```json
         {{
           "node": "Symptom A",
           "given": ["Condition X", "Condition Y"],
           "probabilities": {{
             "present | present, absent": 0.75,
             "absent | present, absent": 0.25,
             "present | absent, absent": 0.1,
             "absent | absent, absent": 0.9
           }}
         }}
         ```

       - If the node has **more than 5 parents**:
         Use a **Noisy-OR model** and provide a dictionary of parent weights (between 0.1 and 0.3):
         ```json
         {{
           "node": "{target_condition}",
           "given": ["Obesity", "Smoking", "High blood pressure", "Diabetes", "Age", "Cholesterol"],
           "noisy_or_weights": {{
             "Obesity": 0.2,
             "Smoking": 0.3,
             "High blood pressure": 0.25
           }}
         }}
         ```

       - Only use `"present"` and `"absent"` as variable states.

    **JSON Format (strictly adhere to this):**
    {{
      "nodes": [...],
      "edges": [{{"from": "...", "to": "..."}}],
      "cpds": [...]
    }}

    Only return valid JSON. No explanations or markdown.
    """
    # Send to OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    content = response.choices[0].message.content if response and response.choices else None
    return content

