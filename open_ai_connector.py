from openai import OpenAI
import re
import json

client = OpenAI(
    api_key="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")
model = "gpt-4.1"


#Extract triples that came from test scraping
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

    output = response.choices[0].message.content
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


#Extract triples that came from test scraping
def generate_bn_structure_and_probabilities_from_llm(target_condition, causes, symptoms):
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

