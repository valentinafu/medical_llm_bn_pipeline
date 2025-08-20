"""
This module defines the `Neo4jUploader` class, which provides utilities
to insert, organize, and query knowledge graph data in a Neo4j database.
Key features:
1. Connection management:
   - Opens and closes sessions for a Neo4j instance.
2. Data insertion:
   - sanitize_rel(rel): Cleans and standardizes relation labels into Neo4j formats.
   - insert_triple(subj, rel, obj): Inserts one triple (subject–relation–object) in the graph.
   - insert_triples(triples): Bulk inserts of the multiple triples.
3. Category and grouping:
   - attach_centers_for_page(triples, category, attach_all_nodes):
        Groups related nodes under a category which identifies the condition, identifies "center" nodes,
        and attaches them to the category.
   - get_categories(): Retrieves all available categories from the graph
4. Querying knowledge:
   - get_causal_triples(category, allowed_relations, depth):
        Returns causal triples (source, relation, target) for a given category
        with optional filtering and traversal depth.
   - get_symptoms(category, symptom_rels):
        Extracts symptom nodes connected to a given category through predefined
        symptom-like relations (e.g., HAS_SYMPTOM, PRESENTS_WITH).
Purpose: This class acts as the main bridge between extracted triples (from text/LLM)
         and Neo4j storage, enabling structured querying of medical
         knowledge for downstream tasks like Bayesian Network construction.
"""
import re
from typing import Optional, Set, List, Tuple
from neo4j import GraphDatabase


class Neo4jUploader:
    def __init__(self, uri, user, password, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def sanitize_rel(self, rel: str) -> str:
        if rel is None:
            rel = "RELATED_TO"
        rel = rel.replace(" ", "_")
        rel = re.sub(r"[^A-Za-z0-9_]", "_", rel)
        rel = rel.upper()
        if not re.match(r"^[A-Z_][A-Z0-9_]*$", rel):
            rel = "R_" + rel
        return rel or "RELATED_TO"

    def _session(self):
        return self.driver.session(database=self.database)

    def insert_triple(self, subj: str, rel: str, obj: str):
        # sanitize relation into label-safe form
        triple = self.sanitize_rel(rel)
        rel_type = rel.upper().replace(" ", "_")
        cypher = f"""
        MERGE (s:Entity {{name: $subj}})
        MERGE (o:Entity {{name: $obj}})
        MERGE (s)-[r:{rel_type}]->(o)
        ON CREATE SET r.original_label = $rel
        """
        with self._session() as session:
            session.run(cypher, {"subj": subj, "obj": obj, "rel": rel})

    def insert_triples(self, triples):
        for subj, rel, obj in triples:
            self.insert_triple(subj, rel, obj)

    def attach_centers_for_page(
            self,
            triples,
            category: str,
            attach_all_nodes: bool = False,
    ) -> list[str]:
        names, seen = [], set()
        for s, _, o in triples:
            for x in (s, o):
                if not x:
                    continue
                nm = " ".join(str(x).split())
                if nm and nm not in seen:
                    seen.add(nm)
                    names.append(nm)
        if not names or not category or not category.strip():
            return []
        with self._session() as session:
            q_edges = """
            UNWIND $names AS nm
            MATCH (a:Entity {name:nm})
            WITH collect(DISTINCT a) AS nodes
            UNWIND nodes AS a
            UNWIND nodes AS b
            WITH a,b WHERE id(a) < id(b)
            MATCH (a)-[r]-(b)
            RETURN a.name AS a, b.name AS b
            """
            rows = session.run(q_edges, {"names": names}).data()

            adj = {n: set() for n in names}
            for row in rows:
                a, b = row["a"], row["b"]
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)

            components, visited = [], set()
            for n in names:
                if n in visited:
                    continue
                comp, stack = [], [n]
                visited.add(n)
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    for v in adj.get(u, ()):
                        if v not in visited:
                            visited.add(v)
                            stack.append(v)
                components.append(comp)

            if not components:
                return []

            centers = []
            cat = category.strip()
            cy_attach = """
            MERGE (g:Category {name:$cat})
            WITH g
            MATCH (c:Entity {name:$center})
            MERGE (c)-[:PART_OF]->(g)
            """
            cy_all = """
            MERGE (g:Category {name:$cat})
            WITH g, $nodes AS nodes
            UNWIND nodes AS nm
            MATCH (n:Entity {name:nm})
            MERGE (n)-[:PART_OF]->(g)
            """
            for comp in components:
                comp_set = set(comp)
                deg = {u: len(adj.get(u, set()) & comp_set) for u in comp}
                center = sorted(comp, key=lambda u: (-deg[u], u))[0]
                centers.append(center)
                session.run(cy_attach, {"cat": cat, "center": center})
                if attach_all_nodes:
                    session.run(cy_all, {"cat": cat, "nodes": comp})
            return centers

    def get_categories(self, fallback: Optional[str] = None) -> List[str]:
        q_main = """
           CALL {
             MATCH (c:Category)
             RETURN DISTINCT c.name AS category
             UNION
             MATCH (g:Group)
             RETURN DISTINCT g.name AS category
             UNION
             MATCH (n:Entity)
             WHERE n.category IS NOT NULL AND trim(n.category) <> ''
             RETURN DISTINCT n.category AS category
           }
           RETURN category
           ORDER BY toLower(category)
           """

        query_any_nodes = "MATCH (n) RETURN count(n) AS cnt"

        with self._session() as s:
            cats = [r["category"] for r in s.run(q_main)]
            if cats:
                return cats
            if fallback is not None:
                cnt = s.run(query_any_nodes).single()["cnt"]
                if cnt and cnt > 0:
                    return [fallback]

        return []

    def get_causal_triples(
            self,
            category: Optional[str] = None,
            allowed_relations: Optional[Set[str]] = None,
            depth: int = 2,
    ) -> List[Tuple[str, str, str]]:

        depth = max(1, int(depth))

        if allowed_relations:
            rels = sorted({self.sanitize_rel(r) for r in allowed_relations})
            type_union = ":" + "|".join(rels)
            hop_clause = f"{type_union}*1..5"
            single_hop_clause = f"{type_union}"
        else:
            hop_clause = "*1..5"
            single_hop_clause = ""

        with self._session() as s:
            if category:
                cat = category.strip()
                q_scope = f"""
                WITH toLower(trim($cat)) AS seed, $depth AS d

                // 1) Collect hub members (if a Category exists)
                OPTIONAL MATCH (h:Category) WHERE toLower(trim(h.name)) = seed
                OPTIONAL MATCH (n:Entity)-[:PART_OF]->(h)
                WITH seed, d, collect(DISTINCT n) AS scope1

                // 2) Also collect a name-equal seed list
                OPTIONAL MATCH (m:Entity) WHERE toLower(trim(m.name)) = seed
                WITH d, scope1, collect(DISTINCT m) AS seeds

                // 3) Choose scope: hub members if any, otherwise the name seed
                WITH d, CASE WHEN size(scope1) > 0 THEN scope1 ELSE seeds END AS scope
                WHERE size(scope) > 0

                // 4) Traverse strictly inside scope; limit by length(p) <= depth
                UNWIND scope AS s0
                MATCH p = (s0)-[r{hop_clause}]-(x:Entity)
                WHERE x IN scope AND ALL(n IN nodes(p) WHERE n IN scope) AND length(p) <= d

                WITH DISTINCT relationships(p) AS rs
                UNWIND rs AS e
                RETURN DISTINCT startNode(e).name AS source, type(e) AS relation, endNode(e).name AS target
                """
                rows = s.run(q_scope, {"cat": cat, "depth": depth}).data()

                # Final: if still is empty, just fetch any triples mentioning the category by name
                if not rows:
                    q_loose = f"""
                    WITH toLower(trim($cat)) AS seed
                    MATCH (a:Entity)-[r{single_hop_clause}]->(b:Entity)
                    WHERE toLower(a.name) CONTAINS seed OR toLower(b.name) CONTAINS seed
                    RETURN DISTINCT a.name AS source, type(r) AS relation, b.name AS target
                    """
                    rows = s.run(q_loose, {"cat": cat}).data()

            else:
                q_global = f"""
                MATCH (a:Entity)-[r{single_hop_clause}]->(b:Entity)
                RETURN DISTINCT a.name AS source, type(r) AS relation, b.name AS target
                """
                rows = s.run(q_global).data()

        return [
            (r["source"], r["relation"], r["target"])
            for r in rows if r.get("source") and r.get("relation") and r.get("target")
        ]

    def get_symptoms(self, category: str, symptom_rels: Optional[Set[str]] = None) -> List[str]:
        if not category or not category.strip():
            return []

        if symptom_rels is None:
            symptom_rels = {
                "HAS_SYMPTOM", "SYMPTOMS_INCLUDE", "INCLUDES", "INCLUDE",
                "HAS_SIGN", "PRESENTS_WITH"
            }
        rels = sorted({self.sanitize_rel(r) for r in symptom_rels})
        cat = category.strip()

        cypher = """
                 WITH toLower($cat) AS seed, $rels AS rels

                     // category member -> symptom
                     CALL {
                 WITH seed, rels
                     MATCH (hub:Category)<-[:PART_OF]-(a:Entity)-[r]->(s:Entity)
                 WHERE toLower(hub.name) = seed \
                   AND type (r) IN rels
                     RETURN collect(DISTINCT s.name) AS A1
                     }

                     //: symptom -
                     > category member
                     CALL {
                 WITH seed, rels
                     MATCH (hub:Category)<-[:PART_OF]-(s2:Entity)-[r2]->(a2:Entity)
                 WHERE toLower(hub.name) = seed \
                   AND type (r2) IN rels
                     RETURN collect(DISTINCT s2.name) AS A2
                     }

                 WITH seed, rels, coalesce (A1, []) + coalesce (A2, []) AS A

                     // B: heading- style nodes
                     CALL {
                 WITH seed, rels
                     OPTIONAL MATCH (h:Entity)
                 WHERE toLower(h.name) IN ['symptoms of ' + seed \
                     , 'signs of ' + seed]
                     OPTIONAL MATCH (h)-[rh]- \
                     >(s3:Entity)
                 WHERE type (rh) IN rels
                     RETURN collect(DISTINCT s3.name) AS B
                     }

                     // C : fallback \
                 from entity named exactly like category
                     CALL {
                 WITH seed, rels
                     OPTIONAL MATCH (e:Entity) \
                 WHERE toLower(e.name) = seed
                     OPTIONAL MATCH (e)-[re]- \
                     >(s4:Entity)
                 WHERE type (re) IN rels
                     RETURN collect(DISTINCT s4.name) AS C
                     }
                 WITH coalesce (A, []) + coalesce (B, []) + coalesce (C, []) AS ABC
                     UNWIND ABC AS nm
                 WITH DISTINCT toString(nm) AS symName
                 WHERE symName IS NOT NULL \
                   AND trim (symName) <> ''
                     RETURN symName AS symptom
                 ORDER BY toLower(symptom) \
                 """

        with self.driver.session() as session:
            rows = session.run(cypher, {"cat": cat, "rels": rels}).data()

        return [r["symptom"] for r in rows]
