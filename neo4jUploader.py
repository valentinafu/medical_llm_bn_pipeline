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
            """
            Make a Cypher-safe relationship type:
            - Replace spaces with underscores
            - Replace any non [A-Za-z0-9_] with underscores
            - Uppercase
            - Ensure it doesn't start with a digit by prefixing 'R_' if needed
            """
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
        rel_type = rel.upper().replace(" ", "_")  # sanitize relation into label-safe form
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

    def assign_category_from_triples(
            self,
            triples,  # Iterable[(subj, rel, obj)]
            category: str,  # the page/category name
            attach_all: bool = True,  # also attach all nodes from the set
            set_category_property: bool = False  # set n.category = category on those nodes
    ) -> dict:
        """
        Given a batch of triples you've *already inserted*, pick the 'center' node
        (highest degree *within this set*) and attach it to (:Category {name: category}).
        Optionally attach *all* nodes from the set and/or set a category property.

        Returns: {"center": <name or None>, "nodes_considered": <int>, "attached_all": bool}
        """
        # 1) Build compact rows of names from triples (subjects/objects only)
        rows, seen = [], set()
        names_s, names_o = [], []
        for s, _, o in triples:
            s = " ".join(str(s).split()) if s else None
            o = " ".join(str(o).split()) if o else None
            names_s.append(s)
            names_o.append(o)
            # Keep a unique list if you want to inspect later
            for nm in (s, o):
                if nm and nm not in seen:
                    seen.add(nm)
                    rows.append({"name": nm})

        if not rows or not category or not category.strip():
            return {"center": None, "nodes_considered": 0, "attached_all": False}

        cypher = """
        // Inputs
        WITH $rows AS rows, $cat AS cat, $attachAll AS attachAll, $setProp AS setProp

        // Resolve the component = all nodes that appeared in these triples
        UNWIND rows AS r
        MATCH (n:Entity {name: r.name})
        WITH cat, attachAll, setProp, collect(DISTINCT n) AS comp

        // If nothing matched (e.g., inserts haven't happened), try a seed by category name
        OPTIONAL MATCH (seed:Entity) WHERE toLower(seed.name) = toLower(cat)
        WITH comp, cat, attachAll, setProp, collect(seed) AS seeds
        WITH (CASE WHEN size(comp) = 0 THEN seeds ELSE comp END) AS comp, cat, attachAll, setProp

        // stop if still empty
        WHERE size(comp) > 0

        // Choose center by highest degree *within this set*
        UNWIND comp AS n
        OPTIONAL MATCH (n)-[e]-(m) WHERE m IN comp
        WITH cat, attachAll, setProp, comp, n, count(e) AS deg
        ORDER BY deg DESC
        WITH cat, attachAll, setProp, comp, head(collect(n)) AS center

        // Create page hub and attach center
        MERGE (g:Category {name: cat})
        MERGE (center)-[:PART_OF]->(g)

        // Optionally attach *all* nodes to the page hub
        FOREACH (_ IN CASE WHEN attachAll THEN comp ELSE [] END |
          MERGE (_)-[:PART_OF]->(g)
        )

        // Optionally set a category property on those nodes
        FOREACH (_ IN CASE WHEN setProp THEN comp ELSE [] END |
          SET _.category = cat
        )

        RETURN center.name AS centerName, size(comp) AS countNodes, attachAll AS attachedAll
        """

        params = {
            "rows": rows,
            "cat": category.strip(),
            "attachAll": bool(attach_all),
            "setProp": bool(set_category_property),
        }

        with self._session() as s:  # ensure your class has _session() (database-scoped)
            rec = s.run(cypher, params).single()
            if not rec:
                return {"center": None, "nodes_considered": 0, "attached_all": attach_all}
            return {
                "center": rec["centerName"],
                "nodes_considered": rec["countNodes"],
                "attached_all": rec["attachedAll"],
            }

    def attach_centers_for_page_components(
            self,
            triples,
            category: str,
            attach_all_nodes: bool = False,
    ) -> list[str]:
        # 1) Collect unique node names from triples
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

        # Keep ONE session open for reads + writes
        with self._session() as session:
            # 2) Pull edges among those names (undirected subgraph)
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

            # 3) Build adjacency
            adj = {n: set() for n in names}
            for row in rows:
                a, b = row["a"], row["b"]
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)

            # 4) Connected components
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

            # 5) Choose center per component and write attachments
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

    from typing import List, Optional

    def get_categories(self, fallback: Optional[str] = None) -> List[str]:
        """
        Returns all distinct category names from the graph, checking in this order:
          1) (:Category {name})
          2) (:Group {name})              [legacy]
          3) (:Entity {category:<name>})  [legacy]
        If none found and `fallback` is provided and there are any nodes,
        returns [fallback]. Otherwise returns [].
        """
        q_main = """
           // Collect categories from various sources
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

        q_any_nodes = "MATCH (n) RETURN count(n) AS cnt"

        with self._session() as s:  # or self.driver.session(database="neo4j")
            cats = [r["category"] for r in s.run(q_main)]
            if cats:
                return cats

            # No categories found. If caller provided a fallback and graph has data, return it.
            if fallback is not None:
                cnt = s.run(q_any_nodes).single()["cnt"]
                if cnt and cnt > 0:
                    return [fallback]

        return []

    from typing import Optional, Set, List, Tuple

    def get_causal_triples(
            self,
            category: Optional[str] = None,
            allowed_relations: Optional[Set[str]] = None,
            depth: int = 2,
    ) -> List[Tuple[str, str, str]]:
        """
        Returns (source, relation, target) triples.

        Behavior:
          • If `category` is provided:
              1) Try strict page scope: nodes attached to (:Category {name:category}).
              2) If hub has no members, fall back to seeding by an Entity whose name equals the category (case-insensitive).
              3) Traverse up to `depth` hops *inside the chosen scope only*.
              4) If that still finds nothing, final fallback: any triples where a or b name CONTAINS category (case-insensitive).
          • If `category` is None: return global single-hop triples (optionally filtered by allowed_relations).

        Notes:
          • `allowed_relations` is a whitelist. If None, we don't filter by type.
          • `depth` is enforced by filtering `length(p) <= depth` with a fixed *1..5 bound in MATCH.
        """
        depth = max(1, int(depth))

        # Build relationship type clause
        if allowed_relations:
            rels = sorted({self.sanitize_rel(r) for r in allowed_relations})
            type_union = ":" + "|".join(rels)  # e.g. ":CAUSES|CAN_LEAD_TO"
            hop_clause = f"{type_union}*1..5"  # generous upper bound; we'll filter by length(p) <= depth
            single_hop_clause = f"{type_union}"  # for global/single-hop
        else:
            hop_clause = "*1..5"
            single_hop_clause = ""

        with self._session() as s:
            # --- CATEGORY BRANCH ---
            if category:
                cat = category.strip()
                # Strict page-scope OR name-equal fallback, then in-scope multi-hop
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

                # Final fallback: if still empty, just fetch any triples mentioning the category by name (CONTAINS)
                if not rows:
                    q_loose = f"""
                    WITH toLower(trim($cat)) AS seed
                    MATCH (a:Entity)-[r{single_hop_clause}]->(b:Entity)
                    WHERE toLower(a.name) CONTAINS seed OR toLower(b.name) CONTAINS seed
                    RETURN DISTINCT a.name AS source, type(r) AS relation, b.name AS target
                    """
                    rows = s.run(q_loose, {"cat": cat}).data()

            # --- GLOBAL BRANCH (no category) ---
            else:
                q_global = f"""
                MATCH (a:Entity)-[r{single_hop_clause}]->(b:Entity)
                RETURN DISTINCT a.name AS source, type(r) AS relation, b.name AS target
                """
                rows = s.run(q_global).data()

        # Normalize output
        return [
            (r["source"], r["relation"], r["target"])
            for r in rows if r.get("source") and r.get("relation") and r.get("target")
        ]

    # def get_causal_triples(
    #         self,
    #         category: Optional[str] = None,
    #         allowed_relations: Optional[Set[str]] = None,
    #         depth: int = 2,
    # ) -> List[Tuple[str, str, str]]:
    #     """
    #     Return (source, relation, target) triples.
    #     - If `category` is None: return all single-hop edges of allowed relation types.
    #     - If `category` is set: expand up to `depth` hops *inside the page scope*:
    #         scope = { n | (n:Entity)-[:PART_OF]->(:Category {name: category}) }
    #       and only traverse allowed relation types; every node on the path stays in scope.
    #
    #     Example:
    #         triples = get_causal_triples("Heart Failure", depth=2)
    #     """
    #     # Default relations (tweak to your data)
    #     if allowed_relations is None:
    #         allowed_relations = {
    #             "CAUSES", "CAN_LEAD_TO", "RESULTS_IN",
    #             "INCLUDE", "INCLUDES", "HAS_SYMPTOM", "ASSOCIATED_WITH",
    #             "TREATS", "MANAGES", "RELIEVES", "INDICATED_FOR",
    #             "RISK_FACTOR", "INDICATES",
    #             "ADVICE", "WHEN_TO_GET_HELP", "MAY", "MIGHT",
    #         }
    #
    #     # Make a Cypher-safe | union for the variable-length pattern
    #     rels = sorted({self.sanitize_rel(r) for r in allowed_relations})
    #     if not rels:
    #         return []
    #     rel_union = "|".join(rels)
    #     depth = max(1, int(depth))
    #
    #     with self._session() as s:  # use your database-scoped session helper
    #         if not category:
    #             # Global, single-hop over allowed relations
    #             q = f"""
    #             MATCH (a:Entity)-[r:{rel_union}]->(b:Entity)
    #             RETURN DISTINCT a.name AS source, type(r) AS relation, b.name AS target
    #             """
    #             rows = s.run(q).data()
    #         else:
    #             # Page-scoped expansion: stay strictly inside the category slice
    #             q = f"""
    #             MATCH (c:Category {{name:$cat}})
    #             MATCH (n:Entity)-[:PART_OF]->(c)
    #             WITH collect(DISTINCT n) AS scope, $cat AS cat
    #
    #             UNWIND scope AS s
    #             MATCH p=(s)-[r:{rel_union}*1..{depth}]-(x)
    #             WHERE x IN scope AND ALL(n IN nodes(p) WHERE n IN scope)
    #
    #             WITH DISTINCT relationships(p) AS rs
    #             UNWIND rs AS e
    #             RETURN DISTINCT startNode(e).name AS source, type(e) AS relation, endNode(e).name AS target
    #             """
    #             rows = s.run(q, {"cat": category}).data()
    #
    #     # Ensure clean tuples and drop any nulls just in case
    #     out: List[Tuple[str, str, str]] = []
    #     for r in rows:
    #         a, rel, b = r.get("source"), r.get("relation"), r.get("target")
    #         if a and rel and b:
    #             out.append((a, rel, b))
    #     return out

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

                     // A1: category member -> symptom
                     CALL {
                 WITH seed, rels
                     MATCH (hub:Category)<-[:PART_OF]-(a:Entity)-[r]->(s:Entity)
                 WHERE toLower(hub.name) = seed \
                   AND type (r) IN rels
                     RETURN collect(DISTINCT s.name) AS A1
                     }

                     // A2: symptom - \
                     > category member
                     CALL {
                 WITH seed, rels
                     MATCH (hub:Category)<-[:PART_OF]-(s2:Entity)-[r2]->(a2:Entity)
                 WHERE toLower(hub.name) = seed \
                   AND type (r2) IN rels
                     RETURN collect(DISTINCT s2.name) AS A2
                     }

                 WITH seed, rels, coalesce (A1, []) + coalesce (A2, []) AS A

                     // B: heading- style nodes (e.g., "Symptoms of <cat>")
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


