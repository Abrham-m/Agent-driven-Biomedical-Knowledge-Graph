# agent_neo4j_adder.py

import json
import os
from pathlib import Path
from typing import Optional, Dict
from neo4j import GraphDatabase
from dotenv import load_dotenv


load_dotenv()

class Neo4jGraph:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        with self.driver.session() as session:
            session.run("RETURN 1")
        print("✅ Connected to Neo4j")

    def close(self):
        self.driver.close()

    def run_query(self, query: str, params: Optional[Dict] = None):
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

def _connect():
    return Neo4jGraph(os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))


def clear_graph():
    """Delete ALL nodes and relationships. Only called when explicitly requested (--clear-neo4j)."""
    g = _connect()
    g.run_query("MATCH (n) DETACH DELETE n")
    g.close()
    print("🗑  Neo4j database cleared")


def graph_counts():
    g = _connect()
    nodes = g.run_query("MATCH (n) RETURN count(n) AS c")[0]["c"]
    rels = g.run_query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    g.close()
    return {"nodes": nodes, "relationships": rels}


def add_to_neo4j(paper_name: str, relationships_path: Path, run_id: Optional[str] = None):
    if not relationships_path.exists():
        print(f"❌ {relationships_path} not found. Skipping.")
        return

    with open(relationships_path, 'r') as f:
        relationships = json.load(f)

    # Ensure expected fields exist
    for rel in relationships:
        if "source_ids" not in rel:
            rel["source_ids"] = []
        if "target_ids" not in rel:
            rel["target_ids"] = []
        # Provenance (flattened: Neo4j properties can't hold maps)
        prov = rel.get("provenance") or {}
        rel["chunk_id"] = prov.get("chunk_id")
        rel["evidence"] = [f"[{paper_name}] {s}" for s in prov.get("evidence_sentences", [])]
        rel["extraction_model"] = prov.get("model_used")

    # Only store relationships with both source and target grounded in the ontology
    relationships = [rel for rel in relationships if rel.get("valid_entities")]
    if not relationships:
        print(f"⚠️ {paper_name}: No validated relationships found to store in Neo4j.")
        return

    neo4j_graph = Neo4jGraph(
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD")
    )

    query = """
    UNWIND $rels AS rel

    MERGE (source:Entity {name: rel.source})
    ON CREATE SET source.source_paper = $paper_name
    ON MATCH SET source.source_paper = 
        CASE 
            WHEN NOT $paper_name IN split(source.source_paper, ',') 
            THEN coalesce(source.source_paper, '') + 
                 CASE WHEN source.source_paper IS NULL THEN '' ELSE ',' END + 
                 $paper_name 
            ELSE source.source_paper 
        END
    SET source.source_ids = rel.source_ids

    MERGE (target:Entity {name: rel.target})
    ON CREATE SET target.source_paper = $paper_name
    ON MATCH SET target.source_paper = 
        CASE 
            WHEN NOT $paper_name IN split(target.source_paper, ',') 
            THEN coalesce(target.source_paper, '') + 
                 CASE WHEN target.source_paper IS NULL THEN '' ELSE ',' END + 
                 $paper_name 
            ELSE target.source_paper 
        END
    SET target.target_ids = rel.target_ids

    MERGE (source)-[r:RELATED_TO {type: rel.requested_relation}]->(target)
    ON CREATE SET r.source_paper = $paper_name
    ON MATCH SET r.source_paper = 
        CASE 
            WHEN NOT $paper_name IN split(r.source_paper, ',') 
            THEN coalesce(r.source_paper, '') + 
                 CASE WHEN r.source_paper IS NULL THEN '' ELSE ',' END + 
                 $paper_name 
            ELSE r.source_paper
        END
    SET r.run_id = coalesce($run_id, r.run_id),
        r.extraction_model = coalesce(rel.extraction_model, r.extraction_model),
        r.chunk_ids = CASE
            WHEN rel.chunk_id IS NULL OR rel.chunk_id IN coalesce(r.chunk_ids, []) THEN r.chunk_ids
            ELSE coalesce(r.chunk_ids, []) + rel.chunk_id
        END,
        r.evidence = coalesce(r.evidence, []) + [s IN rel.evidence WHERE NOT s IN coalesce(r.evidence, [])]
    """

    neo4j_graph.run_query(query, {
        "rels": relationships,
        "paper_name": paper_name,
        "run_id": run_id
    })

    result = neo4j_graph.run_query("""
        MATCH (n) RETURN count(n) AS count, 'nodes' AS type
        UNION
        MATCH ()-[r]->() RETURN count(r) AS count, 'relationships' AS type
    """)
    node_count = next((r['count'] for r in result if r['type'] == 'nodes'), 0)
    rel_count = next((r['count'] for r in result if r['type'] == 'relationships'), 0)

    print(f"✅ {paper_name}: Added/updated {node_count} nodes and {rel_count} relationships")

    neo4j_graph.close()

#if __name__ == "__main__":
 #   main()
