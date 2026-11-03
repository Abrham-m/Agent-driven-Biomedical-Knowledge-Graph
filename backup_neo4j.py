# backup_neo4j.py
"""Export every node and relationship (with all properties) from Neo4j to JSON.

Usage: python backup_neo4j.py [--out backup/graph_backup_<date>.json]
Credentials are read from .env (NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD) and never printed.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


def export_graph(out_path: Path) -> dict:
    load_dotenv()
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )
    try:
        with driver.session() as session:
            nodes = [
                r.data()
                for r in session.run(
                    "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties"
                )
            ]
            rels = [
                r.data()
                for r in session.run(
                    "MATCH (a)-[r]->(b) RETURN elementId(r) AS id, type(r) AS type, "
                    "elementId(a) AS start_id, elementId(b) AS end_id, properties(r) AS properties"
                )
            ]
    finally:
        driver.close()

    backup = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "node_count": len(nodes),
        "relationship_count": len(rels),
        "nodes": nodes,
        "relationships": rels,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(backup, f, indent=2, ensure_ascii=False, default=str)
    return backup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = Path("backup") / f"graph_backup_{datetime.now().strftime('%Y-%m-%d')}.json"
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()
    b = export_graph(args.out)
    print(f"✅ Backed up {b['node_count']} nodes and {b['relationship_count']} relationships to {args.out}")
