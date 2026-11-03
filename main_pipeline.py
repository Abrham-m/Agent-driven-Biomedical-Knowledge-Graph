import os
import sys
import json
import argparse
from pathlib import Path

from pdf_cleaner import process_all_pdfs
from agent_entity_extractor import extract_entities_from_file, MODEL_NAME as NER_MODEL_NAME
from entity_cleaner import clean_entities
import agent_relationship_extractor as rex
from agent_relationship_extractor import extract_relationships, read_text_file
from ontology_validator import validate
from agent_neo4j_adder import add_to_neo4j, clear_graph, graph_counts
from run_logger import RunLogger

# === Paths ===
CLEANED_DIR = Path("./dataset/cleaned_papers")
RESEARCH_DIR = Path("./dataset/research_papers")
OUTPUT_ROOT = Path("./output")

# Optional subset of papers (stems, e.g. "New_antibody-drug_conjugates_(ADCs)_in_breast_canc"); None = all
SELECTED_PAPERS = None

def _selected(name):
    return SELECTED_PAPERS is None or name in SELECTED_PAPERS


# === User Input Helpers ===
def get_user_choice():
    print("\n🧠 Biomedical Knowledge Graph Pipeline")
    return input("Do you want to reuse previous results? (yes/no): ").strip().lower() in ['yes', 'y']

def get_core_entity():
    return input("Enter the core topic entity (e.g. 'breast cancer'): ").strip()

def get_model_backend():
    while True:
        backend = input("Choose model backend ('ollama', 'openai', or 'gemini'): ").strip().lower()
        if backend in ["ollama", "openai", "gemini"]:
            return backend
        print("⚠ Invalid choice. Please enter 'ollama', 'openai', or 'gemini'.")

def ask_store_in_neo4j():
    return input("Do you want to store the output in Neo4j? (yes/no): ").strip().lower() in ['yes', 'y']


# === Pipeline Steps ===
def clean_all_pdfs():
    print("\n📚 Cleaning PDFs...")
    if SELECTED_PAPERS is None:
        process_all_pdfs(RESEARCH_DIR, CLEANED_DIR)
    else:
        from pdf_cleaner import extract_and_clean_pdf
        for stem in sorted(SELECTED_PAPERS):
            extract_and_clean_pdf(str(RESEARCH_DIR / f"{stem}.pdf"), str(CLEANED_DIR / f"{stem}.txt"))

def run_entity_extraction():
    print("\n🔍 Extracting entities from cleaned papers...")
    for txt_file in CLEANED_DIR.glob("*.txt"):
        if not _selected(txt_file.stem):
            continue
        paper_id = txt_file.stem
        output_dir = OUTPUT_ROOT / paper_id
        extract_entities_from_file(txt_file.name, output_dir)

def run_entity_cleaning():
    print("\n🧹 Cleaning extracted entities...")
    for folder in OUTPUT_ROOT.iterdir():
        if folder.is_dir() and _selected(folder.name):
            input_path = folder / "extracted_entities.json"
            cleaned_output = folder / "cleaned_entities.json"
            final_output = folder / "final_entities.json"
            if input_path.exists():
                clean_entities(input_path, cleaned_output, final_output)

def run_relationship_extraction(core_entity, backend, run=None):
    import time
    print(f"\n🔗 Extracting relationships (core entity: {core_entity}) using [{backend}]...")
    folders = [f for f in OUTPUT_ROOT.iterdir() if f.is_dir() and _selected(f.name)]
    for i, folder in enumerate(folders):
        txt_path = CLEANED_DIR / f"{folder.name}.txt"
        entity_path = folder / "final_entities.json"
        output_path = folder / "extracted_relationships.json"

        if not txt_path.exists() or not entity_path.exists():
            continue

        text = read_text_file(txt_path)
        with open(entity_path, "r") as f:
            entities = json.load(f)

        relationships = extract_relationships(text, entities, core_entity, backend)

        if run:
            info = dict(rex.LAST_CALL_INFO)
            if info.get("error"):
                run.error("relation-extraction", folder.name, info["error"])
            run.relation_provenance(folder.name, text, relationships, info)

        with open(output_path, "w") as f:
            json.dump(relationships, f, indent=2)

        # Small delay between papers to avoid rate limiting (skip after last paper)
        if backend == "gemini" and i < len(folders) - 1:
            print("⏸️  Waiting 5s before next paper to respect rate limits...")
            time.sleep(5)

def run_validation(run=None):
    print("\n🧪 Validating relationships with NCIt ontology...")
    for folder in OUTPUT_ROOT.iterdir():
        if folder.is_dir() and _selected(folder.name):
            input_path = folder / "extracted_relationships.json"
            output_path = folder / "validated_relationships.json"
            if input_path.exists():
                validate(str(input_path), str(output_path))
                if run:
                    run.log_validation(folder.name, input_path, output_path, folder / "final_entities.json")

def run_neo4j_store(run=None):
    for folder in OUTPUT_ROOT.iterdir():
        if folder.is_dir() and _selected(folder.name):
            rel_path = folder / "validated_relationships.json"
            if rel_path.exists():
                add_to_neo4j(paper_name=folder.name, relationships_path=rel_path,
                             run_id=run.run_id if run else None)

def run_qa(model_choice: str = None):
    from agent_qa_feedback import main_loop
    print("\n🧠 Launching interactive QA system...")
    main_loop(model_choice)


# === Run bookkeeping ===
def _count_json_items(filename):
    total = 0
    for folder in OUTPUT_ROOT.iterdir():
        p = folder / filename
        if folder.is_dir() and _selected(folder.name) and p.exists():
            with open(p, encoding="utf-8") as f:
                total += len(json.load(f))
    return total

def start_run(mode, backend, core_entity, run_id=None):
    run = RunLogger(run_id)
    papers = sorted(p.name for p in RESEARCH_DIR.glob("*.pdf") if _selected(p.stem))
    run.write_config(
        mode=mode,
        backend=backend,
        core_entity=core_entity,
        extraction_model=rex.configured_model_name(backend) if backend else None,
        temperature=rex.EXTRACTION_TEMPERATURE,
        max_output_tokens=rex.GEMINI_MAX_OUTPUT_TOKENS if backend == "gemini" else None,
        ner_model=NER_MODEL_NAME,
        num_papers=len(papers),
        papers=papers,
    )
    return run

def finish_run(run, stored, status="completed"):
    extra = {
        "papers_processed": len(run.models_used) or sum(
            1 for f in OUTPUT_ROOT.iterdir() if _selected(f.name) and (f / "validated_relationships.json").exists()),
        "ner_raw_entities": _count_json_items("extracted_entities.json"),
        "ner_final_entities": _count_json_items("final_entities.json"),
    }
    if stored:
        extra["neo4j"] = graph_counts()
    return run.finish(status=status, **extra)

def run_extraction_steps(core_entity, backend, run):
    """Steps 1-5, same order as before; `run` only adds logging."""
    clean_all_pdfs()
    run_entity_extraction()
    run_entity_cleaning()
    run_relationship_extraction(core_entity, backend, run)
    run_validation(run)


# === Non-interactive CLI ===
def parse_args(argv):
    p = argparse.ArgumentParser(description="Biomedical Knowledge Graph Pipeline (no args = interactive menu)")
    p.add_argument("--mode", choices=["full", "reuse", "qa"], help="full: run everything; reuse: re-validate "
                   "+ store existing outputs; qa: go straight to QA")
    p.add_argument("--backend", choices=["ollama", "openai", "gemini"], default="gemini")
    p.add_argument("--core-entity", default="breast cancer")
    p.add_argument("--store-neo4j", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clear-neo4j", action="store_true",
                   help="DELETE all existing Neo4j nodes/relationships before storing (default: off)")
    p.add_argument("--qa", action=argparse.BooleanOptionalAction, default=False,
                   help="launch interactive QA after storing (default: off)")
    p.add_argument("--run-id", help="run identifier for runs/<run_id>/ (default: timestamp)")
    p.add_argument("--papers", nargs="+", metavar="PAPER",
                   help="only process these papers (PDF file names, with or without .pdf); default: all")
    return p.parse_args(argv)

def main_cli(args):
    global SELECTED_PAPERS
    if args.papers:
        stems = {Path(p).stem if p.endswith(".pdf") else p for p in args.papers}
        missing = sorted(s for s in stems if not (RESEARCH_DIR / f"{s}.pdf").exists())
        if missing:
            sys.exit(f"❌ Unknown paper(s) (no PDF in {RESEARCH_DIR}): {missing}")
        SELECTED_PAPERS = stems
    if args.mode == "qa":
        run_qa(args.backend)
        return
    run = start_run(args.mode, args.backend if args.mode == "full" else None,
                    args.core_entity if args.mode == "full" else None, args.run_id)
    try:
        if args.mode == "full":
            run_extraction_steps(args.core_entity, args.backend, run)
        else:  # reuse
            run_validation(run)
        if args.store_neo4j:
            if args.clear_neo4j:  # only after extraction succeeded, so a failed run leaves the old graph intact
                clear_graph()
            run_neo4j_store(run)
        finish_run(run, stored=args.store_neo4j)
    except BaseException as e:
        run.error("pipeline", None, f"{type(e).__name__}: {e}")
        run.finish(status="failed")
        raise
    if args.store_neo4j and args.qa:
        run_qa(args.backend)


# === Main Entry Point ===
def main():
    if len(sys.argv) > 1:
        return main_cli(parse_args(sys.argv[1:]))

    print("\n🔁 Welcome to the Biomedical Knowledge Graph Pipeline")
    print("1. Run full pipeline")
    print("2. Go directly to QA")
    print("3. Exit")
    
    choice = input("Select an option (1/2/3): ").strip()

    if choice == "3":
        print("👋 Exiting.")
        return

    if choice == "2":
        run_qa()
        return

    # === Full Pipeline ===
    reuse = get_user_choice()
    if reuse:
        if ask_store_in_neo4j():
            run = start_run("reuse", None, None)
            run_validation(run)  # Re-validate to ensure fresh validated_relationships.json
            run_neo4j_store(run)
            finish_run(run, stored=True)
            run_qa()
        else:
            print("🛑 Reuse selected but no Neo4j storage requested.")
        return

    core_entity = get_core_entity()
    backend = get_model_backend()

    run = start_run("full", backend, core_entity)
    run_extraction_steps(core_entity, backend, run)

    if ask_store_in_neo4j():
        run_neo4j_store(run)
        finish_run(run, stored=True)
        run_qa()
    else:
        finish_run(run, stored=False)
        print("✅ Pipeline completed. Results not stored to Neo4j.")


if __name__ == "__main__":
    main()
