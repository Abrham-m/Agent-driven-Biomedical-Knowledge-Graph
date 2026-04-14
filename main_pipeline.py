import os
import json
from pathlib import Path

from pdf_cleaner import process_all_pdfs
from agent_entity_extractor import extract_entities_from_file
from entity_cleaner import clean_entities
from agent_relationship_extractor import extract_relationships, read_text_file
from ontology_validator import validate
from agent_neo4j_adder import add_to_neo4j

# === Paths ===
CLEANED_DIR = Path("./dataset/cleaned_papers")
RESEARCH_DIR = Path("./dataset/research_papers")
OUTPUT_ROOT = Path("./output")


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
    process_all_pdfs(RESEARCH_DIR, CLEANED_DIR)

def run_entity_extraction():
    print("\n🔍 Extracting entities from cleaned papers...")
    for txt_file in CLEANED_DIR.glob("*.txt"):
        paper_id = txt_file.stem
        output_dir = OUTPUT_ROOT / paper_id
        extract_entities_from_file(txt_file.name, output_dir)

def run_entity_cleaning():
    print("\n🧹 Cleaning extracted entities...")
    for folder in OUTPUT_ROOT.iterdir():
        if folder.is_dir():
            input_path = folder / "extracted_entities.json"
            cleaned_output = folder / "cleaned_entities.json"
            final_output = folder / "final_entities.json"
            if input_path.exists():
                clean_entities(input_path, cleaned_output, final_output)

def run_relationship_extraction(core_entity, backend):
    import time
    print(f"\n🔗 Extracting relationships (core entity: {core_entity}) using [{backend}]...")
    folders = [f for f in OUTPUT_ROOT.iterdir() if f.is_dir()]
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

        with open(output_path, "w") as f:
            json.dump(relationships, f, indent=2)

        # Small delay between papers to avoid rate limiting (skip after last paper)
        if backend == "gemini" and i < len(folders) - 1:
            print("⏸️  Waiting 5s before next paper to respect rate limits...")
            time.sleep(5)

def run_validation():
    print("\n🧪 Validating relationships with NCIt ontology...")
    for folder in OUTPUT_ROOT.iterdir():
        if folder.is_dir():
            input_path = folder / "extracted_relationships.json"
            output_path = folder / "validated_relationships.json"
            if input_path.exists():
                validate(str(input_path), str(output_path))

def run_neo4j_store():
    for folder in OUTPUT_ROOT.iterdir():
        if folder.is_dir():
            rel_path = folder / "validated_relationships.json"
            if rel_path.exists():
                add_to_neo4j(paper_name=folder.name, relationships_path=rel_path)

def run_qa(model_choice: str = None):
    from agent_qa_feedback import main_loop
    print("\n🧠 Launching interactive QA system...")
    main_loop(model_choice)


# === Main Entry Point ===
def main():
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
            run_neo4j_store()
            run_qa()
        else:
            print("🛑 Reuse selected but no Neo4j storage requested.")
        return

    core_entity = get_core_entity()
    backend = get_model_backend()

    clean_all_pdfs()
    run_entity_extraction()
    run_entity_cleaning()
    run_relationship_extraction(core_entity, backend)
    run_validation()

    if ask_store_in_neo4j():
        run_neo4j_store()
        run_qa()
    else:
        print("✅ Pipeline completed. Results not stored to Neo4j.")


if __name__ == "__main__":
    main()
