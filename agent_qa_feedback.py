import os
import json
import requests
from neo4j import GraphDatabase
from dotenv import load_dotenv
from typing import List, Dict, Optional
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class EnhancedNeo4jGraph:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print("✅ Successfully connected to Neo4j")

    def close(self):
        self.driver.close()

    def query(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        try:
            with self.driver.session() as session:
                result = session.run(cypher, params or {})
                return [dict(record) for record in result]
        except Exception as e:
            print(f"⚠️ Query error: {e}")
            return []

class GeminiLLM:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment")
        self.configured_model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.fallback_models = ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]
        self.try_models = [self.configured_model] + [m for m in self.fallback_models if m != self.configured_model]

    def invoke(self, prompt: str):
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 1024
            }
        }
        headers = {"Content-Type": "application/json"}

        errors = []
        for model_candidate in self.try_models:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_candidate}:generateContent?key={self.api_key}"
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    print(f"✅ Gemini responded using model: {model_candidate}")
                    return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                errors.append((model_candidate, "NoCandidates"))
            except Exception as e:
                errors.append((model_candidate, type(e).__name__, str(e)))

        raise RuntimeError(f"Gemini API failed for all model candidates: {errors}")

class QAModel:
    def __init__(self, model_type: str = "ollama"):
        self.model_type = model_type
        self.llm = self._initialize_model()
        
    def _initialize_model(self):
        if self.model_type == "openai":
            return ChatOpenAI(
                model_name="gpt-4o",
                api_key=OPENAI_API_KEY
            )
        elif self.model_type == "gemini":
            return GeminiLLM()
        else:  # Default to Ollama
            return OllamaLLM(
                model="llama3.2:3b"
            )
    
    def invoke(self, prompt):
        return self.llm.invoke(prompt)

def initialize_services(model_choice: str):
    """Initialize both Neo4j and LLM services"""
    try:
        graph = EnhancedNeo4jGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        qa_model = QAModel(model_choice)
        return graph, qa_model
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        exit(1)

def ensure_graph_connected():
    """Reconnect to Neo4j if driver is closed"""
    global graph
    try:
        # Try a simple query to check connection
        graph.query("RETURN 1")
    except Exception:
        # Reconnect if failed
        print("🔄 Reconnecting to Neo4j...")
        graph = EnhancedNeo4jGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

def get_model_choice() -> str:
    """Prompt user to choose between Ollama, OpenAI, and Gemini"""
    while True:
        choice = input("Choose QA model backend ('ollama', 'openai', or 'gemini'): ").strip().lower()
        if choice in ["ollama", "openai", "gemini"]:
            return choice
        print("⚠️ Invalid choice. Please enter 'ollama', 'openai', or 'gemini'.")

def get_graph_data(question: str = None) -> List[Dict]:
    """Get relevant graph data based on question"""
    ensure_graph_connected()
    base_query = """
    MATCH (source:Entity)-[r:RELATED_TO]->(target:Entity)
    WHERE source.name IS NOT NULL AND target.name IS NOT NULL
    """
    
    if question:
        question_lower = question.lower()
        if "treat" in question_lower:
            base_query += " AND r.type = 'treats'"
        elif "cause" in question_lower:
            base_query += " AND r.type = 'causes'"
        elif "diagnos" in question_lower:
            base_query += " AND r.type = 'diagnoses'"
        elif "associated" in question_lower:
            base_query += " AND r.type = 'associated_with'"
    
    query = base_query + """
    RETURN 
        source.name AS source,
        "" AS source_label,
        source.source_ids AS source_ids,
        source.source_paper AS source_papers,
        r.type AS relation,
        target.name AS target,
        "" AS target_label,
        target.source_ids AS target_ids,
        target.source_paper AS target_papers,
        r.source_paper AS relation_papers
    ORDER BY r.type
    LIMIT 50
    """
    
    return graph.query(query)

def format_entity_info(entity: Dict) -> str:
    """Format entity information with IDs and papers"""
    info = entity['name']
    if entity.get('label') and entity['label'] != 'unknown':
        info += f" ({entity['label']})"
    
    if entity.get('source_ids'):
        info += f" [NCIT IDs: {', '.join(entity['source_ids'])}]"
    
    papers = set()
    if entity.get('source_papers'):
        papers.update(p.strip() for p in entity['source_papers'].split(",") if p.strip())
    
    if papers:
        info += f" [source: {', '.join(sorted(papers))}]"
    
    return info

qa_prompt = PromptTemplate.from_template("""
You are a precise biomedical knowledge graph assistant. Only use the provided relationships.

Available Relationships:
{graph_data}

Guidelines:
1. Only answer using the relationships shown above.
2. If information isn't available, say: "This information is not in the knowledge graph."
3. For entities:
   - Always include NCIT IDs if available.
   - List all source papers.
4. For relationships:
   - State the exact relationship type.
   - Cite supporting papers and NCIt IDs when available.
5. Never invent information.
6. If it is a general question, do not include paper sources unless they support the answer.

Question: {question}

Answer in this format:
<answer> [NCIT IDs if available] [source: papers if available]
""")

def answer_question(question: str, qa_model: QAModel) -> str:
    try:
        ensure_graph_connected()
        # Handle specific queries directly
        if "source id" in question.lower() or "ncit id" in question.lower():
            entity_name = question.split("for")[-1].split("about")[-1].strip()
            result = graph.query("""
                MATCH (e:Entity {name: $name})
                RETURN e.name AS name, e.source_ids AS ids, e.source_paper AS papers
                """, {"name": entity_name})
            
            if not result:
                return "This entity is not in the knowledge graph."
            
            entity = result[0]
            response = f"<{entity['name']}>"
            if entity.get('ids'):
                response += f" [NCIT IDs: {', '.join(entity['ids'])}]"
            if entity.get('papers'):
                papers = [p.strip() for p in entity['papers'].split(",") if p.strip()]
                response += f" [source: {', '.join(sorted(set(papers)))}]"
            return response
        
        # Handle source paper queries
        if "source paper" in question.lower():
            entity_name = question.split("for")[-1].split("about")[-1].strip()
            result = graph.query("""
                MATCH (e:Entity {name: $name})
                RETURN e.name AS name, e.source_paper AS papers
                """, {"name": entity_name})
            
            if not result or not result[0].get('papers'):
                return f"No source papers found for {entity_name} in the knowledge graph."
            
            papers = [p.strip() for p in result[0]['papers'].split(",") if p.strip()]
            return f"<{result[0]['name']}> [source: {', '.join(sorted(set(papers)))}]"
        
        # General question handling
        relationships = get_graph_data(question)
        if not relationships:
            return "No relevant information found in the knowledge graph."
        
        # Format the graph data for the prompt
        formatted_relationships = []
        all_papers = set()
        all_ids = set()
        for rel in relationships:
            source_info = format_entity_info({
                'name': rel['source'],
                'label': rel.get('source_label'),
                'source_ids': rel.get('source_ids'),
                'source_papers': rel.get('source_papers')
            })
            
            target_info = format_entity_info({
                'name': rel['target'],
                'label': rel.get('target_label'),
                'source_ids': rel.get('target_ids'),
                'source_papers': rel.get('target_papers')
            })
            
            # Collect all relevant papers and IDs for fallback citation
            papers = set()
            ids = set()
            for paper_field in ['source_papers', 'target_papers', 'relation_papers']:
                if rel.get(paper_field):
                    papers.update(p.strip() for p in rel[paper_field].split(",") if p.strip())
            if rel.get('source_ids'):
                ids.update(rel['source_ids'])
            if rel.get('target_ids'):
                ids.update(rel['target_ids'])
            
            all_papers.update(papers)
            all_ids.update(ids)
            
            rel_info = f"{source_info} --{rel['relation']}--> {target_info}"
            if papers:
                rel_info += f" [supported by: {', '.join(sorted(papers))}]"
            
            formatted_relationships.append(rel_info)
        
        # Get answer from selected model
        response = qa_model.invoke(qa_prompt.format(
            graph_data="\n".join(formatted_relationships),
            question=question
        ))
        
        # Convert response object to string and append citations if missing
        if hasattr(response, 'content'):
            answer_text = response.content
        elif hasattr(response, 'text'):
            answer_text = response.text
        else:
            answer_text = str(response)

        # Only append citations if the answer is positive (not a "not found" response)
        is_negative_response = any(phrase in answer_text.lower() for phrase in [
            "not in the knowledge graph",
            "no relevant information",
            "no information",
            "not available"
        ])
        
        if not is_negative_response and '[source:' not in answer_text and '[NCIT IDs:' not in answer_text:
            citations = []
            if all_ids:
                citations.append(f"NCIT IDs: {', '.join(sorted(all_ids))}")
            if all_papers:
                citations.append(f"source: {', '.join(sorted(all_papers))}")
            if citations:
                answer_text = f"{answer_text} [{' '.join(citations)}]"

        return answer_text
    
    except Exception as e:
        return f"Error processing question: {e}"

def show_graph_summary():
    """Show summary of the knowledge graph"""
    ensure_graph_connected()
    result = graph.query("""
    MATCH (n)
    RETURN labels(n)[0] AS type, count(*) AS count
    UNION
    MATCH ()-[r]->()
    RETURN type(r) AS type, count(*) AS count
    """)
    
    print("\n📊 Knowledge Graph Summary:")
    for row in result:
        print(f"- {row['type']}: {row['count']}")

def show_entity_types():
    """Show all entity types in the graph"""
    ensure_graph_connected()
    result = graph.query("""
    MATCH (e:Entity)
    WHERE e.label IS NOT NULL
    RETURN DISTINCT e.label AS type
    """)
    
    print("\n🏷️ Entity Types:")
    for row in sorted(result, key=lambda x: x['type']):
        print(f"- {row['type']}")

def main_loop(model_choice: str = None):
    """Main QA loop with model selection"""
    if model_choice is None:
        model_choice = get_model_choice()
    
    global graph, qa_model
    graph, qa_model = initialize_services(model_choice)
    
    print(f"\n🔍 Starting QA Session with {model_choice.upper()} (type 'exit' to end)")
    print("Available commands: 'summary', 'types'")
    
    try:
        while True:
            try:
                user_input = input("\nQuestion: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'exit':
                    break
                    
                if user_input.lower() == 'summary':
                    show_graph_summary()
                    continue
                    
                if user_input.lower() == 'types':
                    show_entity_types()
                    continue
                    
                answer = answer_question(user_input, qa_model)
                print(f"\n💡 {answer}")
                
            except KeyboardInterrupt:
                print("\nEnding QA session...")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
    finally:
        graph.close()

if __name__ == "__main__":
    graph = None
    qa_model = None
    try:
        main_loop()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        if graph:
            graph.close()