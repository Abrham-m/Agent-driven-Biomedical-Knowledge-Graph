import json
import re
import os
import requests
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.prompts import PromptTemplate
import tiktoken

# === CONFIGURATION ===
#SPECIFIC_FILE = "s00262-020-02736-z.txt"
#CORE_ENTITY = "breast cancer"  # Update this per document
#OUTPUT_DIR = Path("./output")
#CLEANED_PAPERS_DIR = Path("./dataset/cleaned_papers")

# === LLM INITIALIZATION ===
# llm = OllamaLLM(
#     model="llama3.3:latest",
#     stop=["<Think>", "</Think>", "<|im_end|>"]
# )

load_dotenv()

# === Extraction model settings (recorded in runs/<run_id>/config.json) ===
EXTRACTION_TEMPERATURE = 0.0
GEMINI_MAX_OUTPUT_TOKENS = 65536  # gemini-2.5-flash max; 8192 truncated every paper (thinking tokens share this budget)
GEMINI_MAX_RETRIES = 6
GEMINI_MAX_WAIT_S = 120
OPENAI_MODEL = "gpt-4o"
OLLAMA_MODEL = "deepseek-coder:6.7b"

# Provenance of the most recent extract_relationships() call (logging only; never contains secrets)
LAST_CALL_INFO = {}


def configured_model_name(backend):
    if backend == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    return OPENAI_MODEL if backend == "openai" else OLLAMA_MODEL


relationship_extraction_prompt = PromptTemplate.from_template("""
Analyze this biomedical text and extract precise relationships between entities with scientific rigor.
The text may contain technical terminology and entity names with minor variations — use contextual understanding to match them.

Return ONLY a valid JSON array following this exact schema:
[{{
  "source": "EntityName (normalized form)",
  "relation": "SpecificBiomedicalRelation",
  "target": "EntityName (normalized form)",
}}]

Text: {text}
Entities: {entities}
Core Topic Entity: {core_entity}

Focus on these biomedical relationship types (ordered by priority):
1. Molecular interactions: binds_to, inhibits, activates, phosphorylates, regulates_expression_of  
2. Pharmacological: treats, contraindicates, metabolizes, potentiates, side_effect_of  
3. Diagnostic: diagnoses, biomarker_for, prognostic_indicator_of  
4. Pathological: causes, predisposes_to, complication_of, manifestation_of  
5. Genetic: associated_with, variant_of, encodes, coexpressed_with  
6. Anatomical: located_in, part_of, connected_to  

Extraction rules:
- Only extract relationships explicitly stated in the text
- For ambiguous cases, prefer more specific relationship types
- Split compound entities into separate relationships
- Try to extract meaningful relationships for as many entities as possible, but never extract unsupported relationships
- Ensure at least one extracted relationship involves the core topic entity: "{core_entity}"
- However, avoid making all relationships about "{core_entity}" unless the text explicitly supports that.
- Include diverse and relevant biomedical relationships, even if they do not directly reference "{core_entity}"

IMPORTANT: You MUST return ONLY a valid JSON array following the exact schema above.
Do not include any additional text, explanations, or markdown formatting.
The output should begin with [ and end with ].
                                                              
Example Output:
[
  {{
    "source": "metformin",
    "relation": "inhibits",
    "target": "mTORC1"
  }},
  {{
    "source": "BRCA1",
    "relation": "associated_with",
    "target": "breast cancer"
  }}
]

Output ONLY the JSON array with no additional commentary:
""")


#relationship_chain = relationship_extraction_prompt | llm

def gemini_generate(prompt, model=None, api_key=None):
    import time
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment")

    configured_model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    # Only use models confirmed available via ListModels
    fallback_models = ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]
    try_models = [configured_model] + [m for m in fallback_models if m != configured_model]

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 8192
        }
    }
    headers = {"Content-Type": "application/json"}
    error_list = []

    for m in try_models:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
        print(f"🤖 Trying Gemini model: {m}")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
                if response.status_code == 429:
                    # Parse retry-after from error message if available
                    err_body = response.json().get("error", {})
                    err_msg = err_body.get("message", "")
                    wait_match = re.search(r'retry in (\d+(?:\.\d+)?)s', err_msg)
                    wait_time = float(wait_match.group(1)) if wait_match else (30 * (attempt + 1))
                    wait_time = min(wait_time, 65)
                    print(f"⏳ Rate limited on {m}, waiting {wait_time:.1f}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue  # retry same model
                if response.status_code == 503:
                    wait_time = 15 * (attempt + 1)
                    print(f"⏳ Service overloaded on {m}, waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue  # retry same model
                if not response.ok:
                    err_msg = response.json().get("error", {}).get("message", response.text)
                    raise ValueError(f"HTTP {response.status_code}: {err_msg}")
                response_data = response.json()
                candidates = response_data.get("candidates", [])
                if not candidates:
                    raise ValueError("No candidate output returned from Gemini API")
                print(f"✅ Gemini responded using model: {m}")
                return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            except ValueError as e:
                # Only re-raise for non-retryable HTTP errors (not 429/503 which are handled above)
                error_list.append((m, type(e).__name__, str(e)))
                break  # move to next model
            except Exception as e:
                print(f"⚠️ Model {m} attempt {attempt+1} failed: {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    error_list.append((m, type(e).__name__, str(e)))
                break  # move to next model on non-HTTP errors

        else:
            # All retries exhausted for this model (only 429s hit this)
            print(f"❌ Model {m} exhausted all retries due to rate limiting, trying next model...")
            error_list.append((m, "RateLimitExhausted", "429 after all retries"))

    raise RuntimeError(f"Gemini generate failed for all model candidates: {error_list}")


def trim_to_token_limit(text, max_tokens=22000, model_name="gpt-4o"):
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)


def _redact(msg, secret):
    """Never let an API key leak into logs/errors."""
    msg = str(msg)
    return msg.replace(secret, "***") if secret else msg


def _retry_wait(response, attempt):
    """Seconds to wait after a 429/5xx: honour Gemini's RetryInfo when present, else back off."""
    try:
        err = response.json().get("error", {})
        for d in err.get("details", []):
            if "retryDelay" in d:
                return min(float(str(d["retryDelay"]).rstrip("s")) + 1, GEMINI_MAX_WAIT_S)
        m = re.search(r"retry in (\d+(?:\.\d+)?)s", err.get("message", ""))
        if m:
            return min(float(m.group(1)) + 1, GEMINI_MAX_WAIT_S)
    except Exception:
        pass
    return min(15 * 2 ** attempt, GEMINI_MAX_WAIT_S)


def gemini_generate(prompt, model=None, api_key=None):
    import time
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment")

    configured_model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    # Build deduplicated fallback list
    fallback_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    try_models = [configured_model] + [m for m in fallback_models if m != configured_model]
    LAST_CALL_INFO["model_requested"] = configured_model

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": EXTRACTION_TEMPERATURE,
            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS
        }
    }

    # API key goes in a header (not the URL) so it can never appear in exception messages
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    error_list = []
    for m in try_models:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
        print(f"🤖 Trying Gemini model: {m}")
        for attempt in range(GEMINI_MAX_RETRIES):
            try:
                response = requests.post(endpoint, json=payload, headers=headers, timeout=300)
                if response.status_code in (429, 500, 503, 504):
                    wait = _retry_wait(response, attempt)
                    print(f"⏳ HTTP {response.status_code} on {m}, waiting {wait:.0f}s "
                          f"(attempt {attempt + 1}/{GEMINI_MAX_RETRIES})...")
                    try:
                        print(f"   ↳ {_redact(response.json().get('error', {}).get('message', ''), api_key)[:500]}")
                    except Exception:
                        pass
                    LAST_CALL_INFO["retries"] = LAST_CALL_INFO.get("retries", 0) + 1
                    time.sleep(wait)
                    continue
                if not response.ok:
                    err_msg = response.json().get("error", {}).get("message", response.text)
                    raise ValueError(f"HTTP {response.status_code}: {err_msg}")
                response_data = response.json()
                candidates = response_data.get("candidates", [])
                if not candidates:
                    raise ValueError("No candidate output returned from Gemini API")
                print(f"✅ Gemini responded using model: {m}")
                LAST_CALL_INFO["model_used"] = m
                LAST_CALL_INFO["finish_reason"] = candidates[0].get("finishReason")
                usage = response_data.get("usageMetadata", {})
                LAST_CALL_INFO["usage"] = {k: usage.get(k) for k in
                                           ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount")}
                return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            except requests.RequestException as e:  # network/timeout: transient, retry
                wait = min(15 * 2 ** attempt, GEMINI_MAX_WAIT_S)
                print(f"⚠️ Model {m} network error ({type(e).__name__}), waiting {wait}s "
                      f"(attempt {attempt + 1}/{GEMINI_MAX_RETRIES})...")
                LAST_CALL_INFO["retries"] = LAST_CALL_INFO.get("retries", 0) + 1
                time.sleep(wait)
            except Exception as e:
                msg = _redact(e, api_key)
                print(f"⚠️ Model {m} failed: {type(e).__name__}: {msg}")
                error_list.append((m, type(e).__name__, msg))
                break  # non-retryable: move to next model
        else:
            print(f"❌ Model {m} exhausted {GEMINI_MAX_RETRIES} retries, trying next model...")
            error_list.append((m, "RetriesExhausted", f"{GEMINI_MAX_RETRIES} retryable failures"))

    raise RuntimeError(f"Gemini generate failed for all model candidates: {error_list}")


def extract_relationships(text, entities, core_entity, backend="ollama"):
    LAST_CALL_INFO.clear()
    LAST_CALL_INFO.update(backend=backend, model_requested=configured_model_name(backend),
                          temperature=EXTRACTION_TEMPERATURE)
    if backend != "gemini":
        LAST_CALL_INFO["model_used"] = configured_model_name(backend)
    try:
        # Dynamically choose backend
        if backend == "gemini":
            prompt_text = relationship_extraction_prompt.format(
                text=trim_to_token_limit(text),
                entities=json.dumps(entities),
                core_entity=core_entity
            )
            try:
                response_content = gemini_generate(prompt_text)
            except Exception as e:
                print(f"❌ Gemini failed: {e}")
                raise
        else:
            if backend == "openai":

                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model=OPENAI_MODEL, temperature=EXTRACTION_TEMPERATURE)
            else:
                from langchain_ollama import OllamaLLM
                llm = OllamaLLM(model=OLLAMA_MODEL, temperature=EXTRACTION_TEMPERATURE,
                                stop=["<Think>", "</Think>", "<|im_end|>"])

            chain = relationship_extraction_prompt | llm

            response = chain.invoke({
                "text": trim_to_token_limit(text),
                "entities": json.dumps(entities),
                "core_entity": core_entity
            })

            response_content = response.content if hasattr(response, "content") else str(response)
        
        # Print raw response for debugging
        print("\n=== RAW RESPONSE ===")
        print(response_content)
        print("===================\n")
        
        # Clean the response - remove markdown code blocks if present
        cleaned_response = re.sub(r'```json|```', '', response_content).strip()

        def fix_json(json_str):
            # Remove trailing commas before ] or }
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            # Remove comments
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
            return json_str

        def recover_partial_json(json_str):
            """Extract all complete JSON objects from a potentially truncated array."""
            objects = []
            depth = 0
            start = None
            for i, ch in enumerate(json_str):
                if ch == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start is not None:
                        fragment = json_str[start:i+1]
                        try:
                            obj = json.loads(fix_json(fragment))
                            objects.append(obj)
                        except json.JSONDecodeError:
                            pass
                        start = None
            return objects

        fixed_response = fix_json(cleaned_response)

        # Parse the JSON response — try full parse first, then recover partial
        relationships = None
        try:
            relationships = json.loads(fixed_response)
        except json.JSONDecodeError:
            # Try extracting a complete [...] block first
            json_match = re.search(r'\[.*\]', fixed_response, re.DOTALL)
            if json_match:
                try:
                    relationships = json.loads(fix_json(json_match.group(0)))
                except json.JSONDecodeError:
                    pass
            # If still broken (truncated), recover all complete objects
            if relationships is None:
                recovered = recover_partial_json(fixed_response)
                if recovered:
                    print(f"⚠️ Truncated response — recovered {len(recovered)} partial relationships")
                    relationships = recovered
                else:
                    raise ValueError("No valid JSON found in response")

        if relationships is None:
            raise ValueError("No valid JSON found in response")

        # Normalize to list format
        if isinstance(relationships, dict):
            relationships = [relationships]
        elif not isinstance(relationships, list):
            raise ValueError("Response is not a JSON array or object")

        # Validate and clean relationships
        valid_relationships = []
        for rel in relationships:
            if isinstance(rel, dict) and all(key in rel for key in ["source", "relation", "target"]):
                valid_relationships.append({
                    "source": rel["source"].strip(),
                    "relation": rel["relation"].strip(),
                    "target": rel["target"].strip()
                })

        # Print final output
        print("\n=== EXTRACTED RELATIONSHIPS ===")
        print(json.dumps(valid_relationships, indent=2))
        print("==============================\n")
        
        return valid_relationships

    except Exception as e:
        LAST_CALL_INFO["error"] = f"{type(e).__name__}: {e}"
        print(f"❌ Error in relationship extraction: {str(e)}")
        if 'response_content' in locals():
            print(f"Raw response was: {response_content[:1000]}")
        return []

    
def read_text_file(file_path):
    """Read content from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Failed to read text file: {str(e)}")

def main():
    output_dir = Path("./output")
    cleaned_papers_dir = Path("./dataset/cleaned_papers")  
    
    entities_path = output_dir / "final_entities.json"
    try:
        with open(entities_path, "r") as f:
            entities = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load entities: {str(e)}")


    text_file_path = cleaned_papers_dir / SPECIFIC_FILE  
    try:
        research_text = read_text_file(text_file_path)  
    except Exception as e:
        raise ValueError(f"Failed to read text file: {str(e)}")

    relationships = extract_relationships(research_text, entities)
    
    output_path = output_dir / "extracted_relationships.json"
    with open(output_path, "w") as f:
        json.dump(relationships, f, indent=2)
    
    print(f"Successfully extracted {len(relationships)} relationships")

if __name__ == "__main__":
    main()