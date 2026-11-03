# run_logger.py
"""Run-level provenance logging. Observes pipeline outputs; never alters extraction/validation results.

Writes to runs/<run_id>/:
  config.json             - run settings (date, model, temperature, papers, ...)
  relations.jsonl         - one line per extracted relation, with source paper, chunk and evidence sentences
  entity_validation.jsonl - one line per candidate entity sent to NCIt, accepted/rejected + NCIt IDs
  summary.json            - counts, run time and errors
"""
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from nltk.tokenize import sent_tokenize

from agent_relationship_extractor import trim_to_token_limit

RUNS_ROOT = Path("./runs")
MAX_EVIDENCE_SENTENCES = 3


def new_run_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _norm(s):
    return " ".join(str(s).split())


def _git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": commit or None, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


class RunLogger:
    def __init__(self, run_id=None):
        self.run_id = run_id or new_run_id()
        self.dir = RUNS_ROOT / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.config = {}
        self.errors = []
        self.models_used = {}  # paper -> model that actually answered
        self.llm_calls = {}  # paper -> finish_reason / token usage / retries
        self.stats = {}
        self._ncit = None
        print(f"🗂  Run ID: {self.run_id}  (logs: {self.dir})")

    # --- generic ---
    def _write_json(self, name, obj):
        with open(self.dir / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def _append_jsonl(self, name, records):
        with open(self.dir / name, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def error(self, stage, paper, message):
        self.errors.append({"stage": stage, "paper": paper, "error": str(message)})
        print(f"❗ [{stage}] {paper}: {message}")

    # --- config ---
    def write_config(self, **fields):
        self.config.update({
            "run_id": self.run_id,
            "date": datetime.now(timezone.utc).isoformat(),
            "git": _git_state(),
            **fields,
        })
        self._write_json("config.json", self.config)

    # --- relation provenance ---
    def relation_provenance(self, paper, text, relationships, call_info):
        """Attach provenance to each relation dict (adds a 'provenance' key, nothing else changes).

        The extractor sends one chunk per paper: the full cleaned text trimmed to the token limit.
        Evidence sentences are found post hoc: sentences of that chunk mentioning both the source
        and the target (case-insensitive). The LLM itself does not return evidence.
        """
        chunk = trim_to_token_limit(text)
        chunk_sha = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        sentences = [_norm(s) for s in sent_tokenize(chunk)]
        lowered = [s.lower() for s in sentences]
        model_used = call_info.get("model_used")
        self.models_used[paper] = model_used
        self.llm_calls[paper] = {k: call_info.get(k) for k in ("finish_reason", "usage", "retries")}
        for rel in relationships:
            src, tgt = _norm(rel["source"]).lower(), _norm(rel["target"]).lower()
            both = [sentences[i] for i, s in enumerate(lowered) if src in s and tgt in s]
            rel["provenance"] = {
                "run_id": self.run_id,
                "source_paper": paper,
                "chunk_id": f"{paper}#chunk0",
                "chunk_sha256": chunk_sha,
                "chunk_chars": len(chunk),
                "chunk_truncated": len(chunk) < len(text),
                "evidence_sentences": both[:MAX_EVIDENCE_SENTENCES],
                "evidence_match": "both_entities" if both else "none",
                "evidence_method": "post-hoc sentence match (source and target both mentioned)",
                "backend": call_info.get("backend"),
                "model_used": model_used,
                "temperature": call_info.get("temperature"),
                "finish_reason": call_info.get("finish_reason"),
                "usage": call_info.get("usage"),
            }
        return relationships

    # --- validation logging ---
    def _validator(self):
        if self._ncit is None:
            from ontology_validator import NCItValidator
            self._ncit = NCItValidator()
        return self._ncit

    def log_validation(self, paper, extracted_path, validated_path, ner_final_path=None):
        """Join extracted + validated results (same order, 1:1), write provenance into the
        validated file, and log relations.jsonl and entity_validation.jsonl."""
        with open(extracted_path, encoding="utf-8") as f:
            extracted = json.load(f)
        with open(validated_path, encoding="utf-8") as f:
            validated = json.load(f)
        if len(extracted) != len(validated):
            self.error("validation-log", paper, "extracted/validated length mismatch; provenance not joined")
            return

        ner_final = set()
        if ner_final_path and Path(ner_final_path).exists():
            with open(ner_final_path, encoding="utf-8") as f:
                ner_final = {_norm(e).lower() for e in json.load(f)}

        rel_records, entities = [], {}
        for i, (ex, va) in enumerate(zip(extracted, validated)):
            prov = ex.get("provenance")
            if prov is not None:
                va["provenance"] = prov
            rel_records.append({
                "run_id": self.run_id, "paper": paper, "index": i,
                "source": ex["source"], "relation": ex["relation"], "target": ex["target"],
                "provenance": prov,
                "valid_entities": va["valid_entities"],
                "source_ids": va["source_ids"], "target_ids": va["target_ids"],
                "requested_relation_found": va["requested_relation_found"],
            })
            for role, name, ids in (("source", va["source"], va["source_ids"]),
                                    ("target", va["target"], va["target_ids"])):
                e = entities.setdefault(name, {"ids": ids, "roles": set(), "n_relations": 0})
                e["roles"].add(role)
                e["n_relations"] += 1

        with open(validated_path, "w", encoding="utf-8") as f:
            json.dump(validated, f, indent=2, ensure_ascii=False)

        v = self._validator()
        ent_records = []
        for name, e in entities.items():
            norm = v.normalize(name)
            match = "exact" if norm in v.entity_index else ("fuzzy" if e["ids"] else "none")
            ent_records.append({
                "run_id": self.run_id, "paper": paper, "entity": name, "normalized": norm,
                "status": "accepted" if e["ids"] else "rejected",
                "ncit_ids": e["ids"], "match_type": match,
                "roles": sorted(e["roles"]), "n_relations": e["n_relations"],
                "in_ner_final_entities": _norm(name).lower() in ner_final,
            })
        self._append_jsonl("relations.jsonl", rel_records)
        self._append_jsonl("entity_validation.jsonl", ent_records)

        s = self.stats
        s["relations_extracted"] = s.get("relations_extracted", 0) + len(validated)
        s["relations_valid_entities"] = s.get("relations_valid_entities", 0) + sum(r["valid_entities"] for r in validated)
        s["relations_with_evidence_sentence"] = s.get("relations_with_evidence_sentence", 0) + sum(
            1 for r in rel_records if r["provenance"] and r["provenance"]["evidence_sentences"])
        s["candidate_entities"] = s.get("candidate_entities", 0) + len(ent_records)
        s["accepted_entities"] = s.get("accepted_entities", 0) + sum(r["status"] == "accepted" for r in ent_records)
        s["rejected_entities"] = s.get("rejected_entities", 0) + sum(r["status"] == "rejected" for r in ent_records)

    # --- summary ---
    def finish(self, status="completed", **extra):
        self.stats.update(extra)
        self.config["models_used"] = self.models_used
        self.config["llm_calls"] = self.llm_calls
        self._write_json("config.json", self.config)
        summary = {
            "run_id": self.run_id,
            "status": status,
            "run_time_s": round(time.time() - self.started, 1),
            "stats": self.stats,
            "errors": self.errors,
        }
        self._write_json("summary.json", summary)
        print(f"\n📊 Run summary ({self.dir / 'summary.json'}):")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary
