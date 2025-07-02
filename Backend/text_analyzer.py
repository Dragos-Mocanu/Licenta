from typing import Any, Dict, Set, List
import spacy
import re
from stopword_manager import StopWordManager
from keyword_extractor import KeywordExtractor
from triple_extractor import TripleExtractor
from utils import Normalizer

# Main class responsible for processing text and extracting information
class TextAnalyzer:
    def __init__(self) -> None:
        # Load Romanian language model
        self.nlp = spacy.load("ro_core_news_lg")

        # Load stopwords
        self.stopwords: Set[str] = StopWordManager.load()

        # Keyword extractor (RAKE + TextRank)
        self.keyword_extractor = KeywordExtractor(self.nlp, self.stopwords)

        # Triplet extractor (subject–predicate–object)
        self.triple_extractor = TripleExtractor(self.nlp)

    # Normalize and split all keywords into individual tokens
    def _token_set(self, keywords: Set[str]) -> Set[str]:
        return {Normalizer.norm(w) for kw in keywords for w in kw.split()}

    # Extract dependency-based relations
    def _extract_relations(self, doc: "spacy.tokens.Doc", keywords: Set[str]) -> List[Dict]:
        relations, seen_relations = [], set()
        keyword_tokens = self._token_set(keywords)
        for token in doc:
            source = Normalizer.norm(token.head.lemma_)
            target = Normalizer.norm(token.lemma_)
            if source == target:
                continue
            if source in keyword_tokens or target in keyword_tokens:
                edge = (source, token.dep_, target)
                if edge in seen_relations:
                    continue
                seen_relations.add(edge)
                relations.append({"source": source, "label": token.dep_, "target": target})
        return relations

    # Main method to perform text analysis and return all results
    def analyze(self, txt: str) -> Dict[str, Any]:
        doc = self.nlp(txt)

        # Named Entity Recognition
        entities = {}
        for ent in doc.ents:
            ent_text = Normalizer.norm(ent.text.strip())
            entities[ent_text] = ent.label_
        for token in doc:
            norm_token = Normalizer.norm(token.text.strip())
            if token.ent_type_ and norm_token not in entities:
                entities[norm_token] = token.ent_type_

        # Keyword Extraction
        rake_keywords = self.keyword_extractor.rake(txt)
        textrank_keywords = self.keyword_extractor.textrank(doc)

        # Dependency Relations
        keyword_phrases = {k["keyword"] for k in rake_keywords} | {k["keyword"] for k in textrank_keywords}
        relations = self._extract_relations(doc, keyword_phrases)

        # Triplet Extraction
        triples = self.triple_extractor.extract(doc)
        kg = self.triple_extractor.to_graph(triples)

        # Question Answering
        qa = {"who": [], "what": [], "where": [], "when": [], "why": []}

        # WHO: Find subject–verb pairs
        for token in doc:
            if token.dep_ in {"nsubj", "nsubjpass"} and token.head.pos_ in {"VERB", "AUX"}:
                qa["who"].append(f"{Normalizer.norm(token.lemma_)} {Normalizer.norm(token.head.lemma_)}")

        # WHAT: Extract predicate–object from triplets
        for subj, pred, obj in triples:
            if pred and obj and pred not in {"amod", "nmod"}:
                qa["what"].append(f"{pred} {obj}")

        # WHERE/WHEN: Based on NER labels
        for key, label in entities.items():
            if label in {"LOC", "GPE", "FACILITY"}:
                qa["where"].append(key)
            elif label in {"DATE", "TIME", "DATETIME"}:
                qa["when"].append(key)

        # WHY: Look for causal phrases in raw text
        cause_phrases = []
        cause_markers = ["pentru că", "deoarece", "fiindcă", "din cauză că", "căci"]
        lowered = txt.lower()
        for marker in cause_markers:
            for match in re.finditer(rf"\b{re.escape(marker)}\b", lowered):
                idx = match.start()
                fragment = lowered[idx:idx + 200].split(".")[0].strip()
                cause_phrases.append(fragment)
        qa["why"].extend(Normalizer.norm(p) for p in cause_phrases)

        # QA remove duplicates
        for key in qa:
            seen = set()
            phrases = sorted(qa[key], key=lambda x: -len(x.split()))
            clean = []
            for phrase in phrases:
                if phrase not in seen and not any(phrase != other and phrase in other for other in clean):
                    clean.append(phrase)
                    seen.add(phrase)
            qa[key] = clean

        # NER remove duplicates
        ner_by_label = {}
        for ent_text, label in entities.items():
            if label not in ner_by_label:
                ner_by_label[label] = []
            ner_by_label[label].append(ent_text)
        for label in ner_by_label:
            phrases = sorted(ner_by_label[label], key=lambda x: -len(x.split()))
            clean = []
            for phrase in phrases:
                if not any(phrase in other and phrase != other for other in clean):
                    clean.append(phrase)
            ner_by_label[label] = clean

        # Final Result
        return {
            "extractedText": txt,
            "rake": rake_keywords,
            "textrank": textrank_keywords,
            "relations": relations,
            "kg": kg,
            "triples": [
                {"subject": s, "predicate": p, "object": o} for s, p, o in triples
            ],
            "qa": qa,
            "ner": ner_by_label
        }