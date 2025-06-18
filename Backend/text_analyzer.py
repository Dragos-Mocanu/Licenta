from typing import Any, Dict, Set, List
import spacy
from stopword_manager import StopWordManager
from keyword_extractor import KeywordExtractor
from triple_extractor import TripleExtractor
from utils import Normalizer

class TextAnalyzer:
    def __init__(self) -> None:
        self.nlp = spacy.load("ro_core_news_lg")
        self.stopwords: Set[str] = StopWordManager.load()
        self.kw = KeywordExtractor(self.nlp, self.stopwords)
        self.trp = TripleExtractor(self.nlp, self.stopwords)
    def _attach_ner(self, items: List[Dict], ents: Dict[str, str]) -> List[Dict]:
        for item in items:
            label = ents.get(item["keyword"])
            if label:
                item["ner"] = label
        return items
    def _token_set(self, keywords: Set[str]) -> Set[str]:
        return {Normalizer.norm(w) for kw in keywords for w in kw.split()}
    def _extract_relations(self, doc: "spacy.tokens.Doc", keywords: Set[str]) -> List[Dict]:
        rels, seen = [], set()
        kw_tokens = self._token_set(keywords)
        for tok in doc:
            src = Normalizer.norm(tok.head.lemma_)
            tgt = Normalizer.norm(tok.lemma_)
            if src == tgt:
                continue
            if src in kw_tokens or tgt in kw_tokens:
                edge = (src, tok.dep_, tgt)
                if edge in seen:
                    continue
                seen.add(edge)
                rels.append({"source": src, "label": tok.dep_, "target": tgt})
        return rels
    def analyze(self, txt: str) -> Dict[str, Any]:
        doc = self.nlp(txt)
        ents = {}
        for ent in doc.ents:
            ent_text = Normalizer.norm(ent.text.strip())
            ents[ent_text] = ent.label_
        for tok in doc:
            norm_tok = Normalizer.norm(tok.text.strip())
            if tok.ent_type_ and norm_tok not in ents:
                ents[norm_tok] = tok.ent_type_
        rake_kw = self._attach_ner(self.kw.rake(txt), ents)
        tr_kw, _ = self.kw.textrank(doc)
        tr_kw = self._attach_ner(tr_kw, ents)
        kw_phrases = {k["keyword"] for k in rake_kw} | {k["keyword"] for k in tr_kw}
        relations = self._extract_relations(doc, kw_phrases)
        triples = self.trp.extract(doc)
        kg = self.trp.to_graph(triples)
        qa = {"who": [], "what": [], "where": [], "when": [], "why": []}
        for tok in doc:
            if tok.dep_ in {"nsubj", "nsubjpass"} and tok.head.pos_ in {"VERB", "AUX"}:
                qa["who"].append(f"{Normalizer.norm(tok.text)} {Normalizer.norm(tok.head.lemma_)}")
        for s, p, o in triples:
            if p and o:
                qa["what"].append(f"{p} {o}")
        for key, label in ents.items():
            if label in {"LOC", "GPE", "FACILITY"}:
                qa["where"].append(key)
            elif label in {"DATE", "TIME", "DATETIME"}:
                qa["when"].append(key)
        for tok in doc:
            if tok.dep_ in {"obl", "prep"}:
                if tok.ent_type_ in {"DATE", "TIME", "DATETIME"}:
                    qa["when"].append(Normalizer.norm(tok.text))
                elif tok.ent_type_ in {"LOC", "GPE", "FACILITY"}:
                    qa["where"].append(Normalizer.norm(tok.text))
        qa["when"] = [w for w in qa["when"] if w not in {"anul", "luna"}]
        cause_phrases = []
        cause_markers = ["pentru că", "deoarece", "fiindcă", "din cauză că", "căci"]
        lowered = txt.lower()
        for marker in cause_markers:
            idx = lowered.find(marker)
            if idx != -1:
                fragment = lowered[idx:idx+200].split(".")[0].strip()
                cause_phrases.append(fragment)
        qa["why"].extend(Normalizer.norm(p) for p in cause_phrases)
        for key in qa:
            seen = set()
            phrases = sorted(qa[key], key=lambda x: -len(x.split()))
            clean = []
            for p in phrases:
                if p not in seen and not any(p != o and p in o for o in clean):
                    clean.append(p)
                    seen.add(p)
            qa[key] = clean
        ner_by_label = {}
        for ent_text, label in ents.items():
            if label not in ner_by_label:
                ner_by_label[label] = []
            ner_by_label[label].append(ent_text)
        for label in ner_by_label:
            phrases = sorted(ner_by_label[label], key=lambda x: -len(x.split()))
            clean = []
            for p in phrases:
                if not any(p in o and p != o for o in clean):
                    clean.append(p)
            ner_by_label[label] = clean
        return {
            "extractedText": txt,
            "rake": rake_kw,
            "textrank": tr_kw,
            "relations": relations,
            "kg": kg,
            "triples": [
                {"subject": s, "predicate": p, "object": o} for s, p, o in triples
            ],
            "qa": qa,
            "ner": ner_by_label
        }