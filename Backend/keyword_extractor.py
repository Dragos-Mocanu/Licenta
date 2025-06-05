from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import spacy
from utils import Normalizer

class KeywordExtractor:
    def __init__(self, nlp: "spacy.language.Language", stopwords: Set[str]) -> None:
        self.nlp = nlp
        self.stopwords = stopwords

    def _normalize(self, tok: "spacy.tokens.Token") -> str:
        return Normalizer.norm(tok.lemma_)

    def _rake_phrases(self, text: str) -> List[List[str]]:
        doc = self.nlp(text)
        phrases, current = [], []
        for tok in doc:
            if tok.is_alpha and self._normalize(tok) not in self.stopwords:
                current.append(self._normalize(tok))
            elif current:
                phrases.append(current)
                current = []
        if current:
            phrases.append(current)
        return [p for p in phrases if 1 <= len(p) <= 3]

    def _word_scores(self, phrases: List[List[str]]) -> Dict[str, float]:
        freq, deg = Counter(), defaultdict(int)
        for ph in phrases:
            L = len(ph)
            for w in ph:
                freq[w] += 1
                deg[w] += L - 1
        return {w: (deg[w] + freq[w]) / freq[w] for w in freq}

    def _phrase_scores(self, phrases: List[List[str]], w_scores: Dict[str, float]) -> Dict[str, float]:
        return {" ".join(ph): sum(w_scores[w] for w in ph) for ph in phrases}

    def rake(self, text: str, top_k: int = 10) -> List[Dict[str, float | str]]:
        phrases = self._rake_phrases(text)
        if not phrases:
            return []
        w_scores = self._word_scores(phrases)
        p_scores = self._phrase_scores(phrases, w_scores)
        ranked = sorted(p_scores.items(), key=lambda x: x[1], reverse=True)

        seen, final = set(), []
        for phrase, score in ranked:
            norm = Normalizer.norm(phrase)
            if norm in seen:
                continue
            seen.add(norm)
            final.append({"keyword": phrase, "score": round(float(score), 4)})
            if len(final) == top_k:
                break
        return final

    def textrank(
        self,
        doc: "spacy.tokens.Doc",
        top_k: int = 10,
        window: int = 4,
        d: float = 0.85,
        iters: int = 100,
        tol: float = 1e-5,
    ) -> tuple[list[dict], dict]:
        norm_lemmas = []
        for tok in doc:
            if tok.pos_ in {"NOUN", "ADJ"} and tok.is_alpha:
                norm = Normalizer.norm(tok.lemma_)
                if norm not in self.stopwords:
                    norm_lemmas.append(norm)

        if not norm_lemmas:
            return [], {"nodes": [], "links": []}

        vocab = list(dict.fromkeys(norm_lemmas))
        w2i = {w: i for i, w in enumerate(vocab)}
        W = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
        for i in range(len(norm_lemmas)):
            for j in range(i + 1, min(i + window, len(norm_lemmas))):
                a, b = w2i[norm_lemmas[i]], w2i[norm_lemmas[j]]
                if a == b:
                    continue
                W[a, b] += 1
                W[b, a] += 1

        row_sum = W.sum(axis=1, keepdims=True)
        M = np.divide(W, row_sum, where=row_sum != 0)
        S = np.full(len(vocab), 1.0 / len(vocab), dtype=np.float32)
        for _ in range(iters):
            prev = S.copy()
            S = (1 - d) + d * M.T @ prev
            if np.linalg.norm(S - prev, 1) < tol:
                break

        scores = list(zip(vocab, S))
        scores.sort(key=lambda x: x[1], reverse=True)

        seen, final = set(), []
        for lemma, score in scores:
            if lemma in seen:
                continue
            seen.add(lemma)
            final.append({"keyword": lemma, "score": round(float(score), 4)})
            if len(final) == top_k:
                break

        nodes = [{"id": k["keyword"]} for k in final]
        links, used = [], set()
        keys = {k["keyword"] for k in final}
        for tok in doc:
            src = Normalizer.norm(tok.head.lemma_)
            tgt = Normalizer.norm(tok.lemma_)
            if src == tgt:
                continue
            if src in keys or tgt in keys:
                edge = (src, tgt, tok.dep_)
                if edge in used:
                    continue
                used.add(edge)
                links.append({"source": src, "target": tgt, "label": tok.dep_})

        return final, {"nodes": nodes, "links": links}
