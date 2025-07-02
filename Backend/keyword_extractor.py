from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import spacy
from utils import Normalizer

# Extracts keywords using RAKE and TextRank
class KeywordExtractor:
    def __init__(self, nlp: "spacy.language.Language", stopwords: Set[str]) -> None:
        self.nlp = nlp  # SpaCy pipeline
        self.stopwords = stopwords  # Custom stopword set

    # Normalize a token using lemmatization
    def _normalize(self, token: "spacy.tokens.Token") -> str:
        return Normalizer.norm(token.lemma_)

    # RAKE IMPLEMENTATION

    # Extract candidate phrases (1 to 3 words, non-stopwords, alphabetic)
    def _rake_phrases(self, text: str) -> List[List[str]]:
        doc = self.nlp(text)
        phrase_list = []
        current_phrase = []
        for token in doc:
            if token.is_alpha and self._normalize(token) not in self.stopwords:
                current_phrase.append(self._normalize(token))
            elif current_phrase:
                phrase_list.append(current_phrase)
                current_phrase = []
        if current_phrase:
            phrase_list.append(current_phrase)
        return [phrase for phrase in phrase_list if 1 <= len(phrase) <= 3]

    # Compute word scores using frequency and degree
    def _word_scores(self, phrases: List[List[str]]) -> Dict[str, float]:
        word_freq = Counter()
        word_degree = defaultdict(int)
        for phrase in phrases:
            length = len(phrase)
            for word in phrase:
                word_freq[word] += 1
                word_degree[word] += length - 1
        return {word: (word_degree[word] + word_freq[word]) / word_freq[word] for word in word_freq}

    # Compute phrase scores by summing word scores
    def _phrase_scores(self, phrases: List[List[str]], word_scores: Dict[str, float]) -> Dict[str, float]:
        return {
            " ".join(phrase): sum(word_scores[word] for word in phrase)
            for phrase in phrases
        }

    # RAKE method to extract top-k keywords
    def rake(self, text: str, top_k: int = 10) -> List[Dict[str, float | str]]:
        phrases = self._rake_phrases(text)
        if not phrases:
            return []
        word_scores = self._word_scores(phrases)
        phrase_scores = self._phrase_scores(phrases, word_scores)
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        seen_norms = set()
        final_keywords = []
        for phrase, score in sorted_phrases:
            normalized = Normalizer.norm(phrase)
            if normalized in seen_norms:
                continue
            seen_norms.add(normalized)
            final_keywords.append({"keyword": phrase, "score": round(float(score), 4)})
            if len(final_keywords) == top_k:
                break
        return final_keywords

    # TEXTRANK IMPLEMENTATION

    def textrank(
        self,
        doc: "spacy.tokens.Doc",
        top_k: int = 10,
        window: int = 4,
        d: float = 0.85,
        convergence_threshold: float = 1e-5,
    ) -> tuple[list[dict], dict]:

        # Step 1: Extract candidate lemmas (nouns and adjectives)
        candidate_lemmas = []
        for i, token in enumerate(doc):
            if token.pos_ in {"NOUN", "ADJ"} and token.is_alpha:
                norm_lemma = Normalizer.norm(token.lemma_)
                if norm_lemma not in self.stopwords:
                    candidate_lemmas.append(norm_lemma)
        if not candidate_lemmas:
            return []

        # Step 2: Build graph (co-occurrence within window)
        vocab_list = list(dict.fromkeys(candidate_lemmas))
        word_to_index = {word: i for i, word in enumerate(vocab_list)}
        adjacency_matrix = np.zeros((len(vocab_list), len(vocab_list)), dtype=np.float32)

        for i in range(len(candidate_lemmas)):
            for j in range(i + 1, min(i + window, len(candidate_lemmas))):
                a, b = word_to_index[candidate_lemmas[i]], word_to_index[candidate_lemmas[j]]
                if a == b:
                    continue
                adjacency_matrix[a, b] += 1
                adjacency_matrix[b, a] += 1

        # Step 3: Normalize matrix and run PageRank
        row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(adjacency_matrix, row_sums, where=row_sums != 0)
        scores = np.random.rand(len(vocab_list)).astype(np.float32)

        while True:
            prev_scores = scores.copy()
            scores = (1 - d) + d * transition_matrix.T @ prev_scores
            if np.linalg.norm(scores - prev_scores, 1) < convergence_threshold:
                break

        word_scores = dict(zip(vocab_list, scores))

        # Step 4: Combine adjacent words into phrases and score them
        phrase_scores = {}
        i = 0
        while i < len(doc):
            token = doc[i]
            if token.pos_ in {"NOUN", "ADJ"} and token.is_alpha:
                norm1 = Normalizer.norm(token.lemma_)
                if norm1 in word_scores:
                    if i + 1 < len(doc):
                        token2 = doc[i + 1]
                        norm2 = Normalizer.norm(token2.lemma_)
                        if token2.pos_ in {"NOUN", "ADJ"} and token2.is_alpha and norm2 in word_scores:
                            phrase = f"{norm1} {norm2}"
                            score = word_scores[norm1] + word_scores[norm2]
                            phrase_scores[phrase] = score
                            i += 2
                            continue
                    phrase_scores[norm1] = word_scores[norm1]
            i += 1

        # Step 5: Sort phrases and return values
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        top_phrases = [
            {"keyword": phrase, "score": round(float(score), 4)}
            for phrase, score in sorted_phrases[:top_k]
        ]

        return top_phrases