from typing import List, Tuple, Dict, Set
import spacy
from utils import Normalizer

# Extracts subject-predicate-object triplets and builds a knowledge graph
class TripleExtractor:
    def __init__(self, nlp: "spacy.language.Language") -> None:
        self.nlp = nlp       # SpaCy language pipeline

    # Extract triplets from a spaCy Doc object
    def extract(self, doc: "spacy.tokens.Doc") -> List[Tuple[str, str, str]]:
        extracted_triplets = []

        # Iterate through all sentences in the document
        for sentence in doc.sents:
            for token in sentence:
                # Verb-centered triplets: subject - verb - object
                if token.pos_ in {"VERB", "AUX"}:
                    subject = next((child for child in token.children if child.dep_ in {"nsubj", "nsubjpass"}), None)
                    obj_tokens = [
                        child for child in token.children
                        if child.dep_ in {"attr", "acomp", "dobj", "obj", "pobj", "obl", "xcomp", "ccomp"}
                    ]
                    for obj in obj_tokens:
                        if subject:
                            extracted_triplets.append((
                                Normalizer.norm(subject.lemma_),
                                Normalizer.norm(token.lemma_),
                                Normalizer.norm(obj.lemma_)
                            ))

                # Adjective modifier: noun - amod - adjective
                if token.dep_ == "amod" and token.head.pos_ == "NOUN":
                    extracted_triplets.append((
                        Normalizer.norm(token.head.lemma_),
                        "amod",
                        Normalizer.norm(token.lemma_)
                    ))

                # Noun modifier: noun - nmod - noun
                if token.dep_ == "nmod" and token.head.pos_ == "NOUN":
                    extracted_triplets.append((
                        Normalizer.norm(token.head.lemma_),
                        "nmod",
                        Normalizer.norm(token.lemma_)
                    ))

        # Remove duplicates
        seen_triplets = set()
        unique_triplets = []
        for triplet in extracted_triplets:
            if triplet not in seen_triplets:
                seen_triplets.add(triplet)
                unique_triplets.append(triplet)
        return unique_triplets

    # Convert triplets to graph structure: nodes + labeled edges
    def to_graph(self, triplets: List[Tuple[str, str, str]]) -> Dict[str, List[Dict]]:
        nodes = []
        links = []
        seen_nodes = set()

        for subject, predicate, obj in triplets:
            # Add subject node if not seen
            if subject not in seen_nodes:
                nodes.append({"id": subject})
                seen_nodes.add(subject)
            # Add object node if not seen
            if obj not in seen_nodes:
                nodes.append({"id": obj})
                seen_nodes.add(obj)
            # Create link from subject to object with label as predicate
            links.append({"source": subject, "target": obj, "label": predicate})

        return {"nodes": nodes, "links": links}