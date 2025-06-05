from typing import List,Tuple,Dict,Set
import spacy
from utils import Normalizer

class TripleExtractor:
    def __init__(self,nlp:"spacy.language.Language",stop:Set[str])->None:
        self.nlp=nlp
        self.stop=stop

    def extract(self, doc: "spacy.tokens.Doc") -> List[Tuple[str, str, str]]:
        tr = []
        for s in doc.sents:
            for tok in s:
                if tok.pos_ in {"VERB", "AUX"}:
                    subj = next((c for c in tok.children if c.dep_ in {"nsubj", "nsubjpass"}), None)
                    objs = [c for c in tok.children if c.dep_ in {"attr", "acomp", "dobj", "obj", "pobj", "obl", "xcomp", "ccomp"}]

                    for obj in objs:
                        if subj:
                            tr.append((
                                Normalizer.norm(subj.lemma_),
                                Normalizer.norm(tok.lemma_),
                                Normalizer.norm(obj.lemma_)
                            ))
        seen, uniq = set(), []
        for t in tr:
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        return uniq

    def to_graph(self,tr:List[Tuple[str,str,str]])->Dict[str,List[Dict]]:
        nodes,links,seen=[],[],set()
        for s,p,o in tr:
            if s not in seen:
                nodes.append({"id":s});seen.add(s)
            if o not in seen:
                nodes.append({"id":o});seen.add(o)
            links.append({"source":s,"target":o,"label":p})
        return{"nodes":nodes,"links":links}
