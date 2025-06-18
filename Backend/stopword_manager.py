from typing import Set
import nltk
from nltk.corpus import stopwords

class StopWordManager:
    _STOPWORDS: Set[str] | None = None
    @classmethod
    def load(cls) -> Set[str]:
        if cls._STOPWORDS is None:
            nltk.download("stopwords", quiet=True)
            cls._STOPWORDS = (
                set(stopwords.words("romanian"))
                | set(stopwords.words("english"))
                | {"etc", "ie", "eg", "et", "al", "fig", "figure"}
            )
        return cls._STOPWORDS