import unicodedata

class Normalizer:
    @staticmethod
    def strip_diacritics(t: str) -> str:
        return ''.join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    @staticmethod
    def norm(t: str) -> str:
        return Normalizer.strip_diacritics(t.lower())