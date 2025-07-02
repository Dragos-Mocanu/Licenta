import unicodedata

# Utility class for normalizing text: lowercase + remove diacritics
class Normalizer:
    @staticmethod
    def strip_diacritics(t: str) -> str:
        # Normalize to decomposed form (NFD) then filter out non-spacing marks (category 'Mn')
        return ''.join(
            c for c in unicodedata.normalize("NFD", t)
            if unicodedata.category(c) != "Mn"
        )

    @staticmethod
    def norm(t: str) -> str:
        return Normalizer.strip_diacritics(t.lower())