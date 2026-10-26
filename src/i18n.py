"""
Module d'internationalisation (i18n) pour OEIS Python Clone.
Prend en charge :
- Le paramètre d'URL (ex: ?language=spanish ou ?language=french ou ?language=de)
- La détection automatique basée sur l'en-tête Accept-Language du navigateur
- Le chargement des traductions depuis src/locales/*.json
"""

import os
import json
import re
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

LOCALES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locales")

SUPPORTED_LANGUAGES = {"en", "fr", "de", "es"}

# Alias vers codes ISO 639-1
LANGUAGE_ALIASES: Dict[str, str] = {
    # Anglais
    "en": "en",
    "english": "en",
    # Français
    "fr": "fr",
    "french": "fr",
    "francais": "fr",
    "français": "fr",
    # Allemand
    "de": "de",
    "german": "de",
    "deutsch": "de",
    # Espagnol
    "es": "es",
    "spanish": "es",
    "espanol": "es",
    "español": "es",
}

# Noms d'affichage dans la langue cible
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
}

# Alias d'URL préférés dans le style de OEIS.org (?language=spanish)
LANGUAGE_URL_PARAM: Dict[str, str] = {
    "en": "english",
    "fr": "french",
    "de": "german",
    "es": "spanish",
}

_TRANSLATIONS_CACHE: Dict[str, Dict[str, str]] = {}


def load_translations(lang: str) -> Dict[str, str]:
    """Charge le fichier JSON de traduction pour la langue demandée."""
    if lang in _TRANSLATIONS_CACHE:
        return _TRANSLATIONS_CACHE[lang]

    file_path = os.path.join(LOCALES_DIR, f"{lang}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _TRANSLATIONS_CACHE[lang] = data
            return data

    # Repli sur l'anglais si indisponible
    if lang != "en":
        return load_translations("en")
    return {}


def normalize_language_tag(val: Optional[str]) -> Optional[str]:
    """Normalise un code ou nom de langue vers un code supporté (en, fr, de, es)."""
    if not val:
        return None
    cleaned = val.strip().lower()
    return LANGUAGE_ALIASES.get(cleaned)


def parse_accept_language(header_value: Optional[str]) -> Optional[str]:
    """
    Parse l'en-tête HTTP Accept-Language selon la RFC 9110 (RFC 7231).
    Ex: 'fr-CH,fr;q=0.9,en;q=0.8,de;q=0.7' -> 'fr'
    """
    if not header_value:
        return None

    items = []
    for part in header_value.split(","):
        part = part.strip()
        if not part:
            continue
        if ";" in part:
            tag, *params = part.split(";")
            tag = tag.strip()
            q = 1.0
            for param in params:
                param = param.strip()
                if param.startswith("q="):
                    try:
                        q = float(param[2:])
                    except ValueError:
                        q = 0.0
            items.append((tag, q))
        else:
            items.append((part, 1.0))

    # Tri par poids q décroissant
    items.sort(key=lambda x: x[1], reverse=True)

    for tag, _ in items:
        # Correspondance exacte (ex: fr)
        norm = normalize_language_tag(tag)
        if norm:
            return norm
        # Correspondance sur le préfixe principal (ex: fr-FR -> fr, es-ES -> es)
        if "-" in tag:
            primary = tag.split("-")[0]
            norm = normalize_language_tag(primary)
            if norm:
                return norm

    return None


def resolve_language(request: Optional[Any] = None, language_param: Optional[str] = None) -> str:
    """
    Détermine la langue à utiliser selon la priorité :
    1. Paramètre explicite 'language' ou 'lang' (query param)
    2. En-tête HTTP 'Accept-Language' du navigateur
    3. Langue par défaut ('en')
    """
    # 1. Vérification paramètre d'URL direct
    if language_param:
        norm = normalize_language_tag(language_param)
        if norm:
            return norm

    if request is not None:
        try:
            query_params = request.query_params
            url_lang = query_params.get("language") or query_params.get("lang")
            if url_lang:
                norm = normalize_language_tag(url_lang)
                if norm:
                    return norm
        except Exception:
            pass

        try:
            cookies = request.cookies
            cookie_lang = cookies.get("language")
            if cookie_lang:
                norm = normalize_language_tag(cookie_lang)
                if norm:
                    return norm
        except Exception:
            pass

        try:
            headers = request.headers
            accept_lang = headers.get("accept-language")
            header_lang = parse_accept_language(accept_lang)
            if header_lang:
                return header_lang
        except Exception:
            pass

    return "en"


def t(key: str, lang: str = "en", **kwargs: Any) -> str:
    """
    Traduit une clé dans la langue spécifiée.
    Si la clé n'existe pas dans la langue cible, bascule sur la version anglaise.
    Supporte l'interpolation avec des variables {var}.
    """
    lang_data = load_translations(lang)
    text = lang_data.get(key)
    if text is None:
        en_data = load_translations("en")
        text = en_data.get(key, key)

    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


def build_language_switch_url(current_url: str, target_lang: str) -> str:
    """
    Génère l'URL avec le paramètre ?language=<target> tout en conservant
    les autres paramètres éventuels (comme ?q=...).
    """
    parsed = urlparse(current_url)
    qs = parse_qs(parsed.query)
    # Style OEIS : ?language=french, ?language=spanish, etc.
    param_val = LANGUAGE_URL_PARAM.get(target_lang, target_lang)
    qs["language"] = [param_val]
    # Retirer l'ancien 'lang' s'il existe
    if "lang" in qs:
        del qs["lang"]
    new_query = urlencode(qs, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
