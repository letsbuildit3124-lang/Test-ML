"""Record normalisation for the Business Entity Resolution challenge.

Produces, for every record, a set of canonical string fields that the blocking
and feature stages consume:

  name_norm     transliterated / accent-folded lowercase name, punctuation removed
  name_core     name_norm minus legal-form suffixes and stop tokens
  name_compact  name_core with spaces removed (robust to domain-style names)
  name_script   'latin' or the Unicode script of the raw name (e.g. 'DEVANAGARI')
  name_domain   1 if the raw name looked like a web domain (foo.com, www.foo.in)
  addr_norm     canonicalised address tokens (abbreviations folded, fillers dropped)
  addr_nums     digit-bearing tokens of the address (house numbers, plots, pins)
  house_no      first digit-bearing token of the first "street-like" component
  state         canonical state / region code ('' when unresolved)

Nothing here looks anything up externally: every table is a fixed, hand-written
dictionary of well-known abbreviations, plus a word-level transliteration
dictionary learnt from the training ground truth (see translit.py).
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

from unidecode import unidecode

# --------------------------------------------------------------------------- #
# Static tables
# --------------------------------------------------------------------------- #

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA",
    "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "washington dc": "DC", "puerto rico": "PR",
}
US_CODES = set(US_STATES.values())

IN_STATES = {
    "maharashtra": "MH", "delhi": "DL", "new delhi": "DL", "uttar pradesh": "UP", "karnataka": "KA",
    "tamil nadu": "TN", "tamilnadu": "TN", "gujarat": "GJ", "west bengal": "WB", "telangana": "TG",
    "haryana": "HR", "kerala": "KL", "keralam": "KL", "rajasthan": "RJ", "bihar": "BR",
    "madhya pradesh": "MP", "andhra pradesh": "AP", "punjab": "PB", "orissa": "OD", "odisha": "OD",
    "jharkhand": "JH", "chhattisgarh": "CG", "chattisgarh": "CG", "uttarakhand": "UK", "uttaranchal": "UK",
    "himachal pradesh": "HP", "goa": "GA", "assam": "AS", "jammu and kashmir": "JK", "jammu kashmir": "JK",
    "chandigarh": "CH", "puducherry": "PY", "pondicherry": "PY", "tripura": "TR", "manipur": "MN",
    "meghalaya": "ML", "mizoram": "MZ", "nagaland": "NL", "sikkim": "SK", "arunachal pradesh": "AR",
    "dadra and nagar haveli": "DN", "daman and diu": "DD", "andaman and nicobar islands": "AN",
    "ladakh": "LA", "lakshadweep": "LD",
}
IN_CODES = set(IN_STATES.values())
# Regional-script spellings that occur in the data (looked up as whole components).
IN_STATES_NATIVE = {
    "महाराष्ट्र": "MH", "दिल्ली": "DL", "उत्तर प्रदेश": "UP", "ಕರ್ನಾಟಕ": "KA", "தமிழ்நாடு": "TN",
    "ગુજરાત": "GJ", "পশ্চিমবঙ্গ": "WB", "తెలంగాణ": "TG", "हरियाणा": "HR", "राजस्थान": "RJ",
    "കേരളം": "KL", "बिहार": "BR", "मध्य प्रदेश": "MP", "ఆంధ్రప్రదేశ్": "AP", "ਪੰਜਾਬ": "PB",
    "ଓଡ଼ିଶା": "OD", "ओडिशा": "OD", "झारखंड": "JH", "छत्तीसगढ़": "CG", "उत्तराखंड": "UK",
    "हिमाचल प्रदेश": "HP", "गोवा": "GA", "असम": "AS", "অসম": "AS", "ಗೋವಾ": "GA",
}

IN_STATES_NATIVE_SET = set(IN_STATES_NATIVE.keys())
FR_REGIONS = {
    "hauts de france": "HDF", "nouvelle aquitaine": "NAQ", "pays de la loire": "PDL",
    "nord": "HDF", "pas de calais": "HDF", "gironde": "NAQ", "loire atlantique": "PDL",
    "ile de france": "IDF", "auvergne rhone alpes": "ARA", "occitanie": "OCC", "grand est": "GES",
    "bretagne": "BRE", "normandie": "NOR", "bourgogne franche comte": "BFC", "centre val de loire": "CVL",
    "provence alpes cote d azur": "PAC", "corse": "COR",
    # cities present in the data, mapped to their region (used only as a partition key)
    "lille": "HDF", "tourcoing": "HDF", "dunkerque": "HDF", "roubaix": "HDF", "calais": "HDF",
    "lomme": "HDF", "hellemmes lille": "HDF", "st pol sur mer": "HDF",
    "bordeaux": "NAQ", "pessac": "NAQ", "merignac": "NAQ", "la teste de buch": "NAQ",
    "lege cap ferret": "NAQ",
    "nantes": "PDL", "saint nazaire": "PDL", "st nazaire": "PDL", "pornic": "PDL",
    "la baule escoublac": "PDL", "saint herblain": "PDL", "st herblain": "PDL",
}

LEGAL_TOKENS = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt", "private", "llp", "lp", "pc", "plc", "pllc", "sarl", "sas", "sasu", "sci", "eurl",
    "sa", "snc", "gmbh", "ag", "bv", "nv", "the", "shri", "shree", "sri", "ms", "messrs", "mr", "mrs",
    "cie", "societe", "st",
}
# Markers that separate a current name from a former / trade name.
NAME_MARKERS = {"fka", "f/k/a", "nee", "dba", "d/b/a", "formerly", "aka", "a/k/a", "tas"}
DOMAIN_RE = re.compile(r"(?:^|\W)(?:www\.)?([a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*)\.(?:com|in|net|org|co|io|biz|info|us|fr|co\.in|co\.uk)(?:\W|$)")
DOMAIN_SUFFIX_RE = re.compile(r"\.(com|in|net|org|co|io|biz|info|us|fr)\b")

NULL_TOKENS = {"null", "none", "na", "n/a", "nil", "unknown", "nan", "<null>"}

# Canonical (short) forms for street-type words; long -> short so that the
# ambiguous "st" (Saint / Street) collapses to a single token.
ADDR_ABBREV = {
    # US / generic
    "street": "st", "saint": "st", "str": "st", "avenue": "ave", "av": "ave", "aven": "ave",
    "road": "rd", "drive": "dr", "drv": "dr", "lane": "ln", "court": "ct", "circle": "cir",
    "boulevard": "blvd", "boul": "blvd", "bd": "blvd", "highway": "hwy", "parkway": "pkwy",
    "place": "pl", "terrace": "ter", "terr": "ter", "trail": "trl", "square": "sq", "north": "n",
    "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se",
    "southwest": "sw", "ridge": "rdg", "creek": "crk", "mount": "mt", "mountain": "mtn",
    "heights": "hts", "point": "pt", "route": "rte", "expressway": "expy", "extension": "ext",
    "extn": "ext", "junction": "jct", "station": "sta", "center": "ctr", "centre": "ctr",
    "apartment": "apt", "apartments": "apt", "apts": "apt", "suite": "ste", "floor": "fl", "flr": "fl",
    "building": "bldg", "bldng": "bldg", "crossing": "xing", "fort": "ft", "harbor": "hbr",
    "island": "is", "lake": "lk", "meadows": "mdws", "village": "vlg", "valley": "vly", "view": "vw",
    "garden": "gdn", "gardens": "gdns", "grove": "grv", "hill": "hl", "hills": "hls", "hollow": "holw",
    "manor": "mnr", "orchard": "orch", "park": "park", "pike": "pike", "plaza": "plz", "ranch": "rnch",
    "river": "riv", "shore": "shr", "spring": "spg", "springs": "spgs", "summit": "smt", "trace": "trce",
    "turnpike": "tpke", "vista": "vis", "walk": "walk", "way": "way", "loop": "loop", "run": "run",
    "bend": "bnd", "bluff": "blf", "branch": "br", "bridge": "brg", "brook": "brk", "canyon": "cyn",
    "cove": "cv", "crest": "crst", "estates": "ests", "estate": "est", "falls": "fls", "field": "fld",
    "fields": "flds", "forest": "frst", "glen": "gln", "green": "grn", "haven": "hvn", "knoll": "knl",
    "landing": "lndg", "lodge": "ldg", "mall": "mall", "mills": "mls", "mill": "ml", "path": "path",
    "pines": "pnes", "prairie": "pr", "rapids": "rpds", "ridges": "rdgs", "shoals": "shls",
    "station": "sta", "trafficway": "trfy", "tunnel": "tunl", "union": "un", "views": "vws",
    "county": "cnty", "cdp": "", "city": "", "town": "", "township": "twp", "borough": "boro",
    # India
    "nagar": "nagar", "ngr": "nagar", "near": "nr", "opposite": "opp", "opp": "opp", "behind": "bhd",
    "beside": "bsd", "colony": "col", "society": "soc", "sector": "sec", "phase": "ph", "block": "blk",
    "house": "h", "hno": "h", "industrial": "ind", "layout": "lyt", "complex": "cmplx", "tower": "twr",
    "towers": "twr", "mandal": "mdl", "taluk": "tq", "taluka": "tq", "tq": "tq", "district": "dist",
    "dist": "dist", "post": "po", "marg": "marg", "main": "main", "cross": "cross", "gali": "gali",
    "chowk": "chowk", "bazar": "bazar", "bazaar": "bazar", "market": "mkt", "galli": "gali",
    "enclave": "encl", "vihar": "vihar", "puram": "puram", "pura": "pura", "ganj": "ganj",
    "compound": "cmpd", "chambers": "chmbr", "chamber": "chmbr", "bhavan": "bhavan", "bhawan": "bhavan",
    "sadan": "sadan", "niwas": "niwas", "nivas": "niwas", "villa": "villa", "residency": "resdy",
    "residence": "res", "khasra": "kh", "khata": "kh", "survey": "sy", "sy": "sy", "sno": "sy",
    "gram": "gram", "village": "vill", "vill": "vill", "tehsil": "teh", "teh": "teh",
    # France
    "rue": "rue", "r": "rue", "allee": "all", "all": "all", "impasse": "imp", "imp": "imp",
    "chemin": "ch", "che": "ch", "ch": "ch", "cours": "crs", "crs": "crs", "quai": "quai",
    "passage": "pass", "pas": "pass", "batiment": "bat", "bat": "bat", "lieu": "lieu", "dit": "dit",
    "rpt": "rpt", "rond": "rpt", "faubourg": "fbg", "fbg": "fbg", "hameau": "ham", "ham": "ham",
    "esplanade": "esp", "esp": "esp", "promenade": "prom", "prom": "prom", "sentier": "sen",
    "sen": "sen", "villa": "villa", "domaine": "dom", "dom": "dom", "zone": "zone", "za": "za",
    "zi": "zi", "zac": "zac", "cite": "cite", "clos": "clos", "mail": "mail", "parvis": "parvis",
    "ste": "ste", "sainte": "st",
}
# Tokens that carry no identity information (dropped from addr_norm).
ADDR_DROP = {
    "unit", "apt", "ste", "fl", "pmb", "po", "box", "no", "number", "num", "door", "plot", "h",
    "of", "the", "and", "at", "a", "an", "in", "for", "de", "du", "des", "la", "le", "les", "l",
    "d", "et", "et", "sur", "sous", "aux", "au", "lot", "bis", "ter", "cedex", "bp", "cs",
    "old", "new", "ground", "first", "second", "third", "fourth", "fifth", "gf", "ff", "sf",
    "tf", "shop", "office", "flat", "room", "rm", "level", "lvl", "basement", "bsmt", "wing",
    "s/o", "c/o", "w/o", "d/o", "off", "opp", "nr", "bhd", "bsd", "ctr", "bldg", "cmplx", "twr",
    "col", "soc", "sec", "ph", "blk", "encl", "cmpd", "chmbr", "res", "resdy", "ind", "est", "lyt",
    "cnty", "twp", "boro", "dist", "mdl", "tq", "teh", "vill", "po", "vlg", "gram",
    "null", "none", "na", "n/a", "nil", "nan", "unknown",
}
ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")
NUMBER_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12",
    "thirteenth": "13", "fourteenth": "14", "fifteenth": "15", "sixteenth": "16",
    "seventeenth": "17", "eighteenth": "18", "nineteenth": "19", "twentieth": "20",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}
UNIT_PREFIXES = ("unit", "apt", "apartment", "suite", "ste", "fl ", "fl.", "floor", "pmb", "po box",
                 "p o box", "p.o. box", "room", "rm", "flat", "office", "shop",
                 "block", "wing", "gf", "ff", "sf", "tf", "ground floor", "first floor",
                 "second floor", "third floor", "lot", "bp", "cs", "cedex")

DOTTED_RE = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)")
SLASH_MARKER_RE = re.compile(r"\b(?:f/k/a|d/b/a|a/k/a|t/a)\b")
MOJIBAKE_RE = re.compile(r"Â?\x80[\x90-\x9f]|â\x80[\x90-\x9f]|Â")
PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
PUNCT_KEEP_SLASH_RE = re.compile(r"[^a-z0-9/\- ]+")
SPACE_RE = re.compile(r"\s+")
LEADING_ZERO_RE = re.compile(r"\b0+(\d)")
DIGIT_RE = re.compile(r"\d")
NUM_PREFIX_RE = re.compile(r"^(?:n|no|nº|n°|#)\s*(\d)")

# --------------------------------------------------------------------------- #
# Transliteration dictionary (learnt from training data; see translit.py)
# --------------------------------------------------------------------------- #
_TRANSLIT: Dict[str, str] = {}


def load_translit(path: str | None) -> None:
    global _TRANSLIT
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            _TRANSLIT = json.load(fh)


def _script_of(text: str) -> str:
    for ch in text:
        if ch.isalpha():
            try:
                name = unicodedata.name(ch).split()[0]
            except ValueError:
                continue
            if name != "LATIN":
                return name
    return "latin"


def _translit_word(word: str) -> str:
    hit = _TRANSLIT.get(word)
    if hit:
        return hit
    return unidecode(word)


def _fold(text: str) -> str:
    """Unicode fold: learnt word transliteration for non-Latin words, unidecode for the rest."""
    if text.isascii():
        return text
    text = MOJIBAKE_RE.sub(" ", text)
    out = []
    for w in text.split():
        if w.isascii():
            out.append(w)
        else:
            out.append(_translit_word(w))
    return " ".join(out)


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
_L1_RE = re.compile(r"[a-z]{2,}")


def _fix_digit_typos(tok: str) -> str:
    # "internationa1" -> "international", "lnvestors" handled by char n-grams; fix 0/1 inside alpha tokens
    if any(c in tok for c in "01") and sum(c.isalpha() for c in tok) >= 3:
        return tok.replace("1", "l").replace("0", "o")
    return tok


def normalize_name(raw: str) -> Tuple[str, str, str, str, int]:
    """Return (name_norm, name_core, name_compact, script, is_domain)."""
    if raw is None:
        raw = ""
    script = _script_of(raw)
    text = _fold(raw).lower()
    is_domain = 0
    if DOMAIN_SUFFIX_RE.search(text) and " " not in text.strip().replace("www.", ""):
        is_domain = 1
    text = text.replace("www.", " ")
    text = DOMAIN_SUFFIX_RE.sub(" ", text)
    text = DOTTED_RE.sub(lambda m: m.group(1).replace(".", ""), text)
    text = SLASH_MARKER_RE.sub(" ", text)
    text = text.replace("&", " and ").replace("+", " plus ").replace("@", " at ")
    text = text.replace("'", "").replace("’", "")
    text = PUNCT_RE.sub(" ", text)
    toks = text.split()
    toks = [_fix_digit_typos(t) for t in toks]
    toks = [t for t in toks if t not in NAME_MARKERS]
    name_norm = " ".join(toks)
    core = [t for t in toks if t not in LEGAL_TOKENS]
    if not core:
        core = toks
    name_core = " ".join(core)
    return name_norm, name_core, name_core.replace(" ", ""), script, is_domain


# --------------------------------------------------------------------------- #
# Addresses
# --------------------------------------------------------------------------- #
def _resolve_state(comp_raw: str, comp_norm: str, country: str) -> str:
    c = comp_raw.strip()
    if c in IN_STATES_NATIVE:
        return IN_STATES_NATIVE[c]
    n = comp_norm
    if not n:
        return ""
    if n.upper() in US_CODES and len(n) == 2 and country not in ("India",):
        return n.upper()
    if n in US_STATES:
        return US_STATES[n]
    if n in IN_STATES:
        return IN_STATES[n]
    if n.upper() in IN_CODES and len(n) == 2 and country == "India":
        return n.upper()
    if n in FR_REGIONS:
        return FR_REGIONS[n]
    return ""


def _norm_component(comp: str) -> str:
    t = _fold(comp).lower()
    t = t.replace("&", " and ").replace("'", "").replace("’", "")
    t = t.replace("-", " ").replace(".", " ")
    t = PUNCT_RE.sub(" ", t)
    return SPACE_RE.sub(" ", t).strip()


def _norm_token(tok: str) -> str:
    tok = tok.strip("-/")
    if not tok:
        return ""
    m = ORDINAL_RE.match(tok)
    if m:
        return m.group(1)
    if tok in NUMBER_WORDS:
        return NUMBER_WORDS[tok]
    if tok in ADDR_ABBREV:
        return ADDR_ABBREV[tok]
    return tok


def normalize_address(raw: str, country: str) -> Tuple[str, str, str, str]:
    """Return (addr_norm, addr_nums, house_no, state)."""
    if raw is None:
        return "", "", "", ""
    raw = MOJIBAKE_RE.sub(" ", raw)
    comps = [c.strip() for c in raw.split(",")]
    state = ""
    out_tokens: List[str] = []
    nums: List[str] = []
    house = ""
    # pass 1: find state-looking components; prefer an explicit code, else the last full name
    resolved = []
    for i, comp in enumerate(comps):
        cn = _norm_component(comp) if comp else ""
        st = _resolve_state(comp, cn, country) if cn else ""
        if st:
            resolved.append((i, st, len(cn) == 2 or cn in IN_STATES_NATIVE_SET))
    state_idx = -1
    if resolved:
        codes = [r for r in resolved if r[2]]
        pick = codes[-1] if codes else resolved[-1]
        state_idx, state = pick[0], pick[1]
    for i, comp in enumerate(comps):
        if not comp:
            continue
        cn = _norm_component(comp)
        if not cn or cn in NULL_TOKENS:
            continue
        if i == state_idx:
            continue
        # unit-like components: keep their numbers but do not treat as house number
        low = comp.lower().strip().lstrip("#").strip()
        is_unit = low.startswith(UNIT_PREFIXES)
        # tokenise keeping '/' and '-' inside digit tokens (plot numbers like 10-28-2/1/3)
        t2 = _fold(comp).lower().replace("&", " and ").replace("'", "")
        t2 = NUM_PREFIX_RE.sub(r"\1", t2.strip())
        t2 = t2.replace("#", " ")
        t2 = PUNCT_KEEP_SLASH_RE.sub(" ", t2)
        t2 = LEADING_ZERO_RE.sub(r"\1", t2)
        toks = [_norm_token(t) for t in t2.split()]
        comp_nums = []
        for t in toks:
            if not t:
                continue
            if DIGIT_RE.search(t):
                comp_nums.append(t)
                nums.append(t)
            else:
                t = t.replace("-", " ").replace("/", " ")
                for tt in t.split():
                    if tt and tt not in ADDR_DROP and tt not in NULL_TOKENS:
                        out_tokens.append(tt)
        if not house and comp_nums and not is_unit:
            # skip pure postal codes (5-6 digit standalone) as house numbers
            for cnum in comp_nums:
                if not (cnum.isdigit() and len(cnum) >= 5 and len(comp_nums) == 1 and len(toks) == 1):
                    house = cnum
                    break
    # de-duplicate while keeping order
    seen = set()
    dedup = []
    for t in out_tokens:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return " ".join(dedup), " ".join(dict.fromkeys(nums)), house, state


def normalize_record(entity_id: str, name: str, addr: str, country: str) -> dict:
    nn, nc, ncp, script, dom = normalize_name(name)
    an, nums, house, state = normalize_address(addr, country)
    return {
        "entity_id": entity_id, "country": country or "",
        "name_norm": nn, "name_core": nc, "name_compact": ncp, "name_script": script,
        "name_domain": dom, "addr_norm": an, "addr_nums": nums, "house_no": house, "state": state,
        "addr_missing": 1 if not an and not nums else 0,
    }


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #
def _worker(args):
    rows, translit_path = args
    load_translit(translit_path)
    return [normalize_record(*r) for r in rows]


def normalize_file(path: str, out_path: str, translit_path: str | None = None, n_jobs: int = 8,
                   chunk: int = 50_000) -> None:
    import polars as pl
    from multiprocessing import Pool

    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    df = df.fill_null("")
    rows = list(zip(df["entity_id"], df["business_name"], df["business_address"], df["country"]))
    del df
    chunks = ((rows[i:i + chunk], translit_path) for i in range(0, len(rows), chunk))
    parts = []
    n = 0
    with Pool(n_jobs) as pool:
        for res in pool.imap(_worker, chunks, chunksize=1):
            parts.append(pl.DataFrame(res).with_columns(pl.col("name_domain").cast(pl.Int8),
                                                        pl.col("addr_missing").cast(pl.Int8)))
            n += len(res)
    res_df = pl.concat(parts)
    res_df.write_parquet(out_path)
    print(f"normalised {n:,} records from {os.path.basename(path)} -> {out_path}", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--translit", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for p in a.inputs:
        stem = os.path.splitext(os.path.basename(p))[0]
        normalize_file(p, os.path.join(a.out_dir, stem + ".parquet"), a.translit, a.jobs)
