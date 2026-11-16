"""
BLAST → SWISS-MODEL workbench
Run:  streamlit run app.py
"""
import html
import io
import random
import re
import time
from collections import Counter

import pandas as pd
import requests
import streamlit as st
from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis

st.set_page_config(page_title="BLAST to SWISS-MODEL", page_icon="🧬", layout="wide")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
DNA_STRICT = set("ACGTUN")
DNA_IUPAC = set("ACGTUNRYKMSWBDHV")
PROTEIN_CHARS = set("ACDEFGHIKLMNPQRSTVWYBXZUOJ")

NUC_DBS = {
    "core_nt": "core_nt  (core nucleotide collection, faster than nt)",
    "nt": "nt  (full nucleotide collection, slow)",
    "refseq_rna": "refseq_rna  (RefSeq transcripts)",
    "refseq_genomic": "refseq_genomic  (RefSeq genomes)",
}
PROT_DBS = {
    "swissprot": "swissprot  (UniProtKB/Swiss-Prot, curated)",
    "refseq_protein": "refseq_protein  (RefSeq proteins)",
    "nr": "nr  (non-redundant, slow)",
    "pdb": "pdb  (proteins with experimental structures)",
}

PROGRAMS = {
    "blastn": dict(label="blastn: DNA query vs DNA database", query="DNA", dbs=NUC_DBS),
    "blastx": dict(label="blastx: DNA query (translated) vs protein database", query="DNA", dbs=PROT_DBS),
    "blastp": dict(label="blastp: protein query vs protein database", query="Protein", dbs=PROT_DBS),
    "tblastn": dict(label="tblastn: protein query vs DNA database (translated)", query="Protein", dbs=NUC_DBS),
}

E_OPTIONS = [1e-100, 1e-50, 1e-20, 1e-10, 1e-5, 1e-3, 1e-2, 0.1, 1.0, 10.0]
UNIPROT_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)
SM_BASE = "https://swissmodel.expasy.org"

# --------------------------------------------------------------------------
# Look and feel
# --------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
:root{
  --ink:#16222B; --paper:#F4F6F3; --line:#D5DCDD; --muted:#5B6770; --blue:#2338B0;
  --A:#17784A; --C:#1F5FBF; --G:#B26A00; --T:#C0352F;
  --fA:#2FA36B; --fC:#3B7DE0; --fG:#E9A92A; --fT:#E0524B;
}
html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp button{
  font-family:'Instrument Sans', system-ui, sans-serif;
}
.stApp h1, .stApp h2, .stApp h3{ font-family:'Bricolage Grotesque', 'Instrument Sans', sans-serif; letter-spacing:-.01em; }
.stApp code, .stApp pre, .stTextArea textarea{ font-family:'JetBrains Mono', ui-monospace, monospace; }
.stTextArea textarea{ font-size:.85rem; letter-spacing:.03em; }
.block-container{ padding-top:2.2rem; max-width:1250px; }
[data-testid="stSidebar"]{ background:#fff; border-right:1px solid var(--line); }
[data-testid="stHeader"]{ background:transparent; }

/* hero: the one expressive moment */
.hero{ padding:.2rem 0 .6rem; }
.hero h1{ font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:clamp(2.1rem,4.6vw,3.3rem);
  line-height:1.05; margin:0; padding:0; color:var(--ink); }
.hero p{ color:var(--muted); max-width:58ch; margin:.7rem 0 1.1rem; font-size:1.05rem; }
.strip{ display:flex; gap:3px; height:34px; align-items:flex-end; max-width:560px; }
.strip i{ flex:1; height:var(--h); border-radius:2px 2px 0 0; transform-origin:bottom; transform:scaleY(0);
  animation:grow .55s cubic-bezier(.2,.8,.2,1) forwards; animation-delay:calc(var(--i) * 16ms); }
@keyframes grow{ to{ transform:scaleY(1); } }

/* step headings: a real sequence, so numbered */
.step{ display:flex; gap:.9rem; align-items:flex-start; margin:2.4rem 0 .9rem; padding-top:1.1rem; border-top:1px solid var(--line); }
.step .n{ flex:none; width:2rem; height:2rem; border-radius:50%; border:1.5px solid var(--blue); color:var(--blue);
  display:grid; place-items:center; font:600 .95rem 'JetBrains Mono',monospace; }
.step h2{ font-size:1.5rem; margin:0; padding:0; line-height:1.2; }
.step p{ margin:.2rem 0 0; color:var(--muted); font-size:.95rem; }

/* flat number strip */
.facts{ display:flex; flex-wrap:wrap; gap:.5rem 2.4rem; padding:.2rem 0 1rem; }
.facts .v{ font:600 1.5rem 'JetBrains Mono',monospace; color:var(--ink); }
.facts .k{ color:var(--muted); font-size:.85rem; }
.pill{ display:inline-block; padding:.15rem .7rem; border-radius:999px; background:#E3EBFF; color:var(--blue);
  font-weight:600; font-size:.85rem; margin-bottom:.7rem; }

/* sequence viewer */
.seqbox{ font:400 .9rem/1.75 'JetBrains Mono',monospace; letter-spacing:.07em; background:#fff; border:1px solid var(--line);
  border-radius:6px; padding:.7rem 1rem; max-height:190px; overflow:auto; word-break:break-all; }
.seqbox b{ font-weight:600; }
.seqbox .gp10{ display:inline-block; width:.7em; }
.seqbox .more{ color:var(--muted); font-style:italic; letter-spacing:0; }
.nA{color:var(--A)} .nC{color:var(--C)} .nG{color:var(--G)} .nT{color:var(--T)} .nN{color:var(--muted)}
.p-hyd{color:var(--G)} .p-pol{color:var(--A)} .p-pos{color:var(--C)} .p-neg{color:var(--T)} .p-spe{color:#6B4BA8} .p-oth{color:var(--muted)}
.legend{ display:flex; flex-wrap:wrap; gap:.3rem 1.1rem; margin:.5rem 0 .2rem; font-size:.82rem; color:var(--muted); }
.legend i{ display:inline-block; width:.6rem; height:.6rem; border-radius:50%; margin-right:.35rem; }

/* alignment */
pre.aln{ font:400 .82rem/1.55 'JetBrains Mono',monospace; background:#fff; border:1px solid var(--line); border-radius:6px;
  padding:.8rem 1rem; overflow-x:auto; white-space:pre; color:var(--ink); }
pre.aln .lbl{ color:var(--muted); }
pre.aln .mm{ background:#F8D9D6; color:#8A1F1A; }
pre.aln .ps{ background:#F6E7C4; color:#6B4A00; }
pre.aln .gp{ color:#98A3A9; }

/* loading: shown only while NCBI works */
.loader{ display:flex; align-items:center; gap:.45rem; padding:.3rem 0 .5rem; }
.loader b{ width:1.9rem; height:1.9rem; display:grid; place-items:center; border-radius:4px; color:#fff;
  font:600 .9rem 'JetBrains Mono',monospace; animation:hop 1.1s ease-in-out infinite; }
.loader b:nth-child(1){ background:var(--fA); animation-delay:0s; }
.loader b:nth-child(2){ background:var(--fC); animation-delay:.12s; }
.loader b:nth-child(3){ background:var(--fG); animation-delay:.24s; }
.loader b:nth-child(4){ background:var(--fT); animation-delay:.36s; }
.loader span{ margin-left:.7rem; color:var(--muted); font-size:.92rem; }
@keyframes hop{ 0%,100%{ transform:translateY(0); } 40%{ transform:translateY(-9px); } }

/* controls */
button[data-testid="stBaseButton-primary"]{ background:var(--blue); border:0; border-radius:6px; font-weight:600;
  transition:transform .12s ease, box-shadow .12s ease; }
button[data-testid="stBaseButton-primary"]:hover{ background:#1B2E94; transform:translateY(-1px); box-shadow:0 5px 14px rgba(35,56,176,.28); }
button[data-testid="stBaseButton-primary"]:active{ transform:translateY(0); box-shadow:none; }
button[data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button, a[data-testid="stBaseLinkButton-secondary"]{
  border-radius:6px; border-color:var(--line); transition:border-color .12s ease; }
button[data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] button:hover{ border-color:var(--blue); color:var(--blue); }
[data-testid="stExpander"] details{ border:1px solid var(--line); border-radius:8px; background:#fff; }
.stTextArea textarea, .stTextInput input{ border-radius:6px; }

@media (prefers-reduced-motion: reduce){
  .strip i{ animation:none; transform:none; }
  .loader b{ animation:none; }
  button[data-testid="stBaseButton-primary"]{ transition:none; }
}
"""

PROT_CLASS = {
    **{c: "hyd" for c in "AVLIMFWC"}, **{c: "pol" for c in "STNQY"},
    **{c: "pos" for c in "KRH"}, **{c: "neg" for c in "DE"}, **{c: "spe" for c in "GP"},
}
LOADER_HTML = (
    '<div class="loader"><b>A</b><b>C</b><b>G</b><b>T</b>'
    "<span>Waiting for NCBI. Small databases take under a minute, nr and nt can take several.</span></div>"
)


def hero_html() -> str:
    rnd = random.Random(7)  # fixed seed: identical markup on every rerun, so the intro plays once
    bars = "".join(
        f'<i style="--i:{i};--h:{rnd.randint(35, 100)}%;background:var(--f{rnd.choice("ACGT")})"></i>'
        for i in range(56)
    )
    return (
        '<div class="hero"><h1>BLAST to SWISS-MODEL</h1>'
        "<p>Search NCBI with your sequence, read the hits, then take a protein on to structure modelling.</p>"
        f'<div class="strip" aria-hidden="true">{bars}</div></div>'
    )


def step_html(n: int, title: str, hint: str = "") -> str:
    hint_html = f"<p>{hint}</p>" if hint else ""
    return f'<div class="step"><span class="n">{n}</span><div><h2>{title}</h2>{hint_html}</div></div>'


def facts_html(items: dict) -> str:
    cells = "".join(f'<div><div class="v">{v}</div><div class="k">{k}</div></div>' for k, v in items.items())
    return f'<div class="facts">{cells}</div>'


def sequence_html(seq: str, kind: str, limit: int = 3000) -> str:
    parts = []
    for i, ch in enumerate(seq[:limit]):
        if kind == "DNA":
            cls = f"n{ch}" if ch in "ACGT" else "nN"
        else:
            cls = "p-" + PROT_CLASS.get(ch, "oth")
        parts.append(f'<b class="{cls}">{ch}</b>')
        if (i + 1) % 10 == 0:
            parts.append('<span class="gp10"></span>')
    more = f'<span class="more"> ... {len(seq) - limit} more not shown</span>' if len(seq) > limit else ""
    return f'<div class="seqbox">{"".join(parts)}{more}</div>'


def legend_html(kind: str) -> str:
    if kind == "DNA":
        items = [("var(--A)", "A"), ("var(--C)", "C"), ("var(--G)", "G"), ("var(--T)", "T / U")]
    else:
        items = [("var(--G)", "hydrophobic"), ("var(--A)", "polar"), ("var(--C)", "positive"),
                 ("var(--T)", "negative"), ("#6B4BA8", "Gly, Pro")]
    return '<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{t}</span>' for c, t in items) + "</div>"


def alignment_html(q: str, m: str, s: str, width: int = 60) -> str:
    """Query / match / subject lines with mismatches, positives and gaps highlighted."""
    def span(ch, cls):
        ch = html.escape(ch)
        return f'<span class="{cls}">{ch}</span>' if cls else ch

    blocks = []
    for i in range(0, len(q), width):
        qa, ma, sa = q[i:i + width], m[i:i + width], s[i:i + width]
        ql, sl = [], []
        for qc, mc, sc in zip(qa, ma, sa):
            if qc == "-" or sc == "-":
                cls = "gp"
            elif mc == " ":
                cls = "mm"
            elif mc == "+":
                cls = "ps"
            else:
                cls = ""
            ql.append(span(qc, cls))
            sl.append(span(sc, cls))
        blocks.append(
            f'<span class="lbl">Query  </span>{"".join(ql)}\n'
            f'<span class="lbl">       </span>{html.escape(ma)}\n'
            f'<span class="lbl">Sbjct  </span>{"".join(sl)}\n'
        )
    return '<pre class="aln">' + "\n".join(blocks) + "</pre>"



# --------------------------------------------------------------------------
# Sequence helpers
# --------------------------------------------------------------------------
def clean_sequence(raw: str):
    """Return (sequence, number_of_records, first_header). Accepts FASTA or raw text."""
    records, current, headers = [], [], []
    for line in raw.strip().splitlines():
        if line.startswith(">"):
            if current:
                records.append("".join(current))
                current = []
            headers.append(line[1:].strip())
        else:
            current.append(line)
    if current:
        records.append("".join(current))
    seq = records[0] if records else ""
    seq = re.sub(r"[\s\d\-]", "", seq).upper().rstrip("*")
    return seq, len(records), (headers[0] if headers else "")


def detect_type(seq: str):
    if not seq:
        return None
    chars = set(seq)
    if chars <= DNA_STRICT:
        return "DNA"
    if chars <= PROTEIN_CHARS:
        return "Protein"
    return None


def invalid_chars(seq: str, seq_type: str):
    allowed = DNA_IUPAC if seq_type == "DNA" else PROTEIN_CHARS
    return sorted(set(seq) - allowed)


def sequence_stats(seq: str, seq_type: str):
    """Return (dict of headline numbers, Series of composition)."""
    if seq_type == "DNA":
        counts = Counter(seq)
        stats = {"Length (nt)": len(seq), "GC content": f"{gc_fraction(seq) * 100:.1f}%"}
        comp = pd.Series({b: counts.get(b, 0) for b in "ACGTN" if counts.get(b, 0)})
    else:
        counts = Counter(seq)
        stats = {"Length (aa)": len(seq)}
        try:
            pa = ProteinAnalysis(seq)
            stats["Mol. weight"] = f"{pa.molecular_weight() / 1000:.1f} kDa"
            stats["Isoelectric point"] = f"{pa.isoelectric_point():.2f}"
        except Exception:
            stats["Mol. weight"] = "n/a (ambiguous residues)"
        comp = pd.Series(dict(sorted(counts.items())))
    return stats, comp


# --------------------------------------------------------------------------
# BLAST
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_blast(program, db, seq, expect, hitlist, entrez_query):
    """Submit to NCBI and wait (Biopython handles job id + polling). Returns XML text."""
    handle = NCBIWWW.qblast(
        program,
        db,
        seq,
        expect=expect,
        hitlist_size=hitlist,
        entrez_query=entrez_query or "(none)",
        format_type="XML",
    )
    xml = handle.read()
    handle.close()
    return xml


def parse_blast_xml(xml_text: str, db: str) -> pd.DataFrame:
    record = NCBIXML.read(io.StringIO(xml_text))
    rows = []
    for rank, aln in enumerate(record.alignments, start=1):
        hsp = aln.hsps[0]  # best HSP per subject
        gaps = hsp.gaps if isinstance(hsp.gaps, int) else 0
        q_span = abs(hsp.query_end - hsp.query_start) + 1
        rows.append(
            {
                "Database": db,
                "Rank": rank,
                "Accession": aln.accession,
                "Description": aln.hit_def,
                "% Identity": round(100 * hsp.identities / hsp.align_length, 1),
                "Query cover %": round(min(100.0, 100 * q_span / record.query_letters), 1),
                "Alignment length": hsp.align_length,
                "E-value": hsp.expect,
                "Bit score": round(hsp.bits, 1),
                "Gaps": gaps,
                "Subject length": aln.length,
                "_q": hsp.query,
                "_m": hsp.match,
                "_s": hsp.sbjct,
            }
        )
    return pd.DataFrame(rows)


def interpret_hit(row, program: str) -> str:
    e, ident, cov = row["E-value"], row["% Identity"], row["Query cover %"]
    if e <= 1e-50:
        e_txt = "extremely significant: almost certainly the same gene/protein or a very close relative"
    elif e <= 1e-10:
        e_txt = "highly significant: a clear homolog"
    elif e <= 1e-3:
        e_txt = "significant: probably homologous, likely sharing a domain or family"
    elif e <= 0.05:
        e_txt = "borderline: treat with caution and check the alignment"
    else:
        e_txt = "weak: could easily be a chance match"

    if program == "blastn":
        i_txt = (
            "same species or gene" if ident >= 95
            else "same gene family or a close species" if ident >= 80
            else "distant, possibly only conserved regions"
        )
    else:
        i_txt = (
            "very likely the same function and fold" if ident >= 70
            else "likely a similar fold (above ~40% is a good sign)" if ident >= 40
            else "twilight zone: the fold may be shared, but check it" if ident >= 25
            else "low: structure transfer is unreliable"
        )
    c_txt = "covers most of your query" if cov >= 80 else "covers only part of your query (a domain-level match?)"
    return (
        f"**{row['Accession']}**: E-value {e:.2g} is {e_txt}. "
        f"{ident}% identity means {i_txt}. This hit {c_txt} ({cov}%)."
    )


# --------------------------------------------------------------------------
# Getting a protein sequence for a hit
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_protein_fasta(accession: str, email: str):
    Entrez.email = email
    handle = Entrez.efetch(db="protein", id=accession, rettype="fasta", retmode="text")
    rec = SeqIO.read(handle, "fasta")
    handle.close()
    return rec.description, str(rec.seq)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_cds_protein(accession: str, email: str):
    """For nucleotide hits: translation of the longest annotated CDS."""
    Entrez.email = email
    handle = Entrez.efetch(db="nuccore", id=accession, rettype="gb", retmode="text")
    rec = SeqIO.read(handle, "gb")
    handle.close()
    cds = [
        (f.qualifiers.get("product", [rec.description])[0], f.qualifiers["translation"][0])
        for f in rec.features
        if f.type == "CDS" and "translation" in f.qualifiers
    ]
    if not cds:
        raise ValueError("No annotated CDS translation in this record.")
    return max(cds, key=lambda x: len(x[1]))


def protein_for_hit(hit, program: str, email: str):
    acc = hit["Accession"]
    if program in ("blastp", "blastx"):
        return fetch_protein_fasta(acc, email)
    if hit["Subject length"] > 200_000:
        raise ValueError(
            "This hit is a large genomic record. Run blastx/blastp against a protein database "
            "to get a protein hit you can model."
        )
    return fetch_cds_protein(acc, email)


# --------------------------------------------------------------------------
# SWISS-MODEL
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def sm_repository_lookup(uniprot_ac: str):
    r = requests.get(f"{SM_BASE}/repository/uniprot/{uniprot_ac}.json", timeout=30)
    r.raise_for_status()
    structures = r.json().get("result", {}).get("structures", [])
    cols = ["provider", "template", "identity", "coverage", "from", "to", "qmean", "method", "coordinates"]
    return pd.DataFrame([{c: s.get(c) for c in cols} for s in structures])


def sm_submit_and_wait(seq: str, token: str, title: str, placeholder):
    headers = {"Authorization": f"Token {token}"}
    r = requests.post(
        f"{SM_BASE}/automodel",
        headers=headers,
        json={"target_sequences": [seq], "project_title": title},
        timeout=60,
    )
    r.raise_for_status()
    project_id = r.json()["project_id"]
    for _ in range(90):  # up to ~15 min
        time.sleep(10)
        s = requests.get(f"{SM_BASE}/project/{project_id}/models/summary/", headers=headers, timeout=60)
        s.raise_for_status()
        data = s.json()
        status = data.get("status", "UNKNOWN")
        placeholder.write(f"Project `{project_id}`: {status}")
        if status in ("COMPLETED", "FAILED"):
            return project_id, data
    return project_id, {"status": "TIMEOUT"}


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
with st.sidebar:
    st.header("Settings")
    email = st.text_input(
        "Your email (NCBI asks for it)",
        value="you@example.com",
        help="Sent to NCBI Entrez when fetching full sequences. Use a real address.",
    )
    expect = st.select_slider("BLAST E-value cutoff", options=E_OPTIONS, value=10.0, format_func=lambda x: f"{x:g}")
    hitlist = st.slider("Hits per database", 5, 100, 25, step=5)
    organism = st.text_input("Limit to organism (optional)", placeholder="Homo sapiens")
    st.caption("NCBI public BLAST is rate-limited. Large databases (nr, nt) can take several minutes.")

entrez_query = f'"{organism.strip()}"[Organism]' if organism.strip() else ""

# --------------------------------------------------------------------------
# Step 1 - sequence
# --------------------------------------------------------------------------
st.markdown(hero_html(), unsafe_allow_html=True)

st.markdown(step_html(1, "Your sequence", "Paste DNA or protein, as FASTA or plain text."), unsafe_allow_html=True)
left, right = st.columns([3, 1])
with left:
    raw = st.text_area("Paste a DNA or protein sequence (FASTA or plain)", height=140, placeholder=">my_query\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF...")
with right:
    upload = st.file_uploader("or upload FASTA", type=["fa", "fasta", "txt", "faa", "fna"])
    type_choice = st.selectbox("Sequence type", ["Auto-detect", "DNA", "Protein"])
if upload is not None:
    raw = upload.read().decode("utf-8", errors="ignore")

seq, n_records, header = clean_sequence(raw or "")
seq_type, ready = None, False
if seq:
    if n_records > 1:
        st.warning(f"{n_records} sequences found. Only the first one is used.")
    seq_type = detect_type(seq) if type_choice == "Auto-detect" else type_choice
    if seq_type is None:
        st.error(f"Could not recognise this as DNA or protein. Unexpected characters: {', '.join(invalid_chars(seq, 'Protein'))}")
    else:
        bad = invalid_chars(seq, seq_type)
        if bad:
            st.error(f"Invalid {seq_type} characters: {', '.join(bad)}")
        else:
            ready = True
            if seq_type == "DNA":
                seq = seq.replace("U", "T")
            stats, comp = sequence_stats(seq, seq_type)
            pill = f"{seq_type} detected" if type_choice == "Auto-detect" else f"Treating as {seq_type}"
            st.markdown(f'<span class="pill">{pill}</span>', unsafe_allow_html=True)
            st.markdown(facts_html(stats), unsafe_allow_html=True)
            st.markdown(sequence_html(seq, seq_type) + legend_html(seq_type), unsafe_allow_html=True)
            with st.expander("Composition"):
                st.bar_chart(comp, color="#2338B0")

# --------------------------------------------------------------------------
# Step 2 - program and databases
# --------------------------------------------------------------------------
st.markdown(step_html(2, "BLAST program and databases", "Only programs that fit your sequence type are offered."), unsafe_allow_html=True)
if not ready:
    st.info("Add a valid sequence above to continue.")
else:
    options = [p for p, v in PROGRAMS.items() if v["query"] == seq_type]
    program = st.radio("Program", options, format_func=lambda p: PROGRAMS[p]["label"], horizontal=False)
    db_map = PROGRAMS[program]["dbs"]
    dbs = st.multiselect(
        "Databases (each one is a separate NCBI job)",
        list(db_map),
        default=[next(iter(db_map))],
        format_func=lambda d: db_map[d],
    )
    run = st.button("Run BLAST", type="primary", disabled=not dbs)

    if run:
        frames = []
        for db in dbs:
            with st.status(f"{program} against {db}", expanded=True) as status:
                loader = st.empty()
                loader.markdown(LOADER_HTML, unsafe_allow_html=True)
                try:
                    xml = run_blast(program, db, seq, expect, hitlist, entrez_query)
                    part = parse_blast_xml(xml, db)
                    frames.append(part)
                    status.update(label=f"{db}: {len(part)} hits", state="complete")
                except Exception as exc:  # network, NCBI busy, parse errors
                    status.update(label=f"{db}: failed ({exc})", state="error")
                finally:
                    loader.empty()
        st.session_state["results"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        st.session_state["meta"] = {"program": program, "seq": seq, "type": seq_type}

# --------------------------------------------------------------------------
# Step 3 - results
# --------------------------------------------------------------------------
results = st.session_state.get("results")
meta = st.session_state.get("meta", {})
selected = pd.DataFrame()

if results is not None:
    st.markdown(step_html(3, "Results", "Filter, read the explanations on the right, and click rows to pick hits."), unsafe_allow_html=True)
    if results.empty:
        st.warning("No hits. Try a larger E-value cutoff, another database, or remove the organism filter.")
    else:
        f1, f2, f3 = st.columns(3)
        max_e = f1.select_slider("Max E-value", options=E_OPTIONS, value=1e-2, format_func=lambda x: f"{x:g}")
        min_id = f2.slider("Min % identity", 0, 100, 0)
        min_cov = f3.slider("Min query cover %", 0, 100, 0)
        view = results[
            (results["E-value"] <= max_e) & (results["% Identity"] >= min_id) & (results["Query cover %"] >= min_cov)
        ].reset_index(drop=True)
        public_cols = [c for c in view.columns if not c.startswith("_")]
        if len(view):
            best_e = view["E-value"].min()
            st.markdown(
                facts_html({
                    "Hits shown": len(view),
                    "Best E-value": "0" if best_e == 0 else f"{best_e:.0e}",
                    "Best identity": f"{view['% Identity'].max():.1f}%",
                    "Best query cover": f"{view['Query cover %'].max():.0f}%",
                }),
                unsafe_allow_html=True,
            )
        st.caption(f"{len(view)} of {len(results)} hits shown. Click rows to select hits for SWISS-MODEL.")

        table_col, help_col = st.columns([3, 1])
        with table_col:
            event = st.dataframe(
                view[public_cols],
                hide_index=True,
                use_container_width=True,
                on_select="rerun",
                selection_mode="multi-row",
                column_config={
                    "% Identity": st.column_config.ProgressColumn("% Identity", min_value=0, max_value=100, format="%.1f%%"),
                    "Query cover %": st.column_config.ProgressColumn("Query cover", min_value=0, max_value=100, format="%.1f%%"),
                    "E-value": st.column_config.NumberColumn("E-value", format="%.2e"),
                    "Description": st.column_config.TextColumn("Description", width="large"),
                },
            )
            selected = view.iloc[event.selection.rows] if event.selection.rows else pd.DataFrame()

            d1, d2, _ = st.columns([1, 1, 4])
            d1.download_button("Download CSV", view[public_cols].to_csv(index=False), "blast_hits.csv", "text/csv")
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as xl:
                view[public_cols].to_excel(xl, index=False, sheet_name="BLAST hits")
            d2.download_button(
                "Download Excel", buf.getvalue(), "blast_hits.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with help_col:
            if len(view):
                st.markdown("**Top hit in plain words**")
                st.markdown(interpret_hit(view.iloc[0], meta.get("program", "blastp")))
            with st.expander("What is an E-value?"):
                st.markdown(
                    "The number of hits this good you would expect by chance in a database this size. "
                    "**Smaller is better.** 0 means essentially impossible by chance. "
                    "Below 1e-5 is usually meaningful; above 0.05 is shaky."
                )
            with st.expander("What does a good hit look like?"):
                st.markdown(
                    "- Low E-value (1e-10 or smaller)\n"
                    "- High query cover (80%+), so the hit matches most of your sequence, not a small piece\n"
                    "- Few gaps\n"
                    "- For modelling: 30-40%+ identity over the region you care about"
                )
            with st.expander("% identity, cover and bit score"):
                st.markdown(
                    "**% identity**: share of aligned positions that are exactly the same.\n\n"
                    "**Query cover**: how much of your sequence is in the alignment.\n\n"
                    "**Bit score**: alignment quality, independent of database size. **Higher is better.**\n\n"
                    "**Gaps**: insertions or deletions needed to line the sequences up."
                )

# --------------------------------------------------------------------------
# Step 4 - structure
# --------------------------------------------------------------------------
if results is not None and not results.empty:
    st.markdown(step_html(4, "Protein structure with SWISS-MODEL", "Selected hits appear here with their sequence and modelling options."), unsafe_allow_html=True)
    if selected.empty:
        st.info("Select one or more hits in the table above.")
    program = meta.get("program", "blastp")
    for _, hit in selected.iterrows():
        label = f"{hit['Accession']}  {hit['Description'][:90]}"
        with st.expander(label, expanded=len(selected) == 1):
            with st.expander("Alignment"):
                st.markdown(alignment_html(hit["_q"], hit["_m"], hit["_s"]), unsafe_allow_html=True)

            try:
                title, prot = protein_for_hit(hit, program, email)
            except Exception as exc:
                st.error(f"Could not get a protein sequence for this hit: {exc}")
                continue

            st.markdown(f"**Protein sequence** ({len(prot)} aa)")
            fasta = f">{hit['Accession']} {title}\n{prot}"
            st.code(fasta, language=None)  # has a copy button

            acc = hit["Accession"].split(".")[0]
            link_cols = st.columns(4)
            link_cols[0].link_button("Open SWISS-MODEL", f"{SM_BASE}/interactive")
            if hit["Database"] == "pdb":
                link_cols[1].link_button("View in RCSB PDB", f"https://www.rcsb.org/structure/{acc.split('_')[0]}")
            if UNIPROT_RE.match(acc):
                link_cols[2].link_button("AlphaFold DB", f"https://alphafold.ebi.ac.uk/entry/{acc}")
                link_cols[3].link_button("UniProt", f"https://www.uniprot.org/uniprotkb/{acc}")
            st.caption("On SWISS-MODEL, click Start Modelling and paste the sequence above.")

            if UNIPROT_RE.match(acc):
                with st.expander("Existing SWISS-MODEL repository models"):
                    try:
                        repo = sm_repository_lookup(acc)
                        if repo.empty:
                            st.write("No pre-computed models. Build one with the sequence above.")
                        else:
                            st.dataframe(repo, hide_index=True, use_container_width=True)
                    except Exception as exc:
                        st.write(f"Lookup failed: {exc}")

            with st.expander("Build a model from here (SWISS-MODEL API token needed)"):
                token = st.text_input("API token", type="password", key=f"tok_{hit['Database']}_{acc}",
                                      help="From your SWISS-MODEL account page.")
                if len(prot) < 20:
                    st.warning("SWISS-MODEL needs at least 20 residues.")
                elif st.button("Submit job", key=f"sm_{hit['Database']}_{acc}", disabled=not token):
                    ph = st.empty()
                    try:
                        pid, data = sm_submit_and_wait(prot, token, f"{acc} from BLAST app", ph)
                        st.write(f"Finished with status: {data.get('status')}")
                        models = data.get("models") or []
                        if models:
                            st.dataframe(pd.json_normalize(models), hide_index=True, use_container_width=True)
                        st.link_button("Open project on SWISS-MODEL", f"{SM_BASE}/project/{pid}/models/")
                    except Exception as exc:
                        st.error(f"SWISS-MODEL request failed: {exc}")
