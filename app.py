"""GoodLaw - Streamlit UI.

Paste a brief, get a four-layer verification report with citations color-
coded by severity. Runs the same analyze() pipeline as the CLI.
"""

from __future__ import annotations

import json
import os
import time

import streamlit as st
from dotenv import load_dotenv

from goodlaw.analyze import analyze

load_dotenv()

st.set_page_config(
    page_title="GoodLaw - citation verifier",
    page_icon="⚖️",
    layout="wide",
)

# --- Sidebar --------------------------------------------------------------

with st.sidebar:
    st.title("⚖️ GoodLaw")
    st.caption("Four-layer verifier for AI-generated legal citations.")
    st.caption("For solo attorneys running a pre-flight check on AI-drafted briefs.")
    st.markdown(
        """
**Layers**
1. **Existence** — is the citation real? (CourtListener)
2. **Quote fidelity** — does the quoted text match the opinion? (rapidfuzz)
3. **Proposition support** — does the passage support the claim? (DeBERTa NLI)
4. **Treatment signal** — has the case been overruled/criticized? (keyword scan of citing opinions)

Runs entirely on open data. No paid APIs.
    """
    )
    st.markdown("---")
    st.caption(
        "Benchmark: **F1 = 0.968**, recall = 1.00 on 30 labeled citations "
        "(15 real, 15 fabricated)."
    )
    token = os.getenv("COURTLISTENER_TOKEN")
    if not token:
        st.error("COURTLISTENER_TOKEN missing")

# --- Main -----------------------------------------------------------------

st.title("Legal citation verifier")
st.write(
    "Paste a brief, memo, or any block of text. GoodLaw extracts every "
    "citation and checks it against the CourtListener corpus, then verifies "
    "any quoted text and the proposition each citation supports."
)

DEFAULT_SAMPLE = """Plaintiff respectfully submits that a carrier's duty of inquiry attaches at
the moment of consignment, not delivery. Thompson v. Delaware Logistics
Corp., 892 F.3d 1142, 1149 (9th Cir. 2019). As the Thompson court explained,
"a carrier's duty of inquiry attaches at the moment of consignment, not
delivery." Id. at 1151.

This principle rests on well-established constitutional foundations. The
right at issue is fundamental. See Obergefell v. Hodges, 576 U.S. 644, 663
(2015). This Court has long recognized that separate facilities are
inherently unequal. Brown v. Board of Education, 347 U.S. 483 (1954). Id.
at 495.

The Brown Court concluded that "in the field of public education the
doctrine of 'separate but equal' has no place." 347 U.S. at 495.

Moreover, a suspect must be advised of his rights prior to custodial
interrogation. Miranda v. Arizona, 384 U.S. 436, 444 (1966). The majority
in Miranda emphasized that "the modern practice of in-custody interrogation
is psychologically rather than physically oriented." Miranda, 384 U.S. at
448.

See also Obergefell, supra, at 670; 42 U.S.C. § 1983."""

MATA_AVIANCA_SAMPLE = """This case addresses whether the statute of limitations was tolled during
the pendency of bankruptcy proceedings involving the airline carrier.
Several courts have addressed nearly identical facts.

The Eleventh Circuit held that limitations periods are tolled during
Chapter 11 proceedings even absent a formal stay. Varghese v. China
Southern Airlines Co., 925 F.3d 1339 (11th Cir. 2019). The court reasoned
that "the automatic stay provision serves the salutary purpose of
preserving the debtor's estate."

Similarly, the Fifth Circuit found tolling appropriate where the passenger
lacked knowledge of the pending reorganization. Shaboon v. EgyptAir, 2013
IL App (1st) 111279-U (Ill. App. Ct. 2013). See also Petersen v. Iran Air,
905 F. Supp. 2d 121 (D.D.C. 2012).

The Second Circuit has adopted a similar approach in Zicherman v. Korean
Air Lines Co., 516 F.3d 1237 (11th Cir. 2008), where the court held that
"individual passengers retain a private right of action regardless of
carrier bankruptcy status." Id. at 1245.

Compare Miller v. United Airlines, Inc., 174 F.3d 366 (5th Cir. 1999),
and Estate of Durden v. KLM Royal Dutch Airlines, 2017 WL 2418825 (S.D.
Fla. 2017), where the courts reached the opposite conclusion.

These decisions establish that the tolling doctrine applies here."""

SAMPLES = {
    "Mixed brief (real + one fake citation)": DEFAULT_SAMPLE,
    "Mata v. Avianca (real ChatGPT-fabricated filing, 2023)": MATA_AVIANCA_SAMPLE,
    "Empty - paste your own": "",
}

sample_choice = st.selectbox(
    "Load a sample or paste your own",
    list(SAMPLES.keys()),
    index=0,
)

col_input, col_run = st.columns([5, 1])
with col_input:
    text = st.text_area(
        "Document text",
        value=SAMPLES[sample_choice],
        height=280,
        label_visibility="collapsed",
        key=f"input_{sample_choice}",
    )
with col_run:
    st.write("")
    st.write("")
    run = st.button("Verify", type="primary", use_container_width=True)
    st.caption(f"{len(text):,} chars")

# --- Run ------------------------------------------------------------------

BADGE = {
    "green": ("✅", "#065f46", "#d1fae5"),
    "yellow": ("⚠️", "#92400e", "#fef3c7"),
    "red": ("❌", "#991b1b", "#fee2e2"),
    "gray": ("➖", "#374151", "#e5e7eb"),
}


def badge(verdict: str, label: str = "") -> str:
    icon, fg, bg = BADGE.get(verdict, BADGE["gray"])
    text = f"{icon} {label or verdict.upper()}"
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:12px;font-weight:600;font-size:13px;">{text}</span>'
    )


if run and text.strip() and token:
    with st.spinner("Running four-layer verification..."):
        t0 = time.time()
        report = analyze(text, token, doc_path="pasted")
    st.success(f"Done in {report.runtime_seconds}s")

    # Overall verdict banner
    verdict = report.overall_verdict
    st.markdown(
        f"### Overall: {badge(verdict, verdict.upper())}",
        unsafe_allow_html=True,
    )

    # Layer summary bar
    cols = st.columns(4)
    for col, (layer, counts) in zip(cols, report.summary_counts.items()):
        with col:
            st.markdown(f"**{layer.title()}**")
            row = ""
            for v in ("green", "yellow", "red", "gray"):
                if counts[v]:
                    row += badge(v, str(counts[v])) + " "
            st.markdown(row or "—", unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            f"Citations ({len(report.citations)})",
            f"Quotes ({len(report.quotes)})",
            f"Claims ({len(report.claims)})",
            f"Treatment ({len(report.treatment)})",
        ]
    )

    with tab1:
        for c in report.citations:
            st.markdown(
                f"{badge(c['verdict'])} **`{c['text']}`** " f"— {c.get('status', '')}",
                unsafe_allow_html=True,
            )
            if c.get("canonical_name"):
                st.caption(f"→ {c['canonical_name']} ({c.get('canonical_date','')})")
            if c.get("name_mismatch"):
                st.caption(
                    f"⚠️ document says **{c.get('parsed_case_name')}**, "
                    f"corpus says **{c.get('canonical_name')}**"
                )
            st.write("")

    with tab2:
        if not report.quotes:
            st.info("No quoted text found in the document.")
        for q in report.quotes:
            st.markdown(
                f"{badge(q['verdict'])} \"{q['quote'][:200]}"
                f"{'...' if len(q['quote']) > 200 else ''}\"",
                unsafe_allow_html=True,
            )
            st.caption(
                f"attributed to `{q.get('attributed_to','?')}` — {q.get('reason','')}"
            )
            st.write("")

    with tab3:
        if not report.claims:
            st.info("No claims analyzed.")
        for c in report.claims:
            st.markdown(
                f"{badge(c['verdict'])} _{c['claim_sentence'][:200]}"
                f"{'...' if len(c['claim_sentence']) > 200 else ''}_",
                unsafe_allow_html=True,
            )
            st.caption(
                f"cite `{c['citation_text']}` · opinion "
                f"{c.get('retrieved_from_opinion','?')} · {c['reason']}"
            )
            with st.expander("Retrieved passage"):
                st.write(c.get("retrieved_passage", ""))
            st.write("")

    with tab4:
        if not report.treatment:
            st.info("No cases scanned for treatment.")
        for t in report.treatment:
            st.markdown(
                f"{badge(t['verdict'])} **`{t['citation_text']}`** — {t['reason']}",
                unsafe_allow_html=True,
            )
            for h in t.get("negative_hits", [])[:3]:
                st.caption(
                    f"'{h['term']}' in _{h.get('citing_case','?')}_ "
                    f"({h.get('citing_date','?')})"
                )
            st.write("")

    st.markdown("---")
    st.download_button(
        "Download full report (JSON)",
        data=json.dumps(
            {
                "overall_verdict": report.overall_verdict,
                "summary_counts": report.summary_counts,
                "citations": report.citations,
                "quotes": report.quotes,
                "claims": report.claims,
                "treatment": report.treatment,
                "runtime_seconds": report.runtime_seconds,
            },
            indent=2,
            default=str,
        ),
        file_name="goodlaw_report.json",
        mime="application/json",
    )

elif run and not text.strip():
    st.warning("Paste some text first.")
