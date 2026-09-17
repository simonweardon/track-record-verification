"""The site speaks plainly: no implementation details, no filing jargon, no build credits.

A reader should see what each number is and how it was calculated, never which language or
library produced it. These tests strip the tags from every rendered page and look for the
words that must not appear in visible text."""
import html as H
import re
from pathlib import Path

import pytest

from trackrecord import research_pages as R
from trackrecord import serve as S
from trackrecord.areas import home_html, area_html

ROOT = Path(__file__).resolve().parents[1]

BANNED = [r"\bClaude\b", r"\bPython\b", r"\bin R\b", r"\bR (?:implementation|twin|script|side)\b", r"data\.table", r"Rglpk", r"xgboost", r"XGBoost",
          r"HiGHS", r"scipy", r"statsmodels", r"pandas", r"numpy", r"13F", r"\bclones?\b", r"python -m", r"Rscript", r"pair programmer"]


def visible(doc: str) -> str:
    doc = re.sub(r"<(script|style)\b.*?</\1>", " ", doc, flags=re.S | re.I)
    return H.unescape(re.sub(r"<[^>]+>", " ", doc))


def offenders(doc: str) -> list[str]:
    text = visible(doc)
    out = []
    for pat in BANNED:
        for m in re.finditer(pat, text):
            out.append(text[max(0, m.start() - 40):m.end() + 40].replace("\n", " "))
    return out


PAGES = {"/": lambda: home_html(S.page, 1, 1), "/active": lambda: area_html(S.page, "active"), "/external": S.directory_html,
         "/research": R.research_index_html, "/research/alpha-lab": R.alphalab_html, "/research/risk-model": R.riskmodel_html,
         "/research/construction": R.construction_html, "/research/r-verify": R.rverify_html, "/research/decay": R.decay_html,
         "/research/fund-of-funds": R.fof_html, "/research/13f-signals": R.signals13f_html}


@pytest.mark.parametrize("path", list(PAGES))
def test_pages_use_plain_language(path):
    doc = PAGES[path]()
    if doc is None:
        pytest.skip(f"{path} not built")
    assert not offenders(doc), offenders(doc)[:8]


def test_memo_and_dashboard_use_plain_language():
    out = ROOT / "output" / "funds"
    built = sorted(d for d in out.glob("*") if (d / "dashboard.html").exists()) if out.exists() else []
    if not built:
        pytest.skip("no built manager")
    from trackrecord.memo import build_memo
    from trackrecord.dashboard import build_dashboard
    d = built[0]
    assert not offenders(build_memo(d, ROOT / "data" / "funds" / d.name)), offenders(build_memo(d, ROOT / "data" / "funds" / d.name))[:8]
    # the dashboard as it would be written today, not the file on disk
    from trackrecord import dashboard as D
    doc = D.render_dashboard(d) if hasattr(D, "render_dashboard") else (d / "dashboard.html").read_text()
    assert not offenders(doc), offenders(doc)[:8]


def test_readme_has_no_build_credits():
    text = (ROOT / "README.md").read_text()
    for pat in (r"\bClaude\b", r"pair programmer", r"AI assist"):
        assert not re.search(pat, text), pat
