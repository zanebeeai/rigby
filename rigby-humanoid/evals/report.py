from __future__ import annotations

import html
import json
from typing import Any


def render_report(report: dict[str, Any], certification: dict[str, Any] | None = None) -> str:
    cards: list[str] = []
    for gate in report["gates"]:
        status = gate["status"]
        failures = "".join(f"<li>{html.escape(str(item))}</li>" for item in gate.get("failures", []))
        cards.append(
            f"<section class='gate {status}'><h2>{html.escape(gate['gate'])} "
            f"<span>{html.escape(status.upper())}</span></h2>"
            f"<p>{html.escape(gate['summary'])}</p>"
            f"<details><summary>Measured evidence</summary><pre>"
            f"{html.escape(json.dumps(gate.get('measured', {}), indent=2, sort_keys=True))}</pre></details>"
            f"{f'<ul>{failures}</ul>' if failures else ''}</section>"
        )
    certification_html = ""
    if certification is not None:
        certification_html = (
            f"<p class='cert'>Certification: <strong>"
            f"{'PASS' if certification.get('certified') else 'NOT YET CERTIFIED'}</strong> - "
            f"{certification.get('consecutive_passes', 0)} consecutive passing run(s).</p>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Rigby acceptance report</title><style>
body{{font:15px/1.45 system-ui;margin:0;background:#10141c;color:#eaf0fa}}main{{max-width:980px;margin:auto;padding:32px}}
.gate{{background:#1a2230;border-left:6px solid #687386;border-radius:8px;padding:14px 18px;margin:14px 0}}
.gate.pass{{border-color:#36c57a}}.gate.fail,.gate.error{{border-color:#f05d5e}}.gate.unverified{{border-color:#f4bf4f}}
h1,h2{{margin:.2em 0}}h2 span{{font-size:.65em;float:right}}pre{{white-space:pre-wrap;overflow:auto;background:#0c1017;padding:12px}}
.cert{{font-size:1.1em;background:#243149;padding:12px;border-radius:8px}}a{{color:#85bdff}}
</style></head><body><main><h1>Rigby acceptance report</h1>
<p>Run {html.escape(str(report.get('run_id', 'pending')))} | {html.escape(report['completed_at'])}</p>
{certification_html}{''.join(cards)}</main></body></html>"""
