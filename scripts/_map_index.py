"""Generate an HTML index page listing all maps in the outputs/ directory."""

import re
from datetime import datetime
from pathlib import Path


def generate_index(output_dir: Path) -> None:
    maps = sorted(
        [p for p in output_dir.glob("*.html") if p.name != "index.html"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not maps:
        return

    def describe(p: Path) -> tuple[str, str, str]:
        name  = p.stem
        mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
        if name == "carte_desserte":
            return "Coverage", "Network coverage score by IRIS district", mtime
        if name.startswith("carte_commute_"):
            slug = name.replace("carte_commute_", "").replace("_", " ").title()
            return "Commute", f"Travel time to: {slug}", mtime
        return "Map", name, mtime

    rows   = ""
    badges = {"Coverage": "#3b82f6", "Commute": "#10b981"}
    for p in maps:
        kind, label, mtime = describe(p)
        color = badges.get(kind, "#6b7280")
        rows += f"""
        <tr>
          <td style="padding:12px 16px">
            <span style="background:{color};color:#fff;font-size:11px;
                  padding:2px 8px;border-radius:10px;font-weight:600">{kind}</span>
          </td>
          <td style="padding:12px 8px">
            <a href="{p.name}" style="color:#1d4ed8;text-decoration:none;
               font-weight:500">{label}</a>
          </td>
          <td style="padding:12px 16px;color:#6b7280;font-size:12px;
               white-space:nowrap">{mtime}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Toulouse Transit Explorer — Maps</title>
  <style>
    body {{font-family:Arial,sans-serif;background:#f8fafc;margin:0;padding:40px 20px}}
    .container {{max-width:780px;margin:0 auto}}
    h1 {{color:#1e293b;font-size:24px;margin-bottom:4px}}
    p.sub {{color:#64748b;margin-top:4px;margin-bottom:32px;font-size:14px}}
    table {{width:100%;border-collapse:collapse;background:#fff;
            border-radius:10px;box-shadow:0 1px 6px rgba(0,0,0,.08);overflow:hidden}}
    thead tr {{background:#f1f5f9}}
    thead th {{padding:10px 16px;text-align:left;font-size:12px;
               color:#475569;text-transform:uppercase;letter-spacing:.05em}}
    tbody tr:hover {{background:#f8fafc}}
    tbody tr:not(:last-child) {{border-bottom:1px solid #f1f5f9}}
    a:hover {{text-decoration:underline}}
    .hint {{margin-top:28px;padding:14px 18px;background:#eff6ff;
            border-left:3px solid #3b82f6;border-radius:4px;
            font-size:13px;color:#1e40af;line-height:1.6}}
    code {{background:#e0e7ff;padding:1px 5px;border-radius:3px;font-size:12px}}
  </style>
</head>
<body>
<div class="container">
  <h1>Toulouse Transit Explorer</h1>
  <p class="sub">Generated maps &bull; Updated: {datetime.now().strftime("%d/%m/%Y %H:%M")}</p>
  <table>
    <thead><tr>
      <th>Type</th><th>Map</th><th>Generated</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="hint">
    <b>Add a commute map:</b><br>
    <code>python scripts/tisseo_commute.py --dest "Address or LAT,LON"</code><br>
    <code>python scripts/tisseo_commute.py --dest "Airbus ZAC" --heure 9</code>
  </div>
</div>
</body>
</html>"""

    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
