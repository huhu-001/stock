"""HTML 图表报告：选股结果表 + 个股K线（叠加指标与买卖点）。"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from . import strategies as S
from .timeutil import now_cn

UP, DOWN = "#e03131", "#2f9e44"   # A股：红涨绿跌
LINE_COLORS = ["#f08c00", "#1c7ed6", "#ae3ec9", "#0ca678", "#868e96", "#d6336c"]


def stock_figure(code: str, name: str, df: pd.DataFrame, strategy_names: list[str],
                 bars: int = 160, cfg: dict | None = None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    strats = [S.build(n, cfg) if isinstance(n, str) else n for n in strategy_names]
    strats = [s for s in strats if s.name in S.TECHNICAL]
    frames = [s.run(df) for s in strats]
    view = df.iloc[-bars:]
    panels = [s for s in strats if s.panel_cols]
    rows = 2 + len(panels)
    heights = [0.55, 0.15] + [0.3 / max(len(panels), 1)] * len(panels)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=heights)
    x = view.index.strftime("%Y-%m-%d")
    fig.add_trace(go.Candlestick(x=x, open=view["open"], high=view["high"], low=view["low"],
                                 close=view["close"], name="K线", increasing_line_color=UP,
                                 decreasing_line_color=DOWN, increasing_fillcolor=UP,
                                 decreasing_fillcolor=DOWN), row=1, col=1)
    ci = 0
    for s, res in zip(strats, frames):
        if res.empty:
            continue
        r = res.iloc[-bars:]
        for col in s.overlay_cols:
            if col in r:
                fig.add_trace(go.Scatter(x=x, y=r[col], name=col.upper(), mode="lines",
                                         line=dict(width=1.2, color=LINE_COLORS[ci % 6])),
                              row=1, col=1)
                ci += 1
        for sig, sym, color, lbl in ((S.BUY, "triangle-up", UP, "买"),
                                     (S.SELL, "triangle-down", DOWN, "卖")):
            pts = r[r["signal"] == sig]
            if pts.empty:
                continue
            y = pts["low"] * 0.98 if sig == S.BUY else pts["high"] * 1.02
            fig.add_trace(go.Scatter(
                x=pts.index.strftime("%Y-%m-%d"), y=y, mode="markers",
                marker=dict(symbol=sym, size=11, color=color, line=dict(width=1, color="white")),
                name=f"{s.label}·{lbl}", text=pts["reason"],
                hovertemplate="%{x}<br>%{text}<extra></extra>"), row=1, col=1)
    vol_colors = [UP if c >= o else DOWN for o, c in zip(view["open"], view["close"])]
    fig.add_trace(go.Bar(x=x, y=view["volume"], marker_color=vol_colors, name="成交量",
                         showlegend=False), row=2, col=1)
    for i, s in enumerate(panels):
        res = frames[strats.index(s)].iloc[-bars:]
        row = 3 + i
        if s.name == "macd":
            fig.add_trace(go.Bar(x=x, y=res["macd"], name="MACD柱", showlegend=False,
                                 marker_color=[UP if v >= 0 else DOWN for v in res["macd"]]),
                          row=row, col=1)
            fig.add_trace(go.Scatter(x=x, y=res["dif"], name="DIF", line=dict(width=1.2,
                                     color="#f08c00")), row=row, col=1)
            fig.add_trace(go.Scatter(x=x, y=res["dea"], name="DEA", line=dict(width=1.2,
                                     color="#1c7ed6")), row=row, col=1)
        else:
            for j, col in enumerate(s.panel_cols):
                fig.add_trace(go.Scatter(x=x, y=res[col], name=col.upper(),
                                         line=dict(width=1.2, color="#ae3ec9")), row=row, col=1)
            if "rsi" in s.panel_cols:
                for lvl in (s.params["rsi_low"], s.params["rsi_high"]):
                    fig.add_hline(y=lvl, line=dict(dash="dot", width=1, color="#adb5bd"),
                                  row=row, col=1)
    fig.update_layout(title=f"{name} {code}", height=380 + 160 * rows, template="plotly_white",
                      xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
                      margin=dict(l=50, r=20, t=140, b=30))
    fig.update_xaxes(type="category", nticks=10)
    return fig


def _table_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p class='empty'>没有符合条件的股票。</p>"
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c, v in r.items():
            cls = ""
            if c in ("信号",):
                cls = "up" if v == "买入" else "down"
            elif "涨跌" in str(c) and isinstance(v, (int, float)) and pd.notna(v):
                cls = "up" if v > 0 else ("down" if v < 0 else "")
            if c == "代码":
                v = f"<a href='#c{v}'>{v}</a>"
            else:
                v = html.escape("" if pd.isna(v) else str(v))
            tds.append(f"<td class='{cls}'>{v}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


CSS = """
:root{--bg:#fff;--fg:#212529;--muted:#868e96;--line:#e9ecef;--head:#f8f9fa}
@media (prefers-color-scheme:dark){:root{--bg:#1a1b1e;--fg:#e9ecef;--muted:#909296;--line:#2c2e33;--head:#25262b}}
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;margin:0 auto;max-width:1200px;
padding:24px 16px;background:var(--bg);color:var(--fg)}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}.meta{color:var(--muted);font-size:13px;margin-bottom:20px}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{background:var(--head);position:sticky;top:0}.up{color:#e03131}.down{color:#2f9e44}
a{color:#1c7ed6;text-decoration:none}.chart{margin-top:28px}.empty{color:var(--muted)}
.note{margin-top:32px;color:var(--muted);font-size:12px}
"""


def build_report(title: str, table: pd.DataFrame, hist: dict[str, pd.DataFrame] | None,
                 strategy_names: list[str], out_path: str | Path, max_charts: int = 20,
                 subtitle: str = "", cfg: dict | None = None,
                 extra_tables: list | None = None, extra_html: str = "") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # plotly.js 放在报告同目录，离线也能打开（国内访问 CDN 不稳定）
    js = out_path.parent / "plotly.min.js"
    if (hist or extra_html) and not js.exists():
        from plotly.offline import get_plotlyjs
        js.write_text(get_plotlyjs(), encoding="utf-8")
    parts = [f"<h1>{html.escape(title)}</h1>",
             f"<div class='meta'>生成时间 {now_cn():%Y-%m-%d %H:%M} · {html.escape(subtitle)}</div>",
             f"<div class='wrap'>{_table_html(table)}</div>"]
    for t_title, t_df in extra_tables or []:
        parts.append(f"<h2>{html.escape(t_title)}</h2><div class='wrap'>{_table_html(t_df)}</div>")
    if extra_html:
        parts.append(extra_html)
    first = True
    if hist and table is not None and not table.empty and "代码" in table:
        names = dict(zip(table["代码"], table.get("名称", table["代码"])))
        for code in list(dict.fromkeys(table["代码"]))[:max_charts]:
            if code not in hist:
                continue
            fig = stock_figure(code, names.get(code, ""), hist[code], strategy_names, cfg=cfg)
            parts.append(f"<div class='chart' id='c{code}'>"
                         + fig.to_html(full_html=False, include_plotlyjs="directory" if first else False)
                         + "</div>")
            first = False
    parts.append("<p class='note'>本报告由程序自动生成，仅供研究参考，不构成任何投资建议。</p>")
    doc = (f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
           + "".join(parts) + "</body></html>")
    out_path.write_text(doc, encoding="utf-8")
    return out_path
