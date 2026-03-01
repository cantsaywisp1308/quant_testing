import warnings
import pandas as pd
import yfinance as yf
import cufflinks as cf
import cufflinks.colors as _cf_colors
import cufflinks.plotlytools as _cf_pt
import cufflinks.tools as _cf_tools
import plotly.offline as _py_offline
from dash import Dash, dcc, html, Input, Output, State
import os

# ── Cufflinks compatibility patches ─────────────────────────────────────────
# Fix: newer numpy formats np.float64 as 'np.float64(x)' in strings,
# breaking cufflinks rgba color generation.
def _patched_to_rgba(color, alpha):
    if type(color) == tuple:
        color, alpha = color
    color = color.lower()
    if 'rgba' in color:
        cl = list(eval(color.replace('rgba', '')))
        if alpha:
            cl[3] = float(alpha)
        r, g, b, a = int(cl[0]), int(cl[1]), int(cl[2]), float(cl[3])
        return f'rgba({r}, {g}, {b}, {a})'
    elif 'rgb' in color:
        r, g, b = eval(color.replace('rgb', ''))
        return f'rgba({int(r)}, {int(g)}, {int(b)}, {float(alpha)})'
    else:
        return _patched_to_rgba(_cf_colors.hex_to_rgb(color), alpha)

_cf_colors.to_rgba = _patched_to_rgba
_cf_pt.to_rgba    = _patched_to_rgba
_cf_tools.to_rgba = _patched_to_rgba

# Force cufflinks offline mode (go_offline() only sets the flag inside Jupyter)
cf.go_offline()
_py_offline.__PLOTLY_OFFLINE_INITIALIZED = True

# ── Constants ────────────────────────────────────────────────────────────────
STOCKS     = ["MSFT", "GOOGL", "TSLA", "AAPL", "META"]
INDICATORS = ["Bollinger Bands", "MACD", "RSI"]

# ── Chart builder ────────────────────────────────────────────────────────────
def build_figure(asset, selected_indicators, start_date, end_date,
                 bb_n, bb_k, macd_fast, macd_slow, macd_signal,
                 rsi_periods, rsi_upper, rsi_lower):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = None
        # Attempt 1: yfinance (preferred because it's usually more robust for all tickers)
        try:
            temp_df = yf.download(asset, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not temp_df.empty:
                df = temp_df
                # Flatten MultiIndex columns (newer yfinance versions)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
        except Exception:
            pass

        # Attempt 2: Fallback to stooq via direct CSV fetch if yfinance fails (happens on Render due to IP blocks)
        if df is None or df.empty:
            try:
                # Stooq format: d1 and d2 are YYYYMMDD
                d1 = start_date.replace('-', '')
                d2 = end_date.replace('-', '')
                url = f"https://stooq.com/q/d/l/?s={asset.lower()}.us&d1={d1}&d2={d2}&i=d"
                stooq_df = pd.read_csv(url)
                if not stooq_df.empty and 'Date' in stooq_df.columns:
                    stooq_df['Date'] = pd.to_datetime(stooq_df['Date'])
                    stooq_df.set_index('Date', inplace=True)
                    stooq_df.sort_index(inplace=True)
                    df = stooq_df
                else:
                    return {}
            except Exception:
                return {}

    if df is None or df.empty:
        return {}

    qf = cf.QuantFig(df, title=f'TA Dashboard — {asset}',
                     legend='right', name=asset)

    if 'Bollinger Bands' in selected_indicators:
        qf.add_bollinger_bands(periods=int(bb_n), boll_std=float(bb_k))

    if 'MACD' in selected_indicators:
        qf.add_macd(fast_period=int(macd_fast),
                    slow_period=int(macd_slow),
                    signal_period=int(macd_signal))

    if 'RSI' in selected_indicators:
        qf.add_rsi(periods=int(rsi_periods),
                   rsi_upper=float(rsi_upper),
                   rsi_lower=float(rsi_lower),
                   showbands=True)
    print("DF HEAD:", df.head())
    print("DF EMPTY:", df.empty)
    return qf.iplot(asFigure=True)

# ── App layout ───────────────────────────────────────────────────────────────
app = Dash(__name__)
server = app.server
app.title = "TA Dashboard"

LABEL_STYLE = {"fontWeight": "bold", "marginBottom": "4px"}
SECTION_STYLE = {
    "background": "#1e1e2e", "padding": "16px", "borderRadius": "10px",
    "marginBottom": "12px"
}
SLIDER_MARKS_STYLE = {"color": "#aaa", "fontSize": "11px"}

app.layout = html.Div(style={"fontFamily": "Inter, sans-serif",
                              "background": "#13131f", "minHeight": "100vh",
                              "padding": "24px", "color": "#e0e0e0"}, children=[

    # ── Header ──────────────────────────────────────────────────────────────
    html.H1("📈 Technical Analysis Dashboard",
            style={"textAlign": "center", "color": "#7c9ef5",
                   "marginBottom": "24px", "fontSize": "28px"}),

    # ── Controls row ────────────────────────────────────────────────────────
    html.Div(style={"display": "flex", "gap": "16px",
                    "flexWrap": "wrap", "marginBottom": "16px"}, children=[

        # Main selectors
        html.Div(style={**SECTION_STYLE, "flex": "1", "minWidth": "260px"}, children=[
            html.Div("Main Selectors",
                     style={"fontSize": "16px", "fontWeight": "bold",
                            "color": "#7c9ef5", "marginBottom": "12px"}),

            html.Div("Stock", style=LABEL_STYLE),
            dcc.Dropdown(id="stock", options=[{"label": s, "value": s} for s in STOCKS],
                         value=STOCKS[0], clearable=False,
                         style={"color": "#000", "marginBottom": "12px"}),

            html.Div("Indicators", style=LABEL_STYLE),
            dcc.Checklist(id="indicators",
                          options=[{"label": f"  {i}", "value": i} for i in INDICATORS],
                          value=INDICATORS,
                          style={"marginBottom": "12px", "lineHeight": "2"},
                          inputStyle={"marginRight": "6px"}),

            html.Div("Date Range", style=LABEL_STYLE),
            dcc.DatePickerRange(id="date-range",
                                start_date="2018-01-01",
                                end_date="2018-12-31",
                                display_format="YYYY-MM-DD",
                                style={"marginBottom": "4px"}),
        ]),

        # Bollinger Bands
        html.Div(style={**SECTION_STYLE, "flex": "1", "minWidth": "260px"}, children=[
            html.Div("🎯 Bollinger Bands",
                     style={"fontSize": "16px", "fontWeight": "bold",
                            "color": "#f5a97f", "marginBottom": "16px"}),

            html.Div("N — Periods", style=LABEL_STYLE),
            dcc.Slider(id="bb-n", min=1, max=40, step=1, value=20,
                       marks={1: {"label": "1", "style": SLIDER_MARKS_STYLE},
                              20: {"label": "20", "style": SLIDER_MARKS_STYLE},
                              40: {"label": "40", "style": SLIDER_MARKS_STYLE}},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Div("k — Std Dev", style={**LABEL_STYLE, "marginTop": "20px"}),
            dcc.Slider(id="bb-k", min=0.5, max=4.0, step=0.5, value=2.0,
                       marks={0.5: {"label": "0.5", "style": SLIDER_MARKS_STYLE},
                              2.0: {"label": "2",   "style": SLIDER_MARKS_STYLE},
                              4.0: {"label": "4",   "style": SLIDER_MARKS_STYLE}},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]),

        # MACD
        html.Div(style={**SECTION_STYLE, "flex": "1", "minWidth": "260px"}, children=[
            html.Div("📊 MACD",
                     style={"fontSize": "16px", "fontWeight": "bold",
                            "color": "#a6e3a1", "marginBottom": "16px"}),

            html.Div("Fast Period", style=LABEL_STYLE),
            dcc.Slider(id="macd-fast", min=2, max=50, step=1, value=12,
                       marks={2: {"label": "2", "style": SLIDER_MARKS_STYLE},
                              12: {"label": "12", "style": SLIDER_MARKS_STYLE},
                              50: {"label": "50", "style": SLIDER_MARKS_STYLE}},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Div("Slow Period", style={**LABEL_STYLE, "marginTop": "20px"}),
            dcc.Slider(id="macd-slow", min=2, max=50, step=1, value=26,
                       marks={2: {"label": "2", "style": SLIDER_MARKS_STYLE},
                              26: {"label": "26", "style": SLIDER_MARKS_STYLE},
                              50: {"label": "50", "style": SLIDER_MARKS_STYLE}},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Div("Signal Period", style={**LABEL_STYLE, "marginTop": "20px"}),
            dcc.Slider(id="macd-signal", min=2, max=50, step=1, value=9,
                       marks={2: {"label": "2", "style": SLIDER_MARKS_STYLE},
                              9: {"label": "9",  "style": SLIDER_MARKS_STYLE},
                              50: {"label": "50", "style": SLIDER_MARKS_STYLE}},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]),

        # RSI
        html.Div(style={**SECTION_STYLE, "flex": "1", "minWidth": "260px"}, children=[
            html.Div("📉 RSI",
                     style={"fontSize": "16px", "fontWeight": "bold",
                            "color": "#cba6f7", "marginBottom": "16px"}),

            html.Div("RSI Period", style=LABEL_STYLE),
            dcc.Slider(id="rsi-period", min=2, max=50, step=1, value=14,
                       marks={2: {"label": "2",  "style": SLIDER_MARKS_STYLE},
                              14: {"label": "14", "style": SLIDER_MARKS_STYLE},
                              50: {"label": "50", "style": SLIDER_MARKS_STYLE}},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Div("Upper Threshold", style={**LABEL_STYLE, "marginTop": "20px"}),
            dcc.Slider(id="rsi-upper", min=50, max=100, step=1, value=70,
                       marks={50: {"label": "50",  "style": SLIDER_MARKS_STYLE},
                              70: {"label": "70",  "style": SLIDER_MARKS_STYLE},
                              100: {"label": "100", "style": SLIDER_MARKS_STYLE}},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Div("Lower Threshold", style={**LABEL_STYLE, "marginTop": "20px"}),
            dcc.Slider(id="rsi-lower", min=1, max=50, step=1, value=30,
                       marks={1: {"label": "1",  "style": SLIDER_MARKS_STYLE},
                              30: {"label": "30", "style": SLIDER_MARKS_STYLE},
                              50: {"label": "50", "style": SLIDER_MARKS_STYLE}},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]),
    ]),

    # ── Chart ────────────────────────────────────────────────────────────────
    html.Div(style=SECTION_STYLE, children=[
        dcc.Loading(
            type="circle",
            color="#7c9ef5",
            children=dcc.Graph(id="ta-chart",
                               style={"height": "700px"},
                               config={"displayModeBar": True})
        )
    ]),
])

# ── Callback ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("ta-chart", "figure"),
    Input("stock",       "value"),
    Input("indicators",  "value"),
    Input("date-range",  "start_date"),
    Input("date-range",  "end_date"),
    Input("bb-n",        "value"),
    Input("bb-k",        "value"),
    Input("macd-fast",   "value"),
    Input("macd-slow",   "value"),
    Input("macd-signal", "value"),
    Input("rsi-period",  "value"),
    Input("rsi-upper",   "value"),
    Input("rsi-lower",   "value"),
)
def update_chart(stock, indicators, start_date, end_date,
                 bb_n, bb_k, macd_fast, macd_slow, macd_signal,
                 rsi_period, rsi_upper, rsi_lower):
    if not stock or not start_date or not end_date:
        return {}
    selected = indicators or []
    return build_figure(
        asset=stock,
        selected_indicators=selected,
        start_date=start_date[:10],
        end_date=end_date[:10],
        bb_n=bb_n, bb_k=bb_k,
        macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
        rsi_periods=rsi_period, rsi_upper=rsi_upper, rsi_lower=rsi_lower,
    )

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import threading, webbrowser

    port = int(os.environ.get("PORT", 8050))
    print("\n🚀  TA Dashboard running at: http://127.0.0.1:8050\n")
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8050")).start()
    app.run(host="0.0.0.0", port=8050)
