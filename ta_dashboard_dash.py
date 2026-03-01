import warnings
import os
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State

# ── Constants ────────────────────────────────────────────────────────────────
STOCKS     = ["MSFT", "GOOGL", "TSLA", "AAPL", "META"]
INDICATORS = ["Bollinger Bands", "MACD", "RSI"]

# ── Indicator calculations ────────────────────────────────────────────────────
def calc_bb(close, n, k):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    return mid + k * std, mid, mid - k * std

def calc_macd(close, fast, slow, signal):
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_rsi(close, n):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs    = gain / loss.replace(0, float('nan'))
    return 100 - (100 / (1 + rs))

# ── Data fetching ─────────────────────────────────────────────────────────────
def fetch_data(asset, start_date, end_date):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Attempt 1: yfinance
        try:
            df = yf.download(asset, start=start_date, end=end_date,
                             progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        except Exception as e:
            print("YFINANCE ERROR:", e)
        # Attempt 2: Stooq direct CSV (fallback when cloud IP is blocked by Yahoo)
        try:
            d1  = start_date.replace('-', '')
            d2  = end_date.replace('-', '')
            url = f"https://stooq.com/q/d/l/?s={asset.lower()}.us&d1={d1}&d2={d2}&i=d"
            df  = pd.read_csv(url)
            if not df.empty and 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                return df
        except Exception as e:
            print("STOOQ ERROR:", e)
    return pd.DataFrame()

# ── Chart builder ─────────────────────────────────────────────────────────────
def build_figure(asset, selected_indicators, start_date, end_date,
                 bb_n, bb_k, macd_fast, macd_slow, macd_signal,
                 rsi_periods, rsi_upper, rsi_lower):
    df = fetch_data(asset, start_date, end_date)
    if df.empty:
        return {}

    show_macd = "MACD" in selected_indicators
    show_rsi  = "RSI"  in selected_indicators

    rows        = 1 + (1 if show_macd else 0) + (1 if show_rsi else 0)
    row_heights = ([0.55] if rows == 1
                   else [0.55, 0.25] if rows == 2
                   else [0.50, 0.25, 0.25])
    specs = [[{"secondary_y": False}]] * rows

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04,
                        row_heights=row_heights, specs=specs)

    close = df["Close"]

    # ── Row 1: Candlestick ───────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"],  close=close, name=asset,
        increasing_line_color="#17BECF",
        decreasing_line_color="#808080",
    ), row=1, col=1)

    # Bollinger Bands overlay
    if "Bollinger Bands" in selected_indicators:
        upper, mid, lower = calc_bb(close, int(bb_n), float(bb_k))
        fig.add_trace(go.Scatter(
            x=df.index, y=upper, name=f"BB Upper({bb_n})",
            line=dict(color="rgba(55,128,191,0.8)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=mid, name=f"BB Mid({bb_n})",
            line=dict(color="rgba(55,128,191,0.5)", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=lower, name=f"BB Lower({bb_n})",
            line=dict(color="rgba(55,128,191,0.8)", width=1),
            fill="tonexty", fillcolor="rgba(55,128,191,0.05)"), row=1, col=1)

    # ── MACD ─────────────────────────────────────────────────────────────────
    current_row = 2
    if show_macd:
        macd_line, signal_line, histogram = calc_macd(
            close, int(macd_fast), int(macd_slow), int(macd_signal))
        colors = ["#17BECF" if v >= 0 else "#EF553B" for v in histogram]
        fig.add_trace(go.Bar(
            x=df.index, y=histogram, name="MACD Hist",
            marker_color=colors, opacity=0.6), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=macd_line, name="MACD",
            line=dict(color="#636EFA", width=1.5)), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=signal_line, name="Signal",
            line=dict(color="#FFA15A", width=1.5)), row=current_row, col=1)
        current_row += 1

    # ── RSI ──────────────────────────────────────────────────────────────────
    if show_rsi:
        rsi = calc_rsi(close, int(rsi_periods))
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi, name=f"RSI({rsi_periods})",
            line=dict(color="#AB63FA", width=1.5)), row=current_row, col=1)
        fig.add_hline(y=float(rsi_upper), line_dash="dot",
                      line_color="rgba(239,85,59,0.6)",
                      row=current_row, col=1)
        fig.add_hline(y=float(rsi_lower), line_dash="dot",
                      line_color="rgba(0,204,150,0.6)",
                      row=current_row, col=1)

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=f"TA Dashboard — {asset}", font=dict(color="#7c9ef5")),
        paper_bgcolor="#13131f",
        plot_bgcolor="#1e1e2e",
        font=dict(color="#e0e0e0"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0")),
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=60, b=20),
        height=700,
    )
    fig.update_xaxes(gridcolor="#2e2e3e", showgrid=True)
    fig.update_yaxes(gridcolor="#2e2e3e", showgrid=True)
    return fig

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
                       marks={1:  {"label": "1",  "style": SLIDER_MARKS_STYLE},
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
                       marks={2:  {"label": "2",  "style": SLIDER_MARKS_STYLE},
                              12: {"label": "12", "style": SLIDER_MARKS_STYLE},
                              50: {"label": "50", "style": SLIDER_MARKS_STYLE}},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Div("Slow Period", style={**LABEL_STYLE, "marginTop": "20px"}),
            dcc.Slider(id="macd-slow", min=2, max=50, step=1, value=26,
                       marks={2:  {"label": "2",  "style": SLIDER_MARKS_STYLE},
                              26: {"label": "26", "style": SLIDER_MARKS_STYLE},
                              50: {"label": "50", "style": SLIDER_MARKS_STYLE}},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Div("Signal Period", style={**LABEL_STYLE, "marginTop": "20px"}),
            dcc.Slider(id="macd-signal", min=2, max=50, step=1, value=9,
                       marks={2:  {"label": "2",  "style": SLIDER_MARKS_STYLE},
                              9:  {"label": "9",  "style": SLIDER_MARKS_STYLE},
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
                       marks={2:  {"label": "2",  "style": SLIDER_MARKS_STYLE},
                              14: {"label": "14", "style": SLIDER_MARKS_STYLE},
                              50: {"label": "50", "style": SLIDER_MARKS_STYLE}},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Div("Upper Threshold", style={**LABEL_STYLE, "marginTop": "20px"}),
            dcc.Slider(id="rsi-upper", min=50, max=100, step=1, value=70,
                       marks={50:  {"label": "50",  "style": SLIDER_MARKS_STYLE},
                              70:  {"label": "70",  "style": SLIDER_MARKS_STYLE},
                              100: {"label": "100", "style": SLIDER_MARKS_STYLE}},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Div("Lower Threshold", style={**LABEL_STYLE, "marginTop": "20px"}),
            dcc.Slider(id="rsi-lower", min=1, max=50, step=1, value=30,
                       marks={1:  {"label": "1",  "style": SLIDER_MARKS_STYLE},
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
    print(f"\n🚀  TA Dashboard running at: http://127.0.0.1:{port}\n")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="0.0.0.0", port=port)
