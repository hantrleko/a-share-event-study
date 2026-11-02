"""EF Lab page styles for the event-study app."""

APP_CSS = """
        <style>
        html, body, [class*="css"] { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
        .stApp { background: #f4f7fb; color: #16324f; }
        header[data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stAppDeployButton"],
        .stAppDeployButton,
        .stDeployButton,
        #MainMenu { display: none !important; }
        .block-container { padding-top: 0.45rem; padding-bottom: 2.2rem; max-width: 1180px; }
        section[data-testid="stSidebar"] {
          background: #eef3f8;
          border-right: 1px solid #d5dee8;
        }
        section[data-testid="stSidebar"] > div { padding-top: 0.85rem; }
        section[data-testid="stSidebar"] h2 {
          font-size: 0.72rem;
          letter-spacing: 0.16em;
          color: #5a6a7a;
          font-weight: 600;
          margin-top: 0.4rem;
          padding-top: 0.85rem;
          border-top: 1px solid #d5dee8;
        }
        section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { font-size: 0.86rem; }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
          background: #ffffff;
          border-color: #d5dee8;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 0.15rem; border-bottom: 1px solid #d5dee8; }
        .stTabs [data-baseweb="tab"] { background: transparent; color: #5a6a7a; border-radius: 0; }
        .stTabs [aria-selected="true"] { color: #1f4e79; border-bottom: 2px solid #1f4e79; }
        div[data-testid="stAlert"] {
          background: #f4f7fb;
          border: 0;
          border-left: 3px solid #1f4e79;
          border-radius: 2px;
          color: #16324f;
        }
        div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
          border: 1px solid #d5dee8;
          border-radius: 2px;
          background: #ffffff;
        }
        div[data-testid="stImage"] {
          background: #ffffff;
          border: 1px solid #d5dee8;
          padding: 0.35rem;
        }
        .stButton > button[kind="primary"] {
          background: #1f4e79;
          border: 1px solid #1f4e79;
        }
        .brand h1 {
          margin: 0;
          color: #1f4e79;
          font-size: 1.65rem;
          font-weight: 650;
          letter-spacing: -0.02em;
          line-height: 1.25;
        }
        .brand .kicker {
          margin: 0 0 0.15rem;
          color: #5a6a7a;
          font-size: 0.72rem;
          letter-spacing: 0.16em;
          font-weight: 600;
        }
        .brand .subtitle { margin: 0.35rem 0 0; color: #5a6a7a; font-size: 0.92rem; }
        .kpi-row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.55rem; margin: 0.4rem 0 0.7rem; }
        .kpi {
          background: #ffffff;
          border: 1px solid #d5dee8;
          border-radius: 8px;
          padding: 0.62rem 0.75rem 0.55rem;
          box-shadow: 0 1px 2px rgba(22, 50, 79, 0.05), 0 6px 16px rgba(22, 50, 79, 0.04);
        }
        .kpi span { display: block; font-size: 0.72rem; letter-spacing: 0.04em; color: #5a6a7a; }
        .kpi strong {
          display: block; margin-top: 0.18rem; font-size: 1.28rem; font-weight: 560;
          font-variant-numeric: tabular-nums; color: #16324f;
        }
        .kpi.neg strong { color: #9b3a3a; }
        .kpi.pos strong { color: #1d6b45; }
        .site-footer {
          margin-top: 1.6rem; padding-top: 0.75rem; border-top: 1px solid #d5dee8;
          color: #5a6a7a; font-size: 0.82rem;
        }
        .lesson {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.7rem;
  margin: 0.2rem 0 0.4rem;
}
.lesson article {
  background: #ffffff;
  border: 1px solid #d5dee8;
  border-radius: 8px;
  padding: 0.85rem 0.95rem 0.8rem;
  box-shadow: 0 1px 2px rgba(22, 50, 79, 0.05);
}
.lesson strong { display: block; color: #1f4e79; margin-bottom: 0.35rem; font-size: 0.95rem; }
        .lesson p { margin: 0; color: #3d4c5c; font-size: 0.92rem; line-height: 1.6; }
        .window-row {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0.55rem;
          margin: 0.2rem 0 0.7rem;
        }
        .window-row article {
          background: #ffffff;
          border: 1px solid #d5dee8;
          border-radius: 8px;
          padding: 0.7rem 0.8rem 0.65rem;
          box-shadow: 0 1px 2px rgba(22, 50, 79, 0.05);
        }
        .window-row strong { display: block; color: #5a6a7a; font-size: 0.78rem; font-weight: 600; }
        .window-row b {
          display: block; margin-top: 0.15rem; color: #16324f; font-size: 1.2rem; font-weight: 560;
          font-variant-numeric: tabular-nums;
        }
        .window-row span { display: block; margin-top: 0.2rem; color: #5a6a7a; font-size: 0.78rem; }
        @media (max-width: 900px) {
          .kpi-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .brand h1 { font-size: 1.25rem; }
          .lesson { grid-template-columns: 1fr; }
          .window-row { grid-template-columns: 1fr; }
        }
        </style>
"""