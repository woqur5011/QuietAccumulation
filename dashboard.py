"""
조용한 매집 - Streamlit 대시보드 v2
날짜별 YYYYMMDD.csv 로드 + 날짜 선택기 + 코스피졤코스닥 전체 지원
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    from pykrx import stock as pykrx_stock
    _PYKRX_OK = True
except ImportError:
    _PYKRX_OK = False

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="조용한 매집",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data(ttl=3600)
def get_market_map() -> dict:
    """ticker -> 'KOSPI' or 'KOSDAQ' 매핑 (1시간 캐시)"""
    if not _PYKRX_OK:
        return {}
    try:
        today = datetime.now().strftime("%Y%m%d")
        kospi  = pykrx_stock.get_market_ticker_list(today, market="KOSPI")
        kosdaq = pykrx_stock.get_market_ticker_list(today, market="KOSDAQ")
        m = {t: "KOSPI" for t in kospi}
        m.update({t: "KOSDAQ" for t in kosdaq})
        return m
    except Exception:
        return {}


def get_available_dates() -> list[str]:
    """data/ 디렉터리에서 YYYYMMDD.csv 파일의 날짜 목록 (내림차순)"""
    files = sorted(
        [f.stem for f in DATA_DIR.glob("????????.csv") if f.stem.isdigit()],
        reverse=True,
    )
    return files


def get_latest_csv() -> Path | None:
    dates = get_available_dates()
    if dates:
        return DATA_DIR / f"{dates[0]}.csv"
    # latest.csv fallback
    p = DATA_DIR / "latest.csv"
    return p if p.exists() else None


def load_summary(date_str: str) -> dict | None:
    """data/YYYYMMDD_summary.json 로드. 없으면 None."""
    path = DATA_DIR / f"{date_str}_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

# ─────────────────────────────────────────────
# CSS (이미지와 유사한 다크 테마)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* 전체 배경 */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
    color: #f9fafb;
}
[data-testid="stSidebar"] {
    background: #0b1220;
}
h1, h2, h3, .stMarkdown p { color: #f9fafb; }

/* 메트릭 카드 */
[data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 8px 12px;
}

/* 테이블 */
.dataframe-container table {
    border-collapse: collapse;
    width: 100%;
}

/* 배지 색상 */
.grade-S  { background:#22c55e; color:#fff; border-radius:4px; padding:2px 8px; font-weight:700; }
.grade-A  { background:#3b82f6; color:#fff; border-radius:4px; padding:2px 8px; font-weight:700; }
.grade-B  { background:#f59e0b; color:#fff; border-radius:4px; padding:2px 8px; font-weight:700; }
.grade-C  { background:#f97316; color:#fff; border-radius:4px; padding:2px 8px; font-weight:700; }
.grade-D  { background:#ef4444; color:#fff; border-radius:4px; padding:2px 8px; font-weight:700; }
.grade-NA { background:#6b7280; color:#fff; border-radius:4px; padding:2px 8px; }

/* 수집시각 */
.collected-at { color:#9ca3af; font-size:12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8-sig", dtype={"티커": str})


def safe_num(val):
    try:
        return float(val)
    except Exception:
        return None


# ─────────────────────────────────────────────
# 52주 위치 색상
# ─────────────────────────────────────────────
def w52_color(val) -> str:
    v = safe_num(val)
    if v is None:
        return ""
    if v >= 70:
        return "background-color:#fca5a5; color:#7f1d1d;"     # 빨강 (과매수)
    if v >= 50:
        return "background-color:#fde68a; color:#78350f;"     # 노랑
    return "background-color:#bbf7d0; color:#14532d;"         # 초록 (저평가)


# ─────────────────────────────────────────────
# 재무등급 색상
# ─────────────────────────────────────────────
GRADE_COLORS = {
    "S": "#22c55e",
    "A": "#3b82f6",
    "B": "#f59e0b",
    "C": "#f97316",
    "D": "#ef4444",
}

def grade_color(val) -> str:
    bg = GRADE_COLORS.get(str(val).strip(), "#6b7280")
    return f"background-color:{bg}; color:white; font-weight:700; border-radius:4px;"


# ─────────────────────────────────────────────
# 점수 계산
# ─────────────────────────────────────────────
def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    4가지 점수 계산 (0~10, 백분위 기반):
      외인연속점수, 기관연속점수, 수급합점수, 조용한매집점수
    합산점수 = 4*외인 + 3*기관 + 2*수급합 + 1*조용한매집 (max 100)
    """
    df = df.copy()
    if len(df) == 0:
        for c in ['외인연속점수', '기관연속점수', '수급합점수', '조용한매집점수', '합산점수']:
            df[c] = 0.0
        return df

    f    = pd.to_numeric(df.get('외인연속매수일',  0), errors='coerce').fillna(0)
    inst = pd.to_numeric(df.get('기관연속매수일',  0), errors='coerce').fillna(0)
    # 외인/기관매수(억) 는 "+43.28억(16일)" 형식 문자열 — 숫자 부분만 추출
    def _parse_bil(col):
        raw = df.get(col, pd.Series(0, index=df.index))
        return pd.to_numeric(
            raw.astype(str).str.extract(r'([+-]?\d+\.?\d*)')[0],
            errors='coerce'
        ).fillna(0)
    f_bil = _parse_bil('외인매수(억)')
    i_bil = _parse_bil('기관매수(억)')

    def pct_score(s: pd.Series) -> pd.Series:
        """0~10 백분위 점수"""
        return (s.rank(pct=True, na_option='bottom') * 10).round(2)

    # 1) 외인연속매수 점수
    df['외인연속점수'] = pct_score(f)

    # 2) 기관연속매수 점수
    df['기관연속점수'] = pct_score(inst)

    # 3) 수급합 점수 (외인 + 기관 순매수 합계)
    df['수급합점수'] = pct_score(f_bil + i_bil)

    # 4) 조용한매집점수
    #   강도 : (외인연속 + 기관연속) / 30  (cap 1.0)
    #   가산 : 둘 다 >=3일이면 x1.0, 하나만이면 x0.6, 없으면 x0.0
    buy_strength = ((f + inst) / 30).clip(0, 1)
    qualify = np.where(
        (f >= 3) & (inst >= 3), 1.0,
        np.where((f >= 3) | (inst >= 3), 0.6, 0.0)
    )
    raw_quiet   = pd.Series(buy_strength * qualify, index=df.index)
    df['조용한매집점수'] = pct_score(raw_quiet)

    # 합산점수 (max 3 + min 2 + 수급합 3 + 조용한 2 = max 100)
    df['합산점수'] = (
        df[['외인연속점수', '기관연속점수']].max(axis=1) * 3 +
        df[['외인연속점수', '기관연속점수']].min(axis=1) * 2 +
        df['수급합점수']     * 3 +
        df['조용한매집점수'] * 2
    ).round(2)

    return df


def score_color(val) -> str:
    v = safe_num(val)
    if v is None:
        return ""
    if v >= 8:   return "background-color:#15803d; color:#bbf7d0; font-weight:700;"
    if v >= 6:   return "background-color:#1e3a2e; color:#86efac;"
    if v >= 4:   return "background-color:#1c2e1e; color:#6ee7a0;"
    return ""


# ─────────────────────────────────────────────
# 점수 컬럼 툴팁 (마우스오버 설명)
# ─────────────────────────────────────────────
SCORE_COL_CONFIG = {
    "전체순위": st.column_config.NumberColumn("순위", width="small", help="전체 종목 합산점수 기준 순위", pinned=True),
    "종목명": st.column_config.TextColumn("종목명", width="medium", pinned=True),
    "시장": st.column_config.TextColumn("시장", width="small", help="KOSPI / KOSDAQ"),
    "네이버": st.column_config.LinkColumn(
        "네이버",
        display_text="📈",
        help="네이버 금융에서 종목 상세 보기",
        width="small",
        pinned=True,
    ),
    "외인연속점수": st.column_config.NumberColumn(
        "외인연속점수",
        format="%.1f",
        help=(
            "📌 외인 연속 순매수일수 백분위 (0-10점)\n"
            "최근일부터 외국인이 연속으로 순매수한 일수를\n"
            "전체 종목 대비 백분위로 환산한 점수"
        ),
    ),
    "기관연속점수": st.column_config.NumberColumn(
        "기관연속점수",
        format="%.1f",
        help=(
            "📌 기관 연속 순매수일수 백분위 (0-10점)\n"
            "최근일부터 기관이 연속으로 순매수한 일수를\n"
            "전체 종목 대비 백분위로 환산한 점수"
        ),
    ),
    "수급합점수": st.column_config.NumberColumn(
        "수급합점수",
        format="%.1f",
        help=(
            "📌 외인+기관 누적 순매수 금액 합산 백분위 (0-10점)\n"
            "연속매수 기간 동안 외인·기관이 순매수한 총 금액(억원) 합계를\n"
            "전체 종목 대비 백분위로 환산한 점수"
        ),
    ),
    "조용한매집점수": st.column_config.NumberColumn(
        "조용한매집점수",
        format="%.1f",
        help=(
            "📌 조용한 장기 매집 지표 (0-10점, 연속 변수)\n"
            "① 연속수급강도 (60%): (외인연속+기관연속)/20 × 가산점(x1.0/x0.6/x0.0)\n"
            "② 주가안정 (20%): P5/P20이 이상적으로 95-108 구간에서 최고점\n"
            "③ 거래량신호 (20%): V5/V20이 100-180 구간(조용한 매집 거래량)"
        ),
    ),
    "합산점수": st.column_config.NumberColumn(
        "합산점수",
        format="%.1f",
        help=(
            "📌 종합 점수 (최대 100점)\n"
            "= max(외인,기관) × 3 + min(외인,기관) × 2\n"
            "+ 수급합점수 × 3 + 조용한매집 × 2"
        ),
    ),
}


# ─────────────────────────────────────────────
# 사이드바 필터
# ─────────────────────────────────────────────
def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("## 필터")

    # ── 상위 N개 ──
    st.sidebar.markdown("### 합산점수 상위")
    show_top = st.sidebar.checkbox("상위 N개만 표시", value=True, key="sb_show_top")
    top_n    = st.sidebar.slider("상위 N개", 5, 200, 20, 5, disabled=not show_top, key="sb_top_n")

    st.sidebar.markdown("### 세부 필터")
    grades = sorted(df["재무등급"].dropna().unique().tolist())
    sel_grades = st.sidebar.multiselect("재무등급", grades, default=grades, key="sb_grades")

    # 외인 연속매수일 최소
    min_foreign = st.sidebar.slider("외인 연속매수일 ≥", 0, 30, 0, key="sb_min_foreign")
    # 기관 연속매수일 최소
    min_inst = st.sidebar.slider("기관 연속매수일 ≥", 0, 30, 0, key="sb_min_inst")
    # 52주 위치 범위
    w52_min, w52_max = st.sidebar.slider("52주 위치(%)", 0.0, 100.0, (0.0, 100.0), step=1.0, key="sb_w52")
    # 라이브PBR 최대
    pbr_max = st.sidebar.slider("라이브PBR ≤", 0.0, 30.0, 30.0, step=0.5, key="sb_pbr")

    # 종목명 검색
    search = st.sidebar.text_input("종목명 검색", "", key="sb_search")

    # 정렬 기준
    sort_col = st.sidebar.selectbox(
        "정렬 기준",
        ["합산점수", "외인연속점수", "기관연속점수", "수급합점수", "조용한매집점수",
         "외인연속매수일", "기관연속매수일", "52주위치(%)", "라이브PBR",
         "외인매수(억)", "기관매수(억)", "V5/V20", "P5/P20", "부채비율"],
        index=0, key="sb_sort_col",
    )
    sort_asc = st.sidebar.checkbox("오름차순", value=False, key="sb_sort_asc")

    # ── 필터 적용 ──
    fdf = df.copy()

    if sel_grades:
        fdf = fdf[fdf["재무등급"].isin(sel_grades)]

    if "외인연속매수일" in fdf.columns:
        fdf = fdf[pd.to_numeric(fdf["외인연속매수일"], errors="coerce").fillna(0) >= min_foreign]
    if "기관연속매수일" in fdf.columns:
        fdf = fdf[pd.to_numeric(fdf["기관연속매수일"], errors="coerce").fillna(0) >= min_inst]

    if "52주위치(%)" in fdf.columns:
        w52 = pd.to_numeric(fdf["52주위치(%)"], errors="coerce")
        fdf = fdf[(w52 >= w52_min) & (w52 <= w52_max)]

    if "라이브PBR" in fdf.columns:
        pbr = pd.to_numeric(fdf["라이브PBR"], errors="coerce")
        fdf = fdf[pbr.isna() | (pbr <= pbr_max)]

    if search.strip():
        fdf = fdf[fdf["종목명"].str.contains(search.strip(), na=False)]

    if sort_col in fdf.columns:
        fdf = fdf.sort_values(
            sort_col,
            key=lambda s: pd.to_numeric(s, errors="coerce"),
            ascending=sort_asc,
            na_position="last"
        )

    # 상위 N개 적용 (세부 필터 후 적용)
    if show_top:
        fdf = fdf.head(top_n)

    fdf = fdf.reset_index(drop=True)

    return fdf


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    # 헤더
    col_title, col_meta = st.columns([3, 2])
    with col_title:
        st.markdown("# 📊 조용한 매집")
        st.caption("pykrx + 네이버 금융 기반 외인·기관 수급 대시보드")

    # ── 날짜 선택 (사이드바) ──
    available = get_available_dates()
    if not available:
        # latest.csv fallback
        p = DATA_DIR / "latest.csv"
        available = ["latest"] if p.exists() else []

    def _date_label(d: str) -> str:
        if len(d) == 8 and d.isdigit():
            p = DATA_DIR / f"{d}.csv"
            try:
                n = sum(1 for _ in open(p, encoding="utf-8-sig")) - 1  # 헤더 제외
            except Exception:
                n = 0
            suffix = f" [전체 {n:,}종목]" if n > 20 else f" [워치리스트 {n}종목]"
            return f"{d[:4]}-{d[4:6]}-{d[6:]}{suffix}"
        return d

    selected_label = st.sidebar.selectbox(
        "📅 기준 날짜",
        options=available if available else ["(없음)"],
        index=0,
        format_func=_date_label,
    )

    # CSV 경로 결정
    if selected_label == "latest":
        csv_path = str(DATA_DIR / "latest.csv")
    elif selected_label and selected_label != "(없음)":
        csv_path = str(DATA_DIR / f"{selected_label}.csv")
    else:
        csv_path = ""

    # AI 요약 탭에서 참조할 날짜를 session_state에 저장
    _date_for_summary = selected_label if (selected_label and selected_label != "(없음)" and selected_label != "latest") else ""
    st.session_state["_sel_date_for_summary"] = _date_for_summary

    df_raw = load_data(csv_path) if csv_path else pd.DataFrame()

    # 시장 컨럼 없으면 pykrx로 자동 보완
    if not df_raw.empty and "시장" not in df_raw.columns:
        mmap = get_market_map()
        if mmap:
            df_raw["시장"] = df_raw["티커"].astype(str).map(mmap).fillna("")


    with col_meta:
        if not df_raw.empty and "수집시각" in df_raw.columns:
            last_update = df_raw["수집시각"].max()
            st.markdown(
                f"<div style='text-align:right; padding-top:16px;'>"
                f"<span class='collected-at'>기준시각: <b>{last_update}</b></span><br>"
                f"<span class='collected-at'>총 {len(df_raw)}행</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if st.button("🔄 데이터 새로고침"):
            st.cache_data.clear()
            st.rerun()

    st.divider()

    if df_raw.empty:
        st.warning("데이터가 없습니다. collector.py 를 먼저 실행하세요. (전체 수집: collector.py --full)")
        return

    # 점수 계산 (전체 데이터기준 백분위)
    df_scored = compute_scores(df_raw)

    # 전체 순위 (합산점수 내림차순, 동점 시 평균 랭크)
    df_scored["전체순위"] = (
        pd.to_numeric(df_scored["합산점수"], errors="coerce")
        .rank(method="min", ascending=False, na_option="bottom")
        .astype(int)
    )

    DISPLAY_COLS = [
        "전체순위", "종목명", "시장",
        "외인연속점수", "기관연속점수", "수급합점수", "조용한매집점수", "합산점수",
        "외인연속매수일", "기관연속매수일",
        "외인매수(억)", "기관매수(억)",
        "종가", "라이브PBR", "부채비율", "유보율", "영업이익률(%)",
        "매출증가율(%)",
        "V5/V20", "P5/P20",
        "52주위치(%)", "재무등급",
    ]

    def make_styled(source_df: pd.DataFrame):
        show_cols = [c for c in DISPLAY_COLS if c in source_df.columns]
        disp = source_df[show_cols].copy()

        # 종목명 바로 다음에 네이버 링크 컨럼 삽입
        if "티커" in source_df.columns and "종목명" in disp.columns:
            urls = "https://finance.naver.com/item/main.nhn?code=" + source_df["티커"].astype(str).str.zfill(6)
            ins_pos = disp.columns.get_loc("종목명") + 1
            disp.insert(ins_pos, "네이버", urls.values)

        # 외인/기관 매수 표시: "+43.28억(16일)" 형식
        for _bil, _day in [("외인매수(억)", "외인연속매수일"), ("기관매수(억)", "기관연속매수일")]:
            if _bil in disp.columns:
                def _fmt_buy(row, b=_bil, d=_day):
                    v = row.get(b)
                    nd = row.get(d, 0)
                    if v is None or (isinstance(v, float) and pd.isna(v)): return "-"
                    d_int = int(nd) if pd.notna(nd) and int(nd) > 0 else 0
                    suffix = f"({d_int}일)" if d_int > 0 else ""
                    return f"{v:+.1f}억{suffix}"
                disp[_bil] = source_df.apply(_fmt_buy, axis=1)

        def style_table(s):
            styles = pd.DataFrame("", index=s.index, columns=s.columns)
            if "52주위치(%)" in s.columns:
                styles["52주위치(%)"] = s["52주위치(%)"].apply(w52_color)
            if "재무등급" in s.columns:
                styles["재무등급"] = s["재무등급"].apply(grade_color)
            for col in ["외인연속점수", "기관연속점수", "수급합점수", "조용한매집점수", "합산점수"]:
                if col in s.columns:
                    styles[col] = s[col].apply(score_color)
            for col in ["외인연속매수일", "기관연속매수일"]:
                if col in s.columns:
                    def _buy_color(v, _col=col):
                        n = safe_num(v)
                        if n is None: return ""
                        if n >= 10: return "background-color:#1e3a5f; color:#93c5fd;"
                        if n >= 5:  return "background-color:#1c3a2e; color:#86efac;"
                        return ""
                    styles[col] = s[col].apply(_buy_color)
            return styles

        score_fmt = {c: (lambda v: f"{v:.1f}" if pd.notna(v) else "-")
                     for c in ["외인연속점수", "기관연속점수", "수급합점수", "조용한매집점수", "합산점수"]}
        return (
            disp.style
            .apply(style_table, axis=None)
            .format({
                "종가":        lambda v: f"{int(v):,}" if pd.notna(v) else "-",
                "외인매수(억)": lambda v: str(v),
                "기관매수(억)": lambda v: str(v),
                "라이브PBR":    lambda v: f"{v:.2f}" if pd.notna(v) else "-",
                "부채비율":      lambda v: f"{v:.1f}%" if pd.notna(v) else "-",
                "유보율":        lambda v: f"{v:.0f}%" if pd.notna(v) else "-",
                "영업이익률(%)": lambda v: f"{v:.1f}%" if pd.notna(v) else "-",
                "매출증가율(%)": lambda v: f"{v:+.1f}%" if pd.notna(v) else "-",
                "V5/V20": lambda v: f"{v:.1f}" if pd.notna(v) else "-",
                "P5/P20": lambda v: f"{v:.1f}" if pd.notna(v) else "-",
                "52주위치(%)": lambda v: f"{v:.1f}" if pd.notna(v) else "-",
                **score_fmt,
            }, na_rep="-")
            .set_properties(**{"text-align": "center", "font-size": "13px"})
            .set_table_styles([
                {"selector": "thead th", "props": [
                    ("background-color", "#1e293b"), ("color", "#93c5fd"),
                    ("font-weight", "700"), ("text-align", "center"),
                    ("padding", "8px 12px"), ("border-bottom", "2px solid #374151"),
                ]},
                {"selector": "tbody tr:hover", "props": [("background-color", "#1e293b")]},
                {"selector": "td", "props": [
                    ("padding", "6px 10px"), ("border-bottom", "1px solid #1f2937"),
                ]},
            ])
            .hide(axis="index")
        ), show_cols

    # ══════════════════════════════════════════
    # sidebar_filters는 한 번만 호출 (key 충돌 방지)
    df_top = sidebar_filters(df_scored)

    # 탭 구성
    # ══════════════════════════════════════════
    tab_top, tab_ai, tab_all, tab_search = st.tabs([
        "🏆 수급 상위",
        "🤖 AI 요약",
        f"📋 전체 ({len(df_scored):,}개)",
        "🔍 종목 검색",
    ])

    # ── Tab 1 : 수급 상위 ──────────────────────
    with tab_top:
        st.caption(
            f"※ 점수는 전체 {len(df_scored):,}개 종목 기준 백분위(0-10점). "
            "상위 20개이므로 점수가 9-10점대로 보이는 건 정상입니다."
        )

        # 메트릭
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("표시 종목", f"{len(df_top)}개", f"전체 {len(df_scored):,}개")
        avg_f  = pd.to_numeric(df_top["외인연속매수일"], errors="coerce").mean()
        avg_i  = pd.to_numeric(df_top["기관연속매수일"], errors="coerce").mean()
        avg_w  = pd.to_numeric(df_top["52주위치(%)"],   errors="coerce").mean()
        m2.metric("외인연속 평균", f"{avg_f:.1f}일"  if not pd.isna(avg_f)  else "-")
        m3.metric("기관연속 평균", f"{avg_i:.1f}일"  if not pd.isna(avg_i)  else "-")
        m4.metric("52주 평균",    f"{avg_w:.1f}%"   if not pd.isna(avg_w)  else "-")
        pbr_vals = pd.to_numeric(df_top.get("라이브PBR"), errors="coerce")
        avg_pbr  = pbr_vals.mean()
        m5.metric("라이브PBR 평균", f"{avg_pbr:.2f}" if not pd.isna(avg_pbr) else "-")

        st.divider()
        styled_top, show_top_cols = make_styled(df_top)
        st.dataframe(styled_top, use_container_width=True, height=640, column_config=SCORE_COL_CONFIG)

        csv_bytes = df_top[show_top_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ CSV 다운로드", csv_bytes,
                           file_name=f"top_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv", key="dl_top")

        with st.expander("📋 Excel 붙여넣기용 복사", expanded=False):
            st.code(df_top[show_top_cols].to_csv(sep="\t", index=False), language=None)

    # ── Tab 2 : AI 요약 ────────────────────────
    with tab_ai:
        # 현재 선택된 날짜 (사이드바 날짜 선택기에서 파싱)
        _sel_date = st.session_state.get("_sel_date_for_summary", "")

        summary_data = load_summary(_sel_date) if _sel_date else None

        # 생성/재생성 버튼
        btn_col, info_col = st.columns([1, 4])
        with btn_col:
            force_regen = st.button("🔄 요약 (재)생성", key="btn_ai_gen",
                                    help="LLM을 호출해 top20 종목 요약을 생성합니다.\nassistant LLM_API_KEY 설정 필요.")
        with info_col:
            if summary_data:
                _rec_cnt  = len(summary_data.get("summaries", {}))
                _skip_cnt = len(summary_data.get("not_recommended", {}))
                st.caption(
                    f"생성일시: {summary_data.get('generated_at', '')}  |  "
                    f"모델: {summary_data.get('model', '')}  |  "
                    f"추천 {_rec_cnt}종목 / 비추천 {_skip_cnt}종목"
                )
            else:
                st.caption("⚠️ 요약 파일 없음 — '요약 (재)생성' 버튼을 눌러 생성하세요.")

        if force_regen:
            summarize_script = Path(__file__).parent / "summarize.py"
            # 수급 상위 탭과 동일한 df_top 티커 목록 전달
            tickers_arg = ",".join(df_top["티커"].astype(str).str.zfill(6).tolist())
            # _sel_date가 비어있으면(latest) 날짜 인자 생략 → summarize.py가 자동 감지
            cmd = [sys.executable, str(summarize_script), "--force", "--tickers", tickers_arg]
            if _sel_date:
                cmd.insert(2, _sel_date)
            # Windows 한글 인코딩 문제 방지: PYTHONUTF8=1 환경변수 전달
            import os as _os
            _env = {**_os.environ, "PYTHONUTF8": "1"}
            with st.spinner("🤖 LLM으로 top20 종목 요약 중... (약 10~30초 소요)"):
                result = subprocess.run(
                    cmd,
                    capture_output=True, text=True, encoding="utf-8",
                    env=_env,
                )
            if result.returncode == 0:
                st.success("요약 생성 완료! 아래에서 확인하세요.")
                summary_data = load_summary(_sel_date)
            else:
                err_msg = result.stderr or result.stdout or "(출력 없음)"
                st.error(f"요약 생성 실패:\n```\n{err_msg}\n```")

        # 요약 카드 표시
        if summary_data and summary_data.get("summaries"):
            summaries      = summary_data["summaries"]
            not_recom      = summary_data.get("not_recommended", {})
            rec_items      = list(summaries.items())
            not_rec_items  = list(not_recom.items())

            # ── 매수 추천 종목 ──
            st.markdown(f"### ✅ 매수 추천 ({len(rec_items)}종목)")
            if rec_items:
                for row_start in range(0, len(rec_items), 2):
                    cols = st.columns(2)
                    for col_idx, (name, info) in enumerate(rec_items[row_start:row_start + 2]):
                        ticker    = info.get("ticker", "")
                        naver_url = f"https://finance.naver.com/item/main.nhn?code={ticker}"
                        biz       = info.get("비즈니스 모델", "")
                        momentum  = info.get("최근 모멘텀", "")
                        consensus = info.get("시장 컨센서스", "")
                        error     = info.get("error", "")

                        with cols[col_idx]:
                            with st.container(border=True):
                                st.markdown(
                                    f"**[{name}]({naver_url})** "
                                    f"<span style='color:#6b7280; font-size:12px;'>({ticker})</span>",
                                    unsafe_allow_html=True,
                                )
                                if error:
                                    st.warning(f"요약 오류: {error}")
                                else:
                                    st.markdown(
                                        f"💼 {biz}\n\n"
                                        f"📈 {momentum}\n\n"
                                        f"🎯 {consensus}"
                                    )
            else:
                st.info("추천 종목이 없습니다.")

            # ── 비추천 종목 ──
            if not_rec_items:
                st.markdown(f"### ❌ 비추천 ({len(not_rec_items)}종목)")
                rows = []
                for name, info in not_rec_items:
                    ticker  = info.get("ticker", "")
                    reasons = ", ".join(info.get("reasons", []))
                    rows.append({"종목명": name, "티커": ticker, "비추천 사유": reasons})
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "종목명": st.column_config.TextColumn("종목명", width="medium"),
                        "티커":   st.column_config.TextColumn("티커",   width="small"),
                        "비추천 사유": st.column_config.TextColumn("비추천 사유", width="large"),
                    }
                )
        else:
            st.info("요약 데이터가 없습니다. 위 버튼을 눌러 생성하세요.\n\n"
                    "**사전 준비**: `QuietAccumulation/llm_api_key.txt`에 API 키를 저장하고, "
                    "`llm_config.txt`에 BASE_URL·MODEL_NAME을 지정하세요.")

    # ── Tab 3 : 전체 ───────────────────────────
    with tab_all:
        st.caption(
            f"전체 {len(df_scored):,}개 종목 — 합산점수 내림차순. "
            "점수 분포 확인 및 중하위 종목 탐색에 활용하세요."
        )

        # 전체 정렬 기준
        sort_all = st.selectbox(
            "정렬 기준",
            ["합산점수", "외인연속점수", "기관연속점수", "수급합점수", "조용한매집점수",
             "외인연속매수일", "기관연속매수일", "52주위치(%)", "라이브PBR", "부채비율"],
            index=0, key="sort_all",
        )
        asc_all = st.checkbox("오름차순", value=False, key="asc_all")

        df_all = df_scored.copy()
        if sort_all in df_all.columns:
            df_all = df_all.sort_values(
                sort_all,
                key=lambda s: pd.to_numeric(s, errors="coerce"),
                ascending=asc_all, na_position="last",
            )
        df_all = df_all.reset_index(drop=True)

        styled_all, show_all_cols = make_styled(df_all)
        # st.dataframe 은 가상 스크롤 지원 → 전체 렌더링
        st.dataframe(styled_all, use_container_width=True, height=700, column_config=SCORE_COL_CONFIG)

        csv_all = df_all[show_all_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ 전체 CSV 다운로드", csv_all,
                           file_name=f"all_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv", key="dl_all")

        with st.expander("📋 Excel 붙여넣기용 복사", expanded=False):
            st.code(df_all[show_all_cols].to_csv(sep="\t", index=False), language=None)

    # ── Tab 3 : 종목 검색 ─────────────────────
    with tab_search:
        st.caption("종목명 또는 티커 코드로 검색합니다.")
        query = st.text_input("🔍 종목명 / 티커 검색", placeholder="예) 삼성전자  또는  005930")

        if query.strip():
            mask = (
                df_scored["종목명"].str.contains(query.strip(), na=False, case=False) |
                df_scored["티커"].astype(str).str.contains(query.strip(), na=False, case=False)
            )
            df_search = df_scored[mask].copy()
            df_search = df_search.sort_values(
                "합산점수",
                key=lambda s: pd.to_numeric(s, errors="coerce"),
                ascending=False, na_position="last",
            ).reset_index(drop=True)

            if df_search.empty:
                st.warning(f"'{query}' 검색 결과 없음")
            else:
                total = len(df_scored)
                st.success(f"'{query}' — {len(df_search)}개 종목 검색됨  (전체 {total:,}개 중 순위 표시)")
                styled_s, show_s_cols = make_styled(df_search)
                st.dataframe(styled_s, use_container_width=True, height=min(400, 60 + len(df_search) * 38), column_config=SCORE_COL_CONFIG)

                csv_s = df_search[show_s_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button("⬇️ 검색 결과 CSV", csv_s,
                                   file_name=f"search_{query}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                   mime="text/csv", key="dl_search")

                with st.expander("📋 Excel 붙여넣기용 복사", expanded=False):
                    st.code(df_search[show_s_cols].to_csv(sep="\t", index=False), language=None)
        else:
            st.info("검색어를 입력하면 해당 종목의 수급 점수를 확인할 수 있습니다.")

    # ── 자동 새로고침 ──
    st.markdown(
        "<div style='color:#6b7280; font-size:11px; text-align:right;'>60초마다 자동 갱신</div>",
        unsafe_allow_html=True,
    )
    time.sleep(1)


if __name__ == "__main__":
    main()
