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
    """data/YYYYMMDD_summary.json 로드. date_str 없으면 latest_summary.json fallback."""
    if date_str:
        path = DATA_DIR / f"{date_str}_summary.json"
    else:
        path = DATA_DIR / "latest_summary.json"
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
        for c in ['외인연속점수', '기관연속점수', '수급합점수', '조용한매집점수', '엔벨점수', '합산점수']:
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

    # 4) 조용한매집점수 (수급강도 60% + 거래량신호 25% + 주가안정 15%)
    v5v20 = pd.to_numeric(
        df['V5/V20'] if 'V5/V20' in df.columns else pd.Series(np.nan, index=df.index),
        errors='coerce'
    ).fillna(100)
    p5p20 = pd.to_numeric(
        df['P5/P20'] if 'P5/P20' in df.columns else pd.Series(np.nan, index=df.index),
        errors='coerce'
    ).fillna(100)

    # ① 수급강도 (max 6.0): 외인/기관 연속매수 강도 × 자격 가산점
    buy_strength = ((f + inst) / 30).clip(0, 1)
    qualify = np.where(
        (f >= 3) & (inst >= 3), 1.0,
        np.where((f >= 3) | (inst >= 3), 0.6, 0.0)
    )
    supply_score = pd.Series(buy_strength * qualify * 6.0, index=df.index)

    # ② 거래량신호 (max 2.5): V5/V20 100→200 선형 증가, 200 이상 cap
    vol_score = pd.Series(
        np.clip((v5v20.values - 100) / 100 * 2.5, 0, 2.5), index=df.index
    )

    # ③ 주가안정 (max 1.5): P5/P20 97~105 구간 최고점, 범위 밖 감점
    _p = p5p20.values
    price_score = pd.Series(np.where(
        _p < 93,   0.0,
        np.where(_p < 97,  (_p - 93) / 4.0 * 1.5,
        np.where(_p <= 105, 1.5,
        np.where(_p <= 112, (112 - _p) / 7.0 * 1.5, 0.0)))
    ), index=df.index)

    raw_quiet = supply_score + vol_score + price_score  # max 10.0
    df['조용한매집점수'] = pct_score(raw_quiet)

    # 5) 엔벨점수 (이평배열 + 엔벨상태 + 골든크로스 + 엔벨타점 → 백분위)
    _align_map = {"완전정배열": 4.0, "상향전환": 2.0, "하향전환": 1.0, "완전역배열": 0.0, "혼합": 1.0}
    _env_map   = {"상단돌파": 2.0, "중립": 1.0, "하단이탈": 0.0}

    _align_raw = (df["이평배열"] if "이평배열" in df.columns
                  else pd.Series("-", index=df.index)).map(_align_map).fillna(1.0)
    _env_raw   = (df["엔벨상태"] if "엔벨상태" in df.columns
                  else pd.Series("중립", index=df.index)).map(_env_map).fillna(1.0)

    def _bool_col(col: str) -> pd.Series:
        s = df[col] if col in df.columns else pd.Series(False, index=df.index)
        return s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).fillna(0.0)

    _gc_bonus  = _bool_col("골든크로스") * 2.0
    _sig_bonus = _bool_col("엔벨타점")   * 1.0

    _raw_env = (_align_raw + _env_raw + _gc_bonus + _sig_bonus).clip(0, 10)
    df['엔벨점수'] = pct_score(_raw_env)

    # 합산점수 (max 2.55 + min 1.70 + 수급합 2.55 + 조용한 1.70 + 엔벨 1.50 = max 100)
    df['합산점수'] = (
        df[['외인연속점수', '기관연속점수']].max(axis=1) * 2.55 +
        df[['외인연속점수', '기관연속점수']].min(axis=1) * 1.70 +
        df['수급합점수']     * 2.55 +
        df['조용한매집점수'] * 1.70 +
        df['엔벨점수']       * 1.50
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
    "종목명": st.column_config.LinkColumn(
        "종목명",
        display_text=r"#(.+)$",
        width="medium",
        pinned=True,
        help="클릭 시 네이버 금융으로 이동",
    ),
    "시장": st.column_config.TextColumn("시장", width="small", help="KOSPI / KOSDAQ"),
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
            "📌 조용한 장기 매집 지표 (0-10점, 백분위)\n"
            "① 수급강도 (60%): (외인+기관 연속매수)/30 × 자격(둘다≥3→×1.0, 하나→×0.6)\n"
            "② 거래량신호 (25%): V5/V20 100→200 선형 증가 (가격 안 오르는데 볼륨↑)\n"
            "③ 주가안정 (15%): P5/P20 97~105 구간 최고점 (과열 없이 조용히 오름)"
        ),
    ),
    "엔벨점수": st.column_config.NumberColumn(
        "엔벨점수",
        format="%.1f",
        help=(
            "📌 엔벨로프 기술적 모멘텀 점수 (0-10점)\n"
            "이평배열(MA5/20/120) + 엔벨상태(MA20±9%) + 골든크로스/타점 보너스\n"
            "완전정배열=4pt, 상단돌파=2pt, 골든크로스=+2pt, 타점=+1pt → 백분위 환산"
        ),
    ),
    "합산점수": st.column_config.NumberColumn(
        "합산점수",
        format="%.1f",
        help=(
            "📌 종합 점수 (최대 100점)\n"
            "= max(외인,기관) × 2.55 + min(외인,기관) × 1.70\n"
            "+ 수급합점수 × 2.55 + 조용한매집 × 1.70 + 엔벨점수 × 1.50"
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

    # 이평배열 / 엔벨 타점 필터 (해당 컬럼이 있을 때만)
    if "이평배열" in df.columns:
        st.sidebar.markdown("### 이평배열 / 엔벨")
        align_opts = ["전체", "완전정배열", "상향전환", "하향전환", "완전역배열", "혼합"]
        sel_align = st.sidebar.selectbox("이평배열", align_opts, index=0, key="sb_align")
        if sel_align != "전체":
            fdf = fdf[fdf["이평배열"] == sel_align]
        only_signal = st.sidebar.checkbox("엔벨 타점만", value=False, key="sb_env_signal")
        if only_signal and "엔벨타점" in fdf.columns:
            fdf = fdf[fdf["엔벨타점"].isin([True, "True"])]

    # 정렬 기준
    sort_col = st.sidebar.selectbox(
        "정렬 기준",
        ["합산점수", "외인연속점수", "기관연속점수", "수급합점수", "조용한매집점수", "엔벨점수",
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
    # latest.csv 항상 맨 위에 추가
    if (DATA_DIR / "latest.csv").exists():
        available = ["latest"] + available
    if not available:
        available = ["(없음)"]

    def _date_label(d: str) -> str:
        if d == "latest":
            p = DATA_DIR / "latest.csv"
            try:
                n = sum(1 for _ in open(p, encoding="utf-8-sig")) - 1
            except Exception:
                n = 0
            suffix = f" [전체 {n:,}종목]" if n > 20 else f" [워치리스트 {n}종목]"
            return f"latest (최신){suffix}"
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
        "외인연속점수", "기관연속점수", "수급합점수", "조용한매집점수", "엔벨점수", "합산점수",
        "외인연속매수일", "기관연속매수일",
        "외인매수(억)", "기관매수(억)",
        "종가", "라이브PBR", "부채비율", "유보율", "영업이익률(%)",
        "매출증가율(%)",
        "V5/V20", "P5/P20",
        "이평배열", "엔벨상태", "엔벨타점",
        "52주위치(%)", "재무등급",
    ]

    def make_styled(source_df: pd.DataFrame):
        show_cols = [c for c in DISPLAY_COLS if c in source_df.columns]
        disp = source_df[show_cols].copy()

        # 종목명을 네이버 링크로 변환 (URL#종목명 → display_text 정규식으로 종목명만 표시)
        if "티커" in source_df.columns and "종목명" in disp.columns:
            disp["종목명"] = (
                "https://finance.naver.com/item/main.nhn?code="
                + source_df["티커"].astype(str).str.zfill(6)
                + "#" + source_df["종목명"].astype(str)
            ).values

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
            for col in ["외인연속점수", "기관연속점수", "수급합점수", "조용한매집점수", "엔벨점수", "합산점수"]:
                if col in s.columns:
                    styles[col] = s[col].apply(score_color)
            if "이평배열" in s.columns:
                def _align_color(v):
                    v = str(v)
                    if v == "완전정배열": return "background-color:#15803d; color:#bbf7d0; font-weight:700;"
                    if v == "상향전환":   return "background-color:#1e3a5f; color:#93c5fd;"
                    if v == "하향전환":   return "background-color:#78350f; color:#fde68a;"
                    if v == "완전역배열": return "background-color:#7f1d1d; color:#fca5a5;"
                    return ""
                styles["이평배열"] = s["이평배열"].apply(_align_color)
            if "엔벨상태" in s.columns:
                def _env_color(v):
                    v = str(v)
                    if v == "상단돌파": return "background-color:#fca5a5; color:#7f1d1d; font-weight:700;"
                    if v == "하단이탈": return "background-color:#bfdbfe; color:#1e3a5f; font-weight:700;"
                    return ""
                styles["엔벨상태"] = s["엔벨상태"].apply(_env_color)
            if "엔벨타점" in s.columns:
                def _signal_color(v):
                    if v in (True, "True", "✓"):
                        return "background-color:#ca8a04; color:#fefce8; font-weight:700;"
                    return ""
                styles["엔벨타점"] = s["엔벨타점"].apply(_signal_color)
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
                     for c in ["외인연속점수", "기관연속점수", "수급합점수", "조용한매집점수", "엔벨점수", "합산점수"]}
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
                "엔벨타점": lambda v: "✓" if v in (True, "True", 1, "1") else "-",
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
            from summarize import summarize_stock_stream, screen_stocks_llm, load_top20, _load_llm_config
            import os as _os

            api_key, base_url, model_name = _load_llm_config()
            if not api_key:
                st.error("LLM API 키가 설정되지 않았습니다. Streamlit Secrets 또는 llm_api_key.txt를 확인하세요.")
            else:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key, base_url=base_url)
                except ImportError:
                    st.error("openai 패키지가 필요합니다.")
                    client = None

                if client:
                    tickers_list = df_top["티커"].astype(str).str.zfill(6).tolist()
                    try:
                        df_for_sum = load_top20(_sel_date or "latest", tickers=tickers_list)
                        rec_df, skip_df, skip_reasons = screen_stocks_llm(df_for_sum, client, model_name)
                    except Exception as e:
                        st.error(f"데이터 로드 실패: {e}")
                        rec_df, skip_df, skip_reasons = pd.DataFrame(), pd.DataFrame(), {}

                    summaries = {}
                    not_recommended = {
                        str(row.get("종목명", "")): {
                            "ticker": str(row.get("티커", "")),
                            "reasons": skip_reasons.get(str(row.get("종목명", "")), []),
                        }
                        for _, row in skip_df.iterrows()
                    }

                    st.markdown(f"### ✅ 매수 추천 ({len(rec_df)}종목) — 실시간 생성 중...")
                    for col_start in range(0, len(rec_df), 2):
                        cols = st.columns(2)
                        for col_idx, (_, stock_row) in enumerate(rec_df.iloc[col_start:col_start+2].iterrows()):
                            name   = str(stock_row.get("종목명", ""))
                            ticker = str(stock_row.get("티커", ""))
                            naver_url = f"https://finance.naver.com/item/main.nhn?code={ticker}"
                            with cols[col_idx]:
                                with st.container(border=True):
                                    st.markdown(
                                        f"**[{name}]({naver_url})** "
                                        f"<span style='color:#6b7280; font-size:12px;'>({ticker})</span>",
                                        unsafe_allow_html=True,
                                    )
                                    stream_placeholder = st.empty()
                                    streamed_text = ""
                                    parsed = {}
                                    for chunk in summarize_stock_stream(client, model_name, name, ticker, row=stock_row.to_dict()):
                                        if isinstance(chunk, dict):
                                            parsed = chunk
                                        else:
                                            streamed_text += chunk
                                            stream_placeholder.markdown(streamed_text + "▌")
                                    stream_placeholder.empty()
                                    if parsed.get("error"):
                                        st.warning(f"요약 오류: {parsed['error']}")
                                    else:
                                        st.markdown(
                                            f"💼 {parsed.get('비즈니스 모델','')}\n\n"
                                            f"📈 {parsed.get('최근 모멘텀','')}\n\n"
                                            f"✅ {parsed.get('추천 이유','')}\n\n"
                                            f"⚠️ {parsed.get('리스크','')}"
                                        )
                                    summaries[name] = {"ticker": ticker, **parsed}
                            if col_start + col_idx < len(rec_df) - 2:
                                time.sleep(4)

                    # JSON 저장 + git push
                    import json as _json, datetime as _dt
                    payload = {
                        "date": _sel_date or "latest",
                        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                        "model": model_name,
                        "summaries": summaries,
                        "not_recommended": not_recommended,
                    }
                    out_name = "latest_summary.json" if not _sel_date else f"{_sel_date}_summary.json"
                    out_path = DATA_DIR / out_name
                    out_path.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    summary_data = payload
                    st.success("✅ 요약 저장 완료!")

                    if not_recommended:
                        st.markdown(f"### ❌ 비추천 ({len(not_recommended)}종목)")
                        rows_nr = [{"종목명": n, "티커": i.get("ticker",""), "비추천 사유": ", ".join(i.get("reasons",[]))}
                                   for n, i in not_recommended.items()]
                        st.dataframe(pd.DataFrame(rows_nr), use_container_width=True, hide_index=True)

        # 요약 카드 표시 (스트리밍 생성 직후는 이미 위에서 렌더링됐으므로 스킵)
        if not force_regen and summary_data and summary_data.get("summaries"):
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
                        reason    = info.get("추천 이유", "")
                        risk      = info.get("리스크", "")
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
                                        f"✅ {reason}\n\n"
                                        f"⚠️ {risk}"
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
            ["합산점수", "외인연속점수", "기관연속점수", "수급합점수", "조용한매집점수", "엔벨점수",
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
