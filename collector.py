"""
조용한 매집 - 데이터 수집기 v3

데이터 소스:
  - pykrx         : OHLCV (종가, 52주 고저, BPS)
  - 네이버 금융   : 투자자별 순매매 + PER/BPS  ← frgn.naver 단일 요청으로 통합
  - DART OpenAPI  : 부채비율, 유보율, 매출증가율, EPS증가율 (분기 재무제표)
                    환경변수 DART_API_KEY 또는 dart_api_key.txt 에 키 입력
                    키 없으면 Naver PER/PBR fallback 으로 재무등급 산출

저장:
  - data/YYYYMMDD.csv  한 파일씩 일별 누적 저장 (덮어쓰기 없음)
  - data/latest.csv    항상 최신 스냅샷 복사

수집 모드:
  - watchlist  : watchlist.json 에 정의된 종목만 (빠름)
  - full       : 코스피 + 코스닥 전체 상장사 (~2400개, 약 20~30분)

티커 검증:
  - 수집한 종목명과 Naver 페이지 실제 종목명 불일치 시 WARN 로그 출력
  - 완전히 다른 종목으로 판단되면 스킵
"""

import json
import logging
import os
import re
import time
from datetime import date, timedelta, datetime
from io import StringIO
from pathlib import Path

import urllib3
import requests
import pandas as pd
from pykrx import stock as pykrx_stock

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
WATCHLIST_JSON = Path(__file__).parent / "watchlist.json"

W52_LOOKBACK      = 252    # 52주 ≈ 252 거래일
REQUEST_TIMEOUT   = 12
REQUEST_DELAY_SEC = 0.25   # 종목 간 딜레이
PROGRESS_SAVE_N   = 50     # N 종목마다 중간 저장

# DART API 키 로드 (환경변수 우선, 없으면 dart_api_key.txt)
def _load_dart_key() -> str:
    key = os.environ.get("DART_API_KEY", "").strip().strip("\"'")
    if key:
        return key
    key_file = Path(__file__).parent / "dart_api_key.txt"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8-sig").strip().strip("\"'")
    return ""

DART_API_KEY = _load_dart_key()

NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer":    "https://finance.naver.com/",
    "Accept-Encoding": "gzip, deflate",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────
def today_str() -> str:
    return date.today().strftime("%Y%m%d")


def trade_date(offset_days: int = 0) -> str:
    return (date.today() - timedelta(days=offset_days)).strftime("%Y%m%d")


def daily_csv_path(date_str: str | None = None) -> Path:
    d = date_str or today_str()
    return DATA_DIR / f"{d}.csv"


def latest_csv_path() -> Path:
    return DATA_DIR / "latest.csv"


def consecutive_buy(series: pd.Series) -> int:
    """최근부터 연속 양수(순매수) 일수"""
    for i, v in enumerate(series.dropna().tolist()):
        if v <= 0:
            return i
    return len(series.dropna())


def consecutive_sell(series: pd.Series) -> int:
    """최근부터 연속 음수(순매도) 일수 — 중간매도기간"""
    for i, v in enumerate(series.dropna().tolist()):
        if v >= 0:
            return i
    return len(series.dropna())


def naver_get(url: str, params: dict) -> requests.Response:
    return requests.get(
        url, params=params, headers=NAVER_HEADERS,
        verify=False, timeout=REQUEST_TIMEOUT,
    )


def _extract_first_num(s: str) -> float | None:
    m = re.search(r"[\d.]+", str(s))
    return float(m.group()) if m else None


# ─────────────────────────────────────────────
# 티커↔종목명 검증
# ─────────────────────────────────────────────
_VERIFY_NAME_RE = re.compile(r'<title>([^<|]+?)\s*[:|]', re.IGNORECASE)

def verify_ticker_name(ticker: str, expected_name: str) -> bool:
    """
    Naver item/main.naver 에서 실제 종목명을 읽어 expected_name 과 비교.
    일치(부분 포함)하면 True. 완전히 다른 종목이면 False 반환 후 WARN 로그.
    """
    try:
        r = naver_get("https://finance.naver.com/item/main.naver", {"code": ticker})
        content = r.content.decode("euc-kr", errors="replace")
        m = _VERIFY_NAME_RE.search(content)
        if not m:
            return True  # 파싱 실패 시 통과 처리
        actual = m.group(1).strip()
        # EUC-KR 디코딩 실패(\ufffd 포함) 시 통과 처리
        if "\ufffd" in actual:
            return True
        # 공백 제거 후 포함 여부 비교 (양방향)
        a = actual.replace(" ", "")
        e = expected_name.replace(" ", "")
        if e in a or a in e:
            return True
        log.warning(f"[{ticker}] 종목명 불일치: 수집명='{expected_name}' / Naver실제='{actual}' → 스킵")
        return False
    except Exception:
        return True  # 네트워크 오류 시 통과 처리


# ─────────────────────────────────────────────
# 네이버 단일 요청으로 투자자 데이터 + 재무등급 통합 수집
# ─────────────────────────────────────────────
def fetch_naver_data(ticker: str) -> dict:
    """
    frgn.naver 한 번만 요청하여 반환:
      foreign_consec, inst_consec          : 연속 순매수일
      foreign_sell, inst_sell              : 중간매도기간
      last_foreign_shares, last_inst_shares: 연속매수 기간 누적 순매매량(주)
      per, pbr, bps                        : 재무 수치 (Naver)
      fin_grade                            : 재무등급 (DART 없을 때 fallback)
    """
    result = {
        "foreign_consec": 0, "inst_consec": 0,
        "foreign_sell": 0,   "inst_sell": 0,
        "last_foreign_shares": None, "last_inst_shares": None,
        "per": None, "pbr": None, "bps": None,
        "fin_grade": "-",
    }
    try:
        r = naver_get("https://finance.naver.com/item/frgn.naver", {"code": ticker})
        content = r.content.decode("euc-kr", errors="replace")
        tables  = pd.read_html(StringIO(content), flavor="lxml")
    except Exception as e:
        log.debug(f"[{ticker}] naver fetch 실패: {e}")
        return result

    for t in tables:
        # ── 투자자 순매매 테이블 (MultiIndex, shape ~31×9) ──
        if isinstance(t.columns, pd.MultiIndex):
            top  = [str(c[0]) for c in t.columns]
            bot  = [str(c[1]) for c in t.columns]
            flat = [f"{a}_{b}" if a != b else a for a, b in zip(top, bot)]
            t.columns = flat

        cols     = t.columns.tolist()
        date_col = next((c for c in cols if "날짜" in str(c)), None)
        if date_col is None:
            continue

        inst_col    = next((c for c in cols if "기관" in str(c) and "순매매" in str(c)), None)
        f_cols      = [c for c in cols if "외국인" in str(c) and "순매매" in str(c)]
        foreign_col = f_cols[0] if f_cols else None

        if inst_col is None and foreign_col is None:
            continue

        valid = t[
            t[date_col].astype(str).str.match(r"\d{4}\.\d{2}\.\d{2}", na=False)
        ].copy()
        if valid.empty:
            continue

        def to_num(col):
            return pd.to_numeric(
                valid[col].astype(str).str.replace(",", "").str.replace("+", ""),
                errors="coerce",
            )

        if foreign_col:
            fs = to_num(foreign_col).reset_index(drop=True)
            result["foreign_consec"]      = consecutive_buy(fs)
            result["foreign_sell"]        = consecutive_sell(fs)
            n_buy = result["foreign_consec"]
            if n_buy > 0:
                result["last_foreign_shares"] = float(fs.iloc[:n_buy].sum())
            else:
                non_na = fs.dropna()
                result["last_foreign_shares"] = float(non_na.iloc[0]) if not non_na.empty else None

        if inst_col:
            ins = to_num(inst_col).reset_index(drop=True)
            result["inst_consec"]         = consecutive_buy(ins)
            result["inst_sell"]           = consecutive_sell(ins)
            n_buy = result["inst_consec"]
            if n_buy > 0:
                result["last_inst_shares"] = float(ins.iloc[:n_buy].sum())
            else:
                non_na = ins.dropna()
                result["last_inst_shares"] = float(non_na.iloc[0]) if not non_na.empty else None

    # ── 재무등급 테이블 (shape 4×2: PER/PBR 행) ──
    for t in tables:
        if isinstance(t.columns, pd.MultiIndex):
            continue
        if t.shape != (4, 2):
            continue

        col0 = t.iloc[:, 0].astype(str)
        col1 = t.iloc[:, 1].astype(str)

        # PER 행이 없으면 (시가총액 등 다른 4×2 테이블) 건너뜀
        if not col0.str.contains("PER", na=False).any():
            continue

        # 추정PER 제외, 일반 PER 행
        per_mask = col0.str.contains("PER", na=False) & ~col0.str.contains("추정", na=False)
        pbr_mask = col0.str.contains("PBR", na=False)

        per_raw = col1[per_mask].iloc[0] if per_mask.any() else ""
        pbr_raw = col1[pbr_mask].iloc[0] if pbr_mask.any() else ""

        per = _extract_first_num(per_raw) if "N/A" not in per_raw else None
        pbr = _extract_first_num(pbr_raw) if "N/A" not in pbr_raw else None

        result["per"] = per
        result["pbr"] = pbr

        # BPS 추출 (PBRlBPS 행에서)
        bps_mask = col0.str.contains("BPS", na=False)
        if bps_mask.any():
            bps_raw = col1[bps_mask].iloc[0]
            # "N/A | 3,754원" 형태에서 숫자 추출
            bps_nums = re.findall(r"[\d,]+", str(bps_raw))
            if bps_nums:
                try:
                    result["bps"] = float(bps_nums[-1].replace(",", ""))
                except ValueError:
                    pass

        # Naver fallback 재무등급 (DART 키 없을 때 사용)
        score = 0
        if pbr and pbr > 0:
            score += 2 if pbr <= 1 else 1 if pbr <= 2 else 0
        if per and per > 0:
            score += 2 if per <= 10 else 1 if per <= 20 else 0

        result["fin_grade"] = (
            "S" if score >= 4
            else "A" if score == 3
            else "B" if score == 2
            else "C" if score == 1
            else "D"
        )
        break  # 첫 번째 (4,2) 테이블만 사용

    return result


# ─────────────────────────────────────────────
# DART 재무 지표 수집
# ─────────────────────────────────────────────
_DART_CORP_MAP: dict[str, str] = {}   # ticker → corp_code 캐시

# DART 재무 결과 캐시 (ticker → {data, cached_at})
# 재무데이터는 분기에 한 번만 바뀌므로 7일간 캐시 유지
_DART_FIN_CACHE: dict[str, dict] = {}
_DART_FIN_CACHE_FILE = Path(__file__).parent / "data" / "dart_fin_cache.json"
DART_CACHE_TTL_DAYS = 7  # 캐시 유효 기간

def _load_dart_fin_cache():
    global _DART_FIN_CACHE
    if _DART_FIN_CACHE:
        return
    if _DART_FIN_CACHE_FILE.exists():
        try:
            with open(_DART_FIN_CACHE_FILE, encoding="utf-8") as f:
                _DART_FIN_CACHE = json.load(f)
        except Exception:
            _DART_FIN_CACHE = {}

def _save_dart_fin_cache():
    try:
        with open(_DART_FIN_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_DART_FIN_CACHE, f, ensure_ascii=False)
    except Exception as e:
        log.debug(f"DART 캐시 저장 실패: {e}")

def _get_cached_dart_fin(ticker: str) -> dict | None:
    """캐시에서 DART 재무 데이터 반환. 만료됐거나 없으면 None."""
    _load_dart_fin_cache()
    entry = _DART_FIN_CACHE.get(ticker)
    if not entry:
        return None
    try:
        cached_at = datetime.fromisoformat(entry["cached_at"])
        if (datetime.now() - cached_at).days < DART_CACHE_TTL_DAYS:
            return entry["data"]
    except Exception:
        pass
    return None

def _set_cached_dart_fin(ticker: str, data: dict):
    _load_dart_fin_cache()
    _DART_FIN_CACHE[ticker] = {
        "data": data,
        "cached_at": datetime.now().isoformat(timespec="seconds"),
    }
    # 100종목마다 디스크에 flush (메모리 유실 방지)
    if len(_DART_FIN_CACHE) % 100 == 0:
        _save_dart_fin_cache()

def get_dart_fin_cache_stats() -> str:
    _load_dart_fin_cache()
    total = len(_DART_FIN_CACHE)
    valid = 0
    for entry in _DART_FIN_CACHE.values():
        try:
            if (datetime.now() - datetime.fromisoformat(entry.get("cached_at", "2000-01-01"))).days < DART_CACHE_TTL_DAYS:
                valid += 1
        except Exception:
            pass
    return f"DART 재무 캐시: {valid}/{total}개 유효 (TTL {DART_CACHE_TTL_DAYS}일)"

def _get_dart_corp_code(ticker: str) -> str | None:
    """DART 기업고유번호 조회 (corpCode.xml 캐시 사용)"""
    global _DART_CORP_MAP
    if _DART_CORP_MAP:
        return _DART_CORP_MAP.get(ticker)
    # 캐시 파일
    cache_file = Path(__file__).parent / "data" / "dart_corp_map.json"
    if cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as f:
                _DART_CORP_MAP = json.load(f)
            return _DART_CORP_MAP.get(ticker)
        except Exception:
            pass
    # DART corpCode.xml 다운로드 및 파싱
    try:
        import zipfile, xml.etree.ElementTree as ET
        from io import BytesIO
        r = requests.get(
            "https://opendart.fss.or.kr/api/corpCode.xml",
            params={"crtfc_key": DART_API_KEY},
            timeout=30,
        )
        with zipfile.ZipFile(BytesIO(r.content)) as z:
            xml_data = z.read(z.namelist()[0])
        root = ET.fromstring(xml_data)
        m: dict[str, str] = {}
        for item in root.iter("list"):
            stock_code = (item.findtext("stock_code") or "").strip()
            corp_code  = (item.findtext("corp_code")  or "").strip()
            if stock_code and corp_code:
                m[stock_code] = corp_code
        _DART_CORP_MAP = m
        cache_file.parent.mkdir(exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False)
        log.info(f"DART 기업코드 맵 로드 완료 ({len(m)}개)")
        return m.get(ticker)
    except Exception as e:
        log.warning(f"DART corpCode 로드 실패: {e}")
        return None


def fetch_dart_financials(ticker: str) -> dict:
    """
    DART API로 최근 연간 재무제표에서:
      debt_ratio   : 부채비율 (부채총계/자본총계 × 100)
      retention    : 유보율   (이익잉여금/납입자본금 × 100)
      revenue_growth: 매출증가율 (전년 대비 %)
      eps_growth   : EPS증가율 (전년 대비 %)
    DART_API_KEY 없거나 실패 시 모두 None 반환.
    """
    empty = {"debt_ratio": None, "retention": None,
             "revenue_growth": None, "eps_growth": None,
             "roe": None, "op_margin": None}
    if not DART_API_KEY:
        return empty

    # 캐시 확인 (TTL 7일 — 재무데이터는 분기 1회만 변경)
    cached = _get_cached_dart_fin(ticker)
    if cached is not None:
        return cached

    try:
        corp_code = _get_dart_corp_code(ticker)
        if not corp_code:
            _set_cached_dart_fin(ticker, empty)
            return empty

        year = date.today().year
        # 전년도 사업보고서(11011)부터 시도, 없으면 전전년
        for bsns_year in [str(year - 1), str(year - 2)]:
            r = requests.get(
                "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json",
                params={
                    "crtfc_key": DART_API_KEY,
                    "corp_code": corp_code,
                    "bsns_year": bsns_year,
                    "reprt_code": "11011",   # 사업보고서
                    "fs_div": "CFS",         # 연결재무제표 우선
                },
                timeout=REQUEST_TIMEOUT,
            )
            data = r.json()
            if data.get("status") != "000" or not data.get("list"):
                # 연결 없으면 별도재무제표 시도
                r2 = requests.get(
                    "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json",
                    params={
                        "crtfc_key": DART_API_KEY,
                        "corp_code": corp_code,
                        "bsns_year": bsns_year,
                        "reprt_code": "11011",
                        "fs_div": "OFS",
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                data = r2.json()
            if data.get("status") != "000" or not data.get("list"):
                continue

            items = {d["account_nm"]: d for d in data["list"]}

            def amt(key: str) -> float | None:
                row = items.get(key)
                if not row:
                    return None
                try:
                    return float(str(row.get("thstrm_amount", "")).replace(",", ""))
                except ValueError:
                    return None

            def amt_prev(key: str) -> float | None:
                row = items.get(key)
                if not row:
                    return None
                try:
                    return float(str(row.get("frmtrm_amount", "")).replace(",", ""))
                except ValueError:
                    return None

            # 부채비율
            total_liab   = amt("부채총계")
            total_equity = amt("자본총계")
            debt_ratio = round(total_liab / total_equity * 100, 1) if (total_liab and total_equity and total_equity != 0) else None

            # 유보율 (이익잉여금 / 납입자본금)
            retained = amt("이익잉여금") or amt("미처분이익잉여금")
            paid_in  = amt("납입자본금") or amt("자본금")
            retention = round(retained / paid_in * 100, 1) if (retained and paid_in and paid_in != 0) else None

            # 매출증가율
            rev_cur  = amt("매출액") or amt("영업수익")
            rev_prev = amt_prev("매출액") or amt_prev("영업수익")
            revenue_growth = round((rev_cur - rev_prev) / abs(rev_prev) * 100, 1) if (rev_cur and rev_prev and rev_prev != 0) else None

            # EPS (당기순이익 / 주식수 근사치 — DART 직접 EPS 항목 우선)
            eps_row = items.get("기본주당이익(손실)") or items.get("기본주당순이익")
            if eps_row:
                try:
                    eps_cur  = float(str(eps_row.get("thstrm_amount", "")).replace(",", ""))
                    eps_prev = float(str(eps_row.get("frmtrm_amount", "")).replace(",", ""))
                    eps_growth = round((eps_cur - eps_prev) / abs(eps_prev) * 100, 1) if eps_prev != 0 else None
                except (ValueError, ZeroDivisionError):
                    eps_growth = None
            else:
                eps_growth = None

            # ROE = 당기순이익 / 자본총계 × 100
            net_income = amt("당기순이익") or amt("분기순이익")
            roe = round(net_income / total_equity * 100, 1) if (net_income and total_equity and total_equity != 0) else None

            # 영업이익률 = 영업이익 / 매출액 × 100
            op_income = amt("영업이익") or amt("영업손익")
            op_margin = round(op_income / rev_cur * 100, 1) if (op_income and rev_cur and rev_cur != 0) else None

            result = {
                "debt_ratio":     debt_ratio,
                "retention":      retention,
                "revenue_growth": revenue_growth,
                "eps_growth":     eps_growth,
                "roe":            roe,
                "op_margin":      op_margin,
            }
            _set_cached_dart_fin(ticker, result)
            return result
        _set_cached_dart_fin(ticker, empty)
        return empty
    except Exception as e:
        log.debug(f"[{ticker}] DART 재무 실패: {e}")
        return empty


def _dart_grade(dart: dict, naver_grade: str) -> str:
    """
    DART 4지표로 재무등급 산출. 데이터 없으면 Naver fallback.
    기준:
      부채비율  ≤100  → +2, ≤200 → +1
      유보율    ≥500  → +2, ≥100 → +1
      매출증가율 ≥10  → +2,  ≥0  → +1
      EPS증가율  ≥10  → +2,  ≥0  → +1
      ROE       ≥15  → +1
      영업이익률 ≥10  → +1
    최대 10점: S≥8, A≥6, B≥4, C≥2, D<2
    """
    d  = dart.get("debt_ratio")
    r  = dart.get("retention")
    mg = dart.get("revenue_growth")
    eg = dart.get("eps_growth")
    roe = dart.get("roe")
    opm = dart.get("op_margin")

    if all(v is None for v in [d, r, mg, eg, roe, opm]):
        return naver_grade  # fallback

    score = 0
    if d  is not None: score += 2 if d  <= 100 else 1 if d  <= 200 else 0
    if r  is not None: score += 2 if r  >= 500 else 1 if r  >= 100 else 0
    if mg is not None: score += 2 if mg >= 10  else 1 if mg >= 0   else 0
    if eg is not None: score += 2 if eg >= 10  else 1 if eg >= 0   else 0
    if roe is not None: score += 1 if roe >= 15 else 0
    if opm is not None: score += 1 if opm >= 10 else 0

    return (
        "S" if score >= 8
        else "A" if score >= 6
        else "B" if score >= 4
        else "C" if score >= 2
        else "D"
    )


# ─────────────────────────────────────────────
# DART 자사주 취득 현황
# ─────────────────────────────────────────────
def fetch_dart_treasury(ticker: str) -> dict:
    """
    DART 자기주식 취득·처분 현황 API (tesstkAcqsDspsSttus)
    반환:
      treasury_buy_qty  : 최근 취득 주수 (없으면 0)
      treasury_buy_amt  : 최근 취득 금액 (억원, 없으면 0)
      treasury_signal   : True = 최근 1년간 자사주 취득 이력 있음
    """
    empty = {"treasury_buy_qty": 0, "treasury_buy_amt": 0.0, "treasury_signal": False}
    if not DART_API_KEY:
        return empty
    try:
        corp_code = _get_dart_corp_code(ticker)
        if not corp_code:
            return empty

        year = date.today().year
        total_qty = 0
        total_amt = 0.0
        found = False

        for bsns_year in [str(year - 1), str(year - 2)]:
            r = requests.get(
                "https://opendart.fss.or.kr/api/tesstkAcqsDspsSttus.json",
                params={
                    "crtfc_key": DART_API_KEY,
                    "corp_code": corp_code,
                    "bgn_de": f"{bsns_year}0101",
                    "end_de": f"{bsns_year}1231",
                },
                timeout=REQUEST_TIMEOUT,
            )
            data = r.json()
            if data.get("status") != "000" or not data.get("list"):
                continue
            for item in data["list"]:
                acqs_mth = str(item.get("acqs_mth1", "") or "")  # 취득방법
                if "취득" not in acqs_mth and "매수" not in acqs_mth:
                    continue
                try:
                    qty = int(str(item.get("acqs_qty", "0") or "0").replace(",", ""))
                    amt_val = float(str(item.get("acqs_amt", "0") or "0").replace(",", ""))
                    total_qty += qty
                    total_amt += amt_val
                    if qty > 0:
                        found = True
                except (ValueError, TypeError):
                    pass
            if found:
                break  # 최근 연도에 이력 있으면 중단

        return {
            "treasury_buy_qty": total_qty,
            "treasury_buy_amt": round(total_amt / 1e8, 2),  # 원 → 억원
            "treasury_signal":  found,
        }
    except Exception as e:
        log.debug(f"[{ticker}] DART 자사주 실패: {e}")
        return empty


# ─────────────────────────────────────────────
# OHLCV (pykrx)
# ─────────────────────────────────────────────
def fetch_ohlcv(ticker: str) -> dict:
    close = high_52w = low_52w = w52_pos = None
    v5v20 = p5p20 = None
    try:
        end_dt   = trade_date(0)
        start_dt = trade_date(W52_LOOKBACK * 2)
        ohlcv = pykrx_stock.get_market_ohlcv_by_date(start_dt, end_dt, ticker)
        if ohlcv.empty:
            return {}
        ohlcv_52 = ohlcv.tail(W52_LOOKBACK)
        close    = float(ohlcv["종가"].iloc[-1])
        high_52w = float(ohlcv_52["고가"].max())
        low_52w  = float(ohlcv_52["저가"].min())
        w52_pos  = (
            round((close - low_52w) / (high_52w - low_52w) * 100, 1)
            if high_52w != low_52w else 50.0
        )
        # V5/V20: 거래대금 MA5/MA20 * 100
        vol = ohlcv["거래대금"] if "거래대금" in ohlcv.columns else ohlcv["거래량"]
        if len(vol) >= 20:
            ma5  = float(vol.iloc[-5:].mean())
            ma20 = float(vol.iloc[-20:].mean())
            v5v20 = round(ma5 / ma20 * 100, 1) if ma20 > 0 else None
        # P5/P20: 종가 MA5/MA20 * 100
        price = ohlcv["종가"]
        if len(price) >= 20:
            pma5  = float(price.iloc[-5:].mean())
            pma20 = float(price.iloc[-20:].mean())
            p5p20 = round(pma5 / pma20 * 100, 1) if pma20 > 0 else None
    except Exception as e:
        log.debug(f"[{ticker}] OHLCV 실패: {e}")
    return {"close": close, "high_52w": high_52w, "low_52w": low_52w,
            "w52_pos": w52_pos, "v5v20": v5v20, "p5p20": p5p20}


# ─────────────────────────────────────────────
# 단일 종목 수집
# ─────────────────────────────────────────────
def fetch_ticker_data(ticker: str, name: str) -> dict:
    nv   = fetch_naver_data(ticker)
    time.sleep(REQUEST_DELAY_SEC)
    ov   = fetch_ohlcv(ticker)
    dart = fetch_dart_financials(ticker) if DART_API_KEY else {"debt_ratio": None, "retention": None, "revenue_growth": None, "op_margin": None}

    close = ov.get("close")

    def to_bil(shares):
        if shares is None or close is None:
            return None
        return round(shares * close / 1e8, 2)

    # 라이브 PBR: 현재 종가 / BPS
    bps = nv.get("bps")
    live_pbr = round(close / bps, 2) if (close and bps and bps > 0) else nv.get("pbr")

    fin_grade = _dart_grade(dart, nv["fin_grade"])

    return {
        "No":               None,
        "종목명":           name,
        "티커":             ticker,
        "외인연속매수일":    nv["foreign_consec"],
        "기관연속매수일":    nv["inst_consec"],
        "외인매수(억)":     to_bil(nv["last_foreign_shares"]),
        "기관매수(억)":     to_bil(nv["last_inst_shares"]),
        "중간매도기간(외)":  nv["foreign_sell"],
        "중간매도기간(기)":  nv["inst_sell"],
        "종가":             close,
        "라이브PBR":        live_pbr,
        "부채비율":          dart.get("debt_ratio"),
        "유보율":            dart.get("retention"),
        "영업이익률(%)":    dart.get("op_margin"),
        "매출증가율(%)":    dart.get("revenue_growth"),
        "V5/V20":           ov.get("v5v20"),
        "P5/P20":           ov.get("p5p20"),
        "52주위치(%)":      ov.get("w52_pos"),
        "재무등급":         fin_grade,
        "수집시각":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────
# 전체 상장사 티커 목록 (네이버 시세 페이지 파싱)
# ─────────────────────────────────────────────
def _get_naver_market_tickers(sosok: int) -> list[dict]:
    """
    sosok=0: KOSPI, sosok=1: KOSDAQ
    네이버 시장 시세 요약 페이지에서 전 종목 코드+이름 추출
    """
    items = []
    seen  = set()
    page  = 1
    # 하나의 <a> 태그에서 코드+이름을 함께 추출 → 순서 불일치 없음
    PAIR_RE = re.compile(
        r'href="/item/main\.naver\?code=(\d{6})"[^>]*>\s*([^<\n]+?)\s*</a>',
        re.IGNORECASE,
    )
    while True:
        try:
            r = naver_get(
                "https://finance.naver.com/sise/sise_market_sum.naver",
                {"sosok": sosok, "page": page},
            )
            content = r.content.decode("euc-kr", errors="replace")

            pairs_found = PAIR_RE.findall(content)  # [(code, name), ...]

            new_count = 0
            for code, name in pairs_found:
                name = name.strip()
                if code not in seen and name:
                    seen.add(code)
                    items.append({"ticker": code, "name": name})
                    new_count += 1

            if new_count == 0:
                break  # 마지막 페이지 이후 (새 종목 없음)

            # 다음 페이지 링크가 없으면 종료
            if f'page={page + 1}' not in content:
                break
            page += 1
            time.sleep(0.1)
        except Exception as e:
            log.warning(f"시장 목록 페이지 {page} 실패: {e}")
            break
    return items


def get_all_tickers() -> list[dict]:
    """코스피 + 코스닥 전 종목 반환 (시장 구분 포함)"""
    log.info("코스피 종목 목록 수집...")
    kospi  = _get_naver_market_tickers(0)
    for item in kospi:
        item["market"] = "KOSPI"
    log.info(f"  \u2192 {len(kospi)}개")
    log.info("코스닥 종목 목록 수집...")
    kosdaq = _get_naver_market_tickers(1)
    for item in kosdaq:
        item["market"] = "KOSDAQ"
    log.info(f"  → {len(kosdaq)}개")
    return kospi + kosdaq

# ─────────────────────────────────────────────
# 워치리스트 로드
# ─────────────────────────────────────────────
def load_watchlist() -> list[dict]:
    if WATCHLIST_JSON.exists():
        with open(WATCHLIST_JSON, encoding="utf-8") as f:
            items = json.load(f)
        # market 필드 없으면 전체 목록으로 판별 (1회만 조회)
        if items and "market" not in items[0]:
            log.info("watchlist market 필드 없음 → 시장 구분 자동 판별...")
            all_tickers = get_all_tickers()
            market_map = {it["ticker"]: it.get("market", "") for it in all_tickers}
            for item in items:
                item["market"] = market_map.get(item["ticker"], "")
        return items
    log.warning("watchlist.json 없음 — 코스피 상위 50 자동 수집 시도")
    return []


# ─────────────────────────────────────────────
# 전체 스냅샷 수집
# ─────────────────────────────────────────────
COL_ORDER = [
    "No", "종목명", "티커", "시장",
    "외인연속매수일", "기관연속매수일",
    "외인매수(억)", "기관매수(억)",
    "중간매도기간(외)", "중간매도기간(기)",
    "종가", "라이브PBR", "부채비율", "유보율", "영업이익률(%)",
    "매출증가율(%)",
    "V5/V20", "P5/P20",
    "52주위치(%)", "재무등급", "수집시각",
]


def collect_snapshot(mode: str = "watchlist") -> pd.DataFrame:
    """
    mode = 'watchlist' : watchlist.json 만 수집 (기본, 빠름)
    mode = 'full'      : 코스피 + 코스닥 전체 (느림, 약 20-30분)
    """
    log.info(f"=== 스냅샷 수집 시작 [mode={mode}] ===")

    watchlist = get_all_tickers() if mode == "full" else load_watchlist()
    log.info(f"수집 대상: {len(watchlist)}개 종목")

    target_csv = daily_csv_path()   # data/YYYYMMDD.csv
    rows: list[dict] = []

    for i, item in enumerate(watchlist, 1):
        ticker = item["ticker"]
        name   = item.get("name", ticker)
        market = item.get("market", "")
        log.info(f"[{i}/{len(watchlist)}] {name} ({ticker})")

        # 티커↔종목명 검증 (full 모드 또는 watchlist 모드 공통)
        if not verify_ticker_name(ticker, name):
            log.warning(f"[{ticker}] '{name}' 종목명 불일치 \u2192 스킵")
            continue

        try:
            row = fetch_ticker_data(ticker, name)
        except Exception as e:
            log.warning(f"[{ticker}] 수집 오류: {e}")
            row = {
                "종목명": name, "티커": ticker,
                "수집시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        row["No"] = i
        if market:
            row["시장"] = market
        rows.append(row)

        # 중간 저장 (중단 시 복구용)
        if i % PROGRESS_SAVE_N == 0:
            _save_df(pd.DataFrame(rows), target_csv)
            log.info(f"  중간 저장 ({i}개)")

    df = pd.DataFrame(rows)
    df = df[[c for c in COL_ORDER if c in df.columns]]

    _save_df(df, target_csv)
    _save_df(df, latest_csv_path())

    # DART 재무 캐시 최종 flush
    _save_dart_fin_cache()
    log.info(get_dart_fin_cache_stats())
    log.info(f"저장 완료 → {target_csv}  ({len(df)}개 종목)")
    return df


def _save_df(df: pd.DataFrame, path: Path):
    ordered = [c for c in COL_ORDER if c in df.columns]
    df[ordered].to_csv(path, index=False, encoding="utf-8-sig")


# ─────────────────────────────────────────────
# 직접 실행
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mode = "full" if "--full" in sys.argv else "watchlist"
    collect_snapshot(mode=mode)
