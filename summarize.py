"""
AI 요약 생성기 (summarize.py)
- data/YYYYMMDD.csv 에서 합산점수 상위 20개 종목을 뽑아 LLM으로 3줄 요약 생성
- 결과: data/YYYYMMDD_summary.json

LLM 설정 (우선순위):
  1) llm_api_key.txt (API 키), llm_config.txt (BASE_URL, MODEL_NAME)
  2) 환경변수 LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME
  3) .env 파일 (python-dotenv)

실행:
  python summarize.py                   # 최신 날짜 CSV 사용
  python summarize.py 20260326          # 특정 날짜 지정
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ─────────────────────────────────────────────
# LLM 설정 로드
# ─────────────────────────────────────────────

def _load_llm_config() -> tuple[str, str, str]:
    """(api_key, base_url, model) 반환. 미설정 시 빈 문자열."""
    # 1) dotenv 시도
    try:
        from dotenv import load_dotenv
        for _env in [BASE_DIR / ".env", BASE_DIR.parent / ".env"]:
            if _env.exists():
                load_dotenv(_env, override=False)
                break
    except ImportError:
        pass

    # 2) llm_api_key.txt / llm_config.txt 파일 우선
    key_file = BASE_DIR / "llm_api_key.txt"
    cfg_file = BASE_DIR / "llm_config.txt"

    api_key  = key_file.read_text(encoding="utf-8").strip() if key_file.exists() else ""
    base_url = ""
    model    = ""

    if cfg_file.exists():
        for line in cfg_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip().upper(), v.strip()
            if k == "LLM_BASE_URL":
                base_url = v
            elif k == "LLM_MODEL_NAME":
                model = v

    # 3) Streamlit Secrets (Streamlit Cloud 배포 환경)
    if not api_key:
        try:
            import streamlit as _st
            api_key  = api_key  or _st.secrets.get("LLM_API_KEY",   "")
            base_url = base_url or _st.secrets.get("LLM_BASE_URL",  "")
            model    = model    or _st.secrets.get("LLM_MODEL_NAME", "")
        except Exception:
            pass

    # 4) 환경변수 fallback
    api_key  = api_key  or os.environ.get("LLM_API_KEY",   "")
    base_url = base_url or os.environ.get("LLM_BASE_URL",  "https://api.openai.com/v1")
    model    = model    or os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")

    return api_key, base_url, model


# ─────────────────────────────────────────────
# CSV 로드 및 top20 추출
# ─────────────────────────────────────────────

def _get_latest_date() -> str | None:
    files = sorted(
        [f.stem for f in DATA_DIR.glob("????????.csv") if f.stem.isdigit()],
        reverse=True,
    )
    if files:
        return files[0]
    # Streamlit Cloud 등 latest.csv만 있는 환경 fallback
    if (DATA_DIR / "latest.csv").exists():
        return "latest"
    return None


def load_top20(date_str: str, tickers: list | None = None) -> pd.DataFrame:
    """해당 날짜 CSV를 읽어 합산점수 상위 20개 반환. tickers 지정 시 해당 종목만 반환."""
    csv_path = DATA_DIR / ("latest.csv" if date_str == "latest" else f"{date_str}.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 없음: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"티커": str}, encoding="utf-8-sig")

    # 특정 티커 목록이 전달된 경우 해당 종목만 순서대로 반환
    if tickers:
        df = df[df["티커"].isin(tickers)].copy()
        # 전달된 티커 순서 유지
        ticker_order = {t: i for i, t in enumerate(tickers)}
        df["_order"] = df["티커"].map(ticker_order)
        df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
        return df

    # 합산점수 기준 내림차순 정렬 (CSV에 합산점수 없으면 외인연속매수일로 대체)
    if "합산점수" in df.columns:
        df["_score"] = pd.to_numeric(df["합산점수"], errors="coerce")
    elif "외인연속매수일" in df.columns:
        df["_score"] = pd.to_numeric(df["외인연속매수일"], errors="coerce")
    else:
        df["_score"] = 0

    df = df.sort_values("_score", ascending=False, na_position="last").head(20).reset_index(drop=True)
    df = df.drop(columns=["_score"])
    return df


# ─────────────────────────────────────────────
# LLM 요약 생성
# ─────────────────────────────────────────────

PROMPT_TEMPLATE = """너는 보수적인 가치투자 애널리스트야. **{name}** (티커: {ticker})에 대해 아래 수급 데이터를 참고하여 4줄로 요약해줘.

[수급 데이터]
- 외인 연속매수: {f_days}일 / 기관 연속매수: {i_days}일
- 외인 누적순매수: {f_bil}억원 / 기관 누적순매수: {i_bil}억원
- 합산점수: {score}점 (전체 종목 백분위 기준, 최대 100점)
- 재무등급: {grade} / PBR: {pbr} / 부채비율: {debt}%

[비즈니스 모델] 이 회사가 어디서 돈을 벌고 업계 내 경쟁력이 어느 수준인지 한 줄.

[최근 모멘텀] 최근 6개월 내 주가에 영향을 준 결정적 이벤트(공시/실적/신기술) 한 줄.

[추천 이유] 위 수급 데이터를 근거로, 지금 이 종목에 주목해야 하는 구체적인 이유 한 줄.

[리스크] 이 종목에서 가장 주의해야 할 리스크 한 줄.

※ 불필요한 수식어 없이 사실 위주로, 항목당 한 줄씩 총 4줄.

반드시 다음 형식으로:
[비즈니스 모델]: ...
[최근 모멘텀]: ...
[추천 이유]: ...
[리스크]: ..."""


def summarize_stock_stream(client, model: str, name: str, ticker: str, row: dict | None = None):
    """단일 종목 LLM 스트리밍 요약. 토큰을 yield하고 마지막에 파싱된 dict를 yield."""
    row = row or {}

    def _fmt(val, default="-", fmt="{}"):
        v = pd.to_numeric(val, errors="coerce") if not isinstance(val, (int, float)) else val
        return fmt.format(v) if not pd.isna(v) else default

    prompt = PROMPT_TEMPLATE.format(
        name=name, ticker=ticker,
        f_days=_fmt(row.get("외인연속매수일"), fmt="{:.0f}"),
        i_days=_fmt(row.get("기관연속매수일"), fmt="{:.0f}"),
        f_bil=_fmt(row.get("외인매수(억)"), fmt="{:+.1f}"),
        i_bil=_fmt(row.get("기관매수(억)"), fmt="{:+.1f}"),
        score=_fmt(row.get("합산점수"), fmt="{:.1f}"),
        grade=str(row.get("재무등급") or "-"),
        pbr=_fmt(row.get("라이브PBR"), fmt="{:.2f}"),
        debt=_fmt(row.get("부채비율"), fmt="{:.0f}"),
    )
    today_str = datetime.now().strftime("%Y년 %m월 %d일")
    system_msg = (
        f"오늘 날짜는 {today_str}입니다. "
        "최신 정보 기준으로 답변하고, 2년 이상 된 정보는 언급하지 마세요. "
        "추천 근거는 반드시 제공된 수급 데이터와 연결하여 구체적으로 작성하세요."
    )

    def _make_stream(messages):
        return client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=500, temperature=0.3, stream=True,
        )

    try:
        try:
            stream = _make_stream([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ])
        except Exception as sys_err:
            if "Developer instruction is not enabled" in str(sys_err) or "system" in str(sys_err).lower():
                stream = _make_stream([
                    {"role": "user", "content": f"{system_msg}\n\n{prompt}"},
                ])
            else:
                raise

        raw = ""
        for chunk in stream:
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if token:
                raw += token
                yield token  # 토큰 스트리밍

        # 파싱 결과를 마지막에 dict로 yield
        parsed = {"raw": raw}
        for line in raw.splitlines():
            line = line.strip()
            for key in ["비즈니스 모델", "최근 모멘텀", "추천 이유", "리스크"]:
                if line.startswith(f"[{key}]:"):
                    parsed[key] = line[len(f"[{key}]:"):].strip()
                    break
                elif line.startswith(f"{key}:"):
                    parsed[key] = line[len(f"{key}:"):].strip()
                    break
        yield parsed  # 마지막에 파싱 dict

    except Exception as e:
        yield {"error": str(e), "raw": ""}



    """단일 종목 LLM 요약. 수급 데이터(row)를 프롬프트에 주입해 추천이유/리스크 설명 포함."""
    row = row or {}

    def _fmt(val, default="-", fmt="{}"):
        v = pd.to_numeric(val, errors="coerce") if not isinstance(val, (int, float)) else val
        return fmt.format(v) if not pd.isna(v) else default

    prompt = PROMPT_TEMPLATE.format(
        name=name,
        ticker=ticker,
        f_days=_fmt(row.get("외인연속매수일"), fmt="{:.0f}"),
        i_days=_fmt(row.get("기관연속매수일"), fmt="{:.0f}"),
        f_bil=_fmt(row.get("외인매수(억)"), fmt="{:+.1f}"),
        i_bil=_fmt(row.get("기관매수(억)"), fmt="{:+.1f}"),
        score=_fmt(row.get("합산점수"), fmt="{:.1f}"),
        grade=str(row.get("재무등급") or "-"),
        pbr=_fmt(row.get("라이브PBR"), fmt="{:.2f}"),
        debt=_fmt(row.get("부채비율"), fmt="{:.0f}"),
    )
    today_str = datetime.now().strftime("%Y년 %m월 %d일")
    system_msg = (
        f"오늘 날짜는 {today_str}입니다. "
        "최신 정보 기준으로 답변하고, 2년 이상 된 정보는 언급하지 마세요. "
        "추천 근거는 반드시 제공된 수급 데이터와 연결하여 구체적으로 작성하세요."
    )

    for attempt in range(3):
        try:
            # Gemma 등 system role 미지원 모델 대응: system 내용을 user 메시지 앞에 삽입
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                    temperature=0.3,
                )
            except Exception as sys_err:
                if "Developer instruction is not enabled" in str(sys_err) or "system" in str(sys_err).lower():
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": f"{system_msg}\n\n{prompt}"},
                        ],
                        max_tokens=500,
                        temperature=0.3,
                    )
                else:
                    raise
            raw = response.choices[0].message.content.strip()
            break
        except Exception as e:
            err_str = str(e)
            if "429" in err_str and attempt < 2:
                wait = 15 * (attempt + 1)
                print(f"429 한도 초과 — {wait}초 대기 후 재시도...", end="", flush=True)
                time.sleep(wait)
            else:
                return {"error": err_str, "raw": ""}
    else:
        return {"error": "최대 재시도 초과", "raw": ""}

    result = {"raw": raw}
    for line in raw.splitlines():
        line = line.strip()
        for key in ["비즈니스 모델", "최근 모멘텀", "추천 이유", "리스크"]:
            prefix1 = f"[{key}]:"
            prefix2 = f"{key}:"
            if line.startswith(prefix1):
                result[key] = line[len(prefix1):].strip()
                break
            elif line.startswith(prefix2):
                result[key] = line[len(prefix2):].strip()
                break
    return result


# ─────────────────────────────────────────────
# 스크리닝: Z-score + LLM 2단계
# ─────────────────────────────────────────────

SCREEN_PROMPT = """너는 퀀트 투자 전문가야. 아래는 오늘의 수급 상위 종목들과 각 종목의 종합점수(Z-score 기반)야.

{table}

위 종목들을 비교해서, **지금 당장 주목할 만한 종목만** 골라줘.
- 선택 기준: 수급 모멘텀이 강하고, 외인+기관이 동시에 또는 지속적으로 매집 중인 종목
- 3~7개 선택 (너무 많으면 안 됨, 진짜 강한 것만)

반드시 아래 JSON 형식으로만 답해. 다른 설명 없이 JSON만:
{{
  "recommended": ["종목명1", "종목명2", ...],
  "not_recommended": {{
    "종목명A": "비추천 이유 한 줄",
    "종목명B": "비추천 이유 한 줄"
  }}
}}"""


def _zscore(series: pd.Series) -> pd.Series:
    """NaN 안전 Z-score 표준화."""
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    std = s.std()
    if std == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.mean()) / std


def compute_z_score(df: pd.DataFrame) -> pd.Series:
    """
    가중 Z-score 종합점수 계산.
    매수일 40% (외인 20% + 기관 20%)
    매수억 40% (외인 20% + 기관 20%)
    합산점수 20%
    """
    z_f_days = _zscore(df.get("외인연속매수일", pd.Series(dtype=float))) * 0.20
    z_i_days = _zscore(df.get("기관연속매수일", pd.Series(dtype=float))) * 0.20
    z_f_bil  = _zscore(df.get("외인매수(억)",   pd.Series(dtype=float))) * 0.20
    z_i_bil  = _zscore(df.get("기관매수(억)",   pd.Series(dtype=float))) * 0.20
    z_score  = _zscore(df.get("합산점수",        pd.Series(dtype=float))) * 0.20
    return z_f_days + z_i_days + z_f_bil + z_i_bil + z_score


def screen_stocks_llm(df: pd.DataFrame, client, model: str) -> tuple:
    """
    1단계: Z-score로 종합점수 계산 → 상위 절반 후보 선정
    2단계: LLM이 후보군 비교해서 최종 추천/비추천 결정
    Returns: (rec_df, skip_df, skip_reasons)
    """
    import json as _json

    df = df.copy()
    df["_z"] = compute_z_score(df)
    df = df.sort_values("_z", ascending=False).reset_index(drop=True)

    # 상위 절반만 LLM 후보로 (최소 5개, 최대 10개)
    n_candidates = max(5, min(10, len(df) // 2))
    candidates = df.head(n_candidates).copy()
    rest = df.iloc[n_candidates:].copy()

    # 테이블 생성
    def _fmt(v, fmt="{:.1f}"):
        n = pd.to_numeric(v, errors="coerce")
        return fmt.format(n) if not pd.isna(n) else "-"

    rows_txt = []
    for i, row in candidates.iterrows():
        rows_txt.append(
            f"{row.get('종목명','')} | Z점수: {row['_z']:.2f} | "
            f"외인연속: {_fmt(row.get('외인연속매수일'))}일 | 기관연속: {_fmt(row.get('기관연속매수일'))}일 | "
            f"외인매수: {_fmt(row.get('외인매수(억)'), '{:+.0f}')}억 | 기관매수: {_fmt(row.get('기관매수(억)'), '{:+.0f}')}억 | "
            f"합산점수: {_fmt(row.get('합산점수'))}"
        )
    table_str = "\n".join(rows_txt)

    prompt = SCREEN_PROMPT.format(table=table_str)

    # LLM 호출 (system role fallback 포함)
    def _call(messages):
        return client.chat.completions.create(
            model=model, messages=messages, max_tokens=800, temperature=0.1,
        )

    try:
        try:
            resp = _call([{"role": "system", "content": "JSON만 출력하는 퀀트 분석기."}, {"role": "user", "content": prompt}])
        except Exception as e:
            if "Developer instruction is not enabled" in str(e) or "system" in str(e).lower():
                resp = _call([{"role": "user", "content": prompt}])
            else:
                raise
        raw = resp.choices[0].message.content.strip()
        # JSON 블록 추출
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        result = _json.loads(raw)
    except Exception as e:
        print(f"[WARN] LLM 스크리닝 실패, Z-score 상위만 사용: {e}")
        # fallback: Z-score 상위 절반 전체 추천
        rec_names = set(candidates["종목명"].tolist())
        skip_reasons = {str(r.get("종목명","")): ["Z-score 하위"] for _, r in rest.iterrows()}
        rec_df  = df[df["종목명"].isin(rec_names)].drop(columns=["_z"]).reset_index(drop=True)
        skip_df = df[~df["종목명"].isin(rec_names)].drop(columns=["_z"]).reset_index(drop=True)
        return rec_df, skip_df, skip_reasons

    rec_names = set(result.get("recommended", []))
    llm_not_rec = result.get("not_recommended", {})

    # Z-score 하위 종목은 LLM 판단 없이 비추천
    skip_reasons: dict[str, list[str]] = {}
    for _, row in rest.iterrows():
        skip_reasons[str(row.get("종목명",""))] = [f"Z-score 하위 (종합점수 {row['_z']:.2f})"]
    # LLM 비추천 추가
    for name, reason in llm_not_rec.items():
        skip_reasons[name] = [f"LLM 판단: {reason}"]

    all_skip = set(skip_reasons.keys())
    df_clean = df.drop(columns=["_z"])
    rec_df  = df_clean[df_clean["종목명"].isin(rec_names)].reset_index(drop=True)
    skip_df = df_clean[df_clean["종목명"].isin(all_skip)].reset_index(drop=True)
    return rec_df, skip_df, skip_reasons


def screen_stocks(df: pd.DataFrame) -> tuple:
    """Z-score만으로 상위 절반 추천 (LLM 없이 호출 시 fallback)."""
    df = df.copy()
    df["_z"] = compute_z_score(df)
    df = df.sort_values("_z", ascending=False).reset_index(drop=True)
    n = max(5, len(df) // 2)
    rec_df  = df.head(n).drop(columns=["_z"]).reset_index(drop=True)
    skip_df = df.iloc[n:].drop(columns=["_z"]).reset_index(drop=True)
    skip_reasons = {str(r.get("종목명","")): [f"Z-score 하위 ({r['_z']:.2f})"]
                    for _, r in df.iloc[n:].iterrows()}
    return rec_df, skip_df, skip_reasons


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def generate_summary(date_str: str | None = None, force: bool = False,
                     tickers: list | None = None) -> Path:
    """
    top20 AI 요약을 생성하고 data/YYYYMMDD_summary.json 에 저장.
    이미 파일이 존재하면 force=True 일 때만 재생성.
    tickers: 지정 시 해당 종목만 요약 (수급 상위 탭 종목과 일치시킬 때 사용)
    반환: 저장된 JSON 파일 경로
    """
    if date_str is None:
        date_str = _get_latest_date()
        if date_str is None:
            raise RuntimeError("data/ 디렉터리에 CSV 파일이 없습니다.")

    out_path = DATA_DIR / ("latest_summary.json" if date_str == "latest" else f"{date_str}_summary.json")
    if out_path.exists() and not force:
        print(f"[INFO] 이미 요약 파일 존재: {out_path}  (재생성하려면 --force)")
        return out_path

    api_key, base_url, model = _load_llm_config()
    if not api_key:
        raise RuntimeError(
            "LLM API 키가 설정되지 않았습니다.\n"
            "  QuietAccumulation/llm_api_key.txt 에 API 키를 저장하거나\n"
            "  LLM_API_KEY 환경변수를 설정하세요."
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai 패키지가 필요합니다: pip install openai")

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"[INFO] {date_str} top20 AI 요약 시작 (모델: {model}, URL: {base_url})")
    df = load_top20(date_str, tickers=tickers)

    # 스크리닝: Z-score + LLM 2단계
    rec_df, skip_df, skip_reasons = screen_stocks_llm(df, client, model)
    print(f"[INFO] 추천: {len(rec_df)}개 / 비추천: {len(skip_df)}개")
    for name, reasons in skip_reasons.items():
        print(f"  ✗ {name}: {', '.join(reasons)}")

    summaries: dict[str, dict] = {}
    for i, row in rec_df.iterrows():
        name   = str(row.get("종목명", ""))
        ticker = str(row.get("티커", ""))
        if not name:
            continue

        print(f"  [{i+1:02d}/{len(rec_df):02d}] {name} ({ticker}) ... ", end="", flush=True)
        summary = summarize_stock(client, model, name, ticker, row=row.to_dict())
        summaries[name] = {"ticker": ticker, **summary}

        if "error" in summary:
            print(f"ERROR: {summary['error']}")
        else:
            print("OK")

        # Rate limit 방지
        if i < len(rec_df) - 1:
            time.sleep(4)

    not_recommended = {
        str(row.get("종목명", "")): {
            "ticker": str(row.get("티커", "")),
            "reasons": skip_reasons.get(str(row.get("종목명", "")), []),
        }
        for _, row in skip_df.iterrows()
    }

    payload = {
        "date": date_str,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "summaries": summaries,
        "not_recommended": not_recommended,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 저장 완료: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("date", nargs="?", default=None, help="YYYYMMDD")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tickers", default="", help="쉼표 구분 티커 목록 (수급상위 종목 전달용)")
    _args = parser.parse_args()

    _tickers = [t.strip() for t in _args.tickers.split(",") if t.strip()] if _args.tickers else None

    try:
        generate_summary(date_str=_args.date, force=_args.force, tickers=_tickers)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
