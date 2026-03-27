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

    # 3) 환경변수 fallback
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
    return files[0] if files else None


def load_top20(date_str: str, tickers: list | None = None) -> pd.DataFrame:
    """해당 날짜 CSV를 읽어 합산점수 상위 20개 반환. tickers 지정 시 해당 종목만 반환."""
    csv_path = DATA_DIR / f"{date_str}.csv"
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

PROMPT_TEMPLATE = """너는 글로벌 투자은행의 수석 애널리스트야. **{name}**에 대해 일반인이 이해하기 쉽게, 하지만 핵심은 놓치지 않고 딱 **3줄(문장)**로만 요약해줘.

[비즈니스 모델] 이 회사가 정확히 어디서 돈을 벌고 있으며, 업계 내 점유율이나 경쟁력이 어느 정도인지 요약할 것.

[최근 모멘텀] 최근 6개월 내 발표된 공시, 실적, 신기술 등 주가에 가장 큰 영향을 준 결정적 이벤트를 요약할 것.

[시장 컨센서스] 현재 전문가들이 보는 적정 가치 대비 저평가/고평가 여부와 향후 가장 큰 변수(리스크 포함)를 요약할 것.

※ 주의사항: 불필요한 수식어는 빼고 데이터와 사실 위주로 문장당 한 줄씩 총 3줄로 작성할 것.

반드시 다음 형식으로 답해줘 (항목 이름 포함):
[비즈니스 모델]: ...
[최근 모멘텀]: ...
[시장 컨센서스]: ..."""


def summarize_stock(client, model: str, name: str, ticker: str) -> dict:
    """단일 종목 LLM 요약. {'기업 개요': ..., '최근 동향': ..., '투자 포인트/리스크': ...}"""
    prompt = PROMPT_TEMPLATE.format(name=name)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3,
            )
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

    # 파싱: "[비즈니스 모델]: ...\n[최근 모멘텀]: ...\n[시장 컨센서스]: ..."
    result = {"raw": raw}
    for line in raw.splitlines():
        line = line.strip()
        for key in ["비즈니스 모델", "최근 모멘텀", "시장 컨센서스"]:
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
# 스크리닝: 매수 추천 / 비추천 분류
# ─────────────────────────────────────────────

def screen_stocks(df: pd.DataFrame) -> tuple:
    """
    지표 기반으로 매수 추천 / 비추천 분류.
    Returns: (rec_df, skip_df, skip_reasons: {종목명: [이유, ...]})
    """
    skip_reasons: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        name = str(row.get("종목명", ""))
        reasons = []

        # 1) 재무등급 D
        grade = str(row.get("재무등급", "") or "").strip().upper()
        if grade == "D":
            reasons.append("재무등급 D (재무위험)")

        # 2) 영업 적자
        op_margin = pd.to_numeric(row.get("영업이익률(%)"), errors="coerce")
        if not pd.isna(op_margin) and op_margin < 0:
            reasons.append(f"영업적자 ({op_margin:.1f}%)")

        # 3) 부채비율 과다
        debt = pd.to_numeric(row.get("부채비율"), errors="coerce")
        if not pd.isna(debt) and debt > 300:
            reasons.append(f"부채비율 과다 ({debt:.0f}%)")

        # 4) 수급 지속성 부족 (외인·기관 모두 2일 미만)
        f_days = pd.to_numeric(row.get("외인연속매수일", 0), errors="coerce")
        i_days = pd.to_numeric(row.get("기관연속매수일", 0), errors="coerce")
        f_ok = (not pd.isna(f_days)) and f_days >= 2
        i_ok = (not pd.isna(i_days)) and i_days >= 2
        if not f_ok and not i_ok:
            fv = f"{f_days:.0f}" if not pd.isna(f_days) else "?"
            iv = f"{i_days:.0f}" if not pd.isna(i_days) else "?"
            reasons.append(f"수급 지속성 부족 (외인 {fv}일, 기관 {iv}일)")

        if reasons:
            skip_reasons[name] = reasons

    skip_names = set(skip_reasons.keys())
    rec_df  = df[~df["종목명"].isin(skip_names)].copy().reset_index(drop=True)
    skip_df = df[df["종목명"].isin(skip_names)].copy().reset_index(drop=True)
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

    out_path = DATA_DIR / f"{date_str}_summary.json"
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

    # 스크리닝: 매수 추천 / 비추천 분류
    rec_df, skip_df, skip_reasons = screen_stocks(df)
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
        summary = summarize_stock(client, model, name, ticker)
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
