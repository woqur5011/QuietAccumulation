"""
스케줄러 - collector.py 를 주기적으로 실행
기본 주기: 1시간 (장 시간 중), 6시간, 1일
APScheduler 사용
"""

import logging
import subprocess
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from collector import collect_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

scheduler = BlockingScheduler(timezone="Asia/Seoul")


# ─────────────────────────────────────────────
# 잡 정의
# ─────────────────────────────────────────────

def job_collect():
    try:
        collect_snapshot(mode="watchlist")
    except Exception as e:
        log.error(f"수집 중 오류: {e}")


def job_collect_full():
    """코스피+코스닥 전 종목 수집 (오래 걸림)"""
    try:
        collect_snapshot(mode="full")
    except Exception as e:
        log.error(f"전체 수집 중 오류: {e}")


def job_ai_summary():
    """전종목 수집 완료 후 top20 AI 요약 생성 (17:30)"""
    summarize_script = Path(__file__).parent / "summarize.py"
    try:
        result = subprocess.run(
            [sys.executable, str(summarize_script)],
            capture_output=True, text=True, encoding="utf-8",
        )
        if result.returncode == 0:
            log.info("AI 요약 생성 완료")
        else:
            log.error(f"AI 요약 실패:\n{result.stderr or result.stdout}")
    except Exception as e:
        log.error(f"AI 요약 실행 오류: {e}")


# 1) 장 시간 중 1시간 간격  (평일 09:00 ~ 15:30)
scheduler.add_job(
    job_collect,
    trigger=CronTrigger(
        day_of_week="mon-fri",
        hour="9-15",
        minute=0,
        timezone="Asia/Seoul"
    ),
    id="market_hours_1h",
    name="장중 1시간 수집",
    replace_existing=True,
    misfire_grace_time=300,
)

# 2) 장 마감 후 전종목 수집 (평일 16:30 — 당일 최종 데이터 기준)
#    약 30~40분 소요. watchlist 장중 수집과 겹치지 않도록 16:30 설정
scheduler.add_job(
    job_collect_full,
    trigger=CronTrigger(
        day_of_week="mon-fri",
        hour=16,
        minute=30,
        timezone="Asia/Seoul"
    ),
    id="daily_full_after_market",
    name="장마감 후 전종목 수집 (KOSPI+KOSDAQ)",
    replace_existing=True,
    misfire_grace_time=1800,
)

# 2-2) 점심 전종목 수집 (평일 12:00)
scheduler.add_job(
    job_collect_full,
    trigger=CronTrigger(
        day_of_week="mon-fri",
        hour=12,
        minute=0,
        timezone="Asia/Seoul"
    ),
    id="midday_full",
    name="점심 전종목 수집 (KOSPI+KOSDAQ)",
    replace_existing=True,
    misfire_grace_time=1800,
)

# 3) 매일 오전 08:50 (장 시작 전 — watchlist 빠른 스냅샷)
scheduler.add_job(
    job_collect,
    trigger=CronTrigger(
        day_of_week="mon-fri",
        hour=8,
        minute=50,
        timezone="Asia/Seoul"
    ),
    id="daily_pre_market",
    name="장전 watchlist 수집",
    replace_existing=True,
    misfire_grace_time=300,
)

# 4) 장마감 후 AI 요약 (평일 17:30 — 전종목 수집 완료 후)
scheduler.add_job(
    job_ai_summary,
    trigger=CronTrigger(
        day_of_week="mon-fri",
        hour=17,
        minute=30,
        timezone="Asia/Seoul"
    ),
    id="daily_ai_summary",
    name="top20 AI 요약 생성",
    replace_existing=True,
    misfire_grace_time=1800,
)

# ─────────────────────────────────────────────
# 시작 시 즉시 1회 실행
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log.info("스케줄러 시작 (Ctrl+C 로 중지)")
    log.info("  · 장중  1시간 간격: 평일 09:00~15:00  (watchlist, 빠름)")
    log.info("  · 점심  전종목:    평일 12:00         (full KOSPI+KOSDAQ, ~30분)")
    log.info("  · 장마감 전종목:    평일 16:30         (full KOSPI+KOSDAQ, ~30분)")
    log.info("  · 장전  watchlist: 평일 08:50         (watchlist, 빠름)")
    log.info("  · AI 요약:         평일 17:30         (top20 LLM 요약)")
    log.info("  → 최초 1회 즉시 실행 (watchlist)...")
    job_collect()
    scheduler.start()
