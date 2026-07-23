import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.scenario import main_bot
from modules.utils import post_to_notion


def main():
    kst = timezone(timedelta(hours=9))
    date_str = (datetime.now(kst) + timedelta(days=1)).strftime('%Y%m%d')

    bot = main_bot(False, date=date_str)
    bot.run()
    post_to_notion(bot._response, bot.credential, heading="📨 공유용 뉴스레터")
    post_to_notion(bot._response_all, bot.credential, heading="📄 전체 수집 기사 요약")


if __name__ == "__main__":
    main()
