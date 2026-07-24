import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.scenario import main_bot
from modules.utils import post_to_notion, get_notion_top_insert_anchor


def main():
    kst = timezone(timedelta(hours=9))
    date_str = (datetime.now(kst) + timedelta(days=1)).strftime('%Y%m%d')

    bot = main_bot(False, date=date_str)
    bot.run()

    # Insert newest content right at the top (below any fixed header), pushing
    # previous days' posts down instead of appending to the bottom of the page.
    anchor = get_notion_top_insert_anchor(bot.credential)
    result = post_to_notion(bot._response, bot.credential, heading="📨 공유용 뉴스레터", after=anchor)
    last_block_id = result["results"][-1]["id"]
    post_to_notion(bot._response_all, bot.credential, heading="📄 전체 수집 기사 요약", after=last_block_id)


if __name__ == "__main__":
    main()
