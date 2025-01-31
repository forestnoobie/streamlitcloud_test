from modules.scenario import parse_bot
from datetime import datetime
import argparse

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cred_path", type=str, default='./credential.yaml')
    parser.add_argument("-s", "--save_path", type=str, default='./data/news_df_')
    parser.add_argument('-d','--date', type=str, default='')


    args = parser.parse_args()
    cred_path = args.cred_path
    save_path = args.save_path
    save_date = args.date

    bot = parse_bot(cred_path=cred_path, save_date=save_date, save_path=save_path)
    bot.run()

