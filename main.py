from modules.scenario import main_bot

from datetime import datetime
import argparse

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cred_path", type=str, default='./credential.yaml')
    parser.add_argument('-l','--load', action='store_true')
    parser.add_argument('-d','--date', type=str, default='')
    parser.add_argument('--custom_url', action='store_true')


    args = parser.parse_args()
    cred_path = args.cred_path
    load_option = args.load
    date = args.date
    custom_url = args.custom_url
    bot = main_bot(cred_path=cred_path, load_option=load_option, date=date, custom_url=custom_url)
    bot.run()

