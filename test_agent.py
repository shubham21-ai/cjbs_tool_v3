import os
from dotenv import load_dotenv
load_dotenv()
from basic import BasicInfoBot
import sys
bot = BasicInfoBot()
try:
    print(bot.process_satellite("Hubble"))
except Exception as e:
    import traceback
    traceback.print_exc()
