import unsloth
import bot
import os
from dotenv import load_dotenv


if __name__ == "__main__":
    # Get the bot token
    load_dotenv()
    
    TOKEN = os.environ.get("TOKEN")
    if not TOKEN:
        print("no discord token in .env")
        
    bot.bot.run(TOKEN)