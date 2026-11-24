from datetime import datetime
import math
import requests
import json
from zoneinfo import ZoneInfo

YEAR_SECONDS = 31536000
CTFTIME_URL = "https://ctftime.org"
# tags that idc about
IGNORED_TAGS = ["organizers", "logo", "ctf_id", "participants", "is_votable_now", "public_votable", "live_feed", "id", "duration", "location", "format_id", "title", "onsite", "description", "weight"]
# tags that need to be converted to PST
DATE_TAGS = ["start", "finish"]
LOCAL_TIMEZONE = ZoneInfo("America/Los_Angeles")

def fetchAPI(limit):
    # Literally get all the ctfs a year in advance, just limit by the # requested
    start = math.floor((datetime.utcnow() - datetime(1970, 1, 1)).total_seconds())
    finish = start + YEAR_SECONDS

    # ctftime auto blocks no user agent
    headers = {
        "User-Agent": "CTFEventBot/99999.0"
    }

    #betch
    try:
        response = requests.get(f"{CTFTIME_URL}/api/v1/events/?limit={limit}&start={start}&finish={finish}", headers=headers)
    except Exception as e:
        return {}
        
    return response.json()

# parse the json into individual event info
def parseInfo(ctftime_response):    
    arr = []
    for event in ctftime_response:
        current_event_info = {"title": event["title"]}
        for tag in event:
            if tag in DATE_TAGS:
                current_event_info.update({tag: toTimezone(event[tag], LOCAL_TIMEZONE)})
            # elif not tag in IGNORED_TAGS:
            #     current_event_info.update({tag: str(event[tag])[:256]})
            current_event_info.update({tag: str(event[tag])[:256]})
        # current_event_info.update({"description": str(event['description'])[:250]})
        arr.append(current_event_info)
    return arr

# convert from utc to local timezone (for me pacific)
def toTimezone(timeUTC, to_timezone):
    dt = datetime.fromisoformat(timeUTC)
    dt = dt.astimezone(to_timezone)
    return dt.strftime("%Y-%m-%d %I:%M %p %Z")

# convert the json to text
def toText(eventsArr):
    ret = ""
    for event in eventsArr:
        for key in event:
            if not key in IGNORED_TAGS:
            # ret = f"{ret}\n{key}:{event[key]}"
                ret = f"{ret}\n{event[key]}"
        ret = f"{ret}\n"
    return ret

# do everything
def fetchUpcomingString(limit):
    ctftime_response_json = fetchAPI(limit)
    raw_ctfInfo = parseInfo(ctftime_response_json)
    text = toText(raw_ctfInfo)
    return text

def fetchCTF(name):
    ctftime_response_json = fetchAPI(100)
    raw_ctfs_info = parseInfo(ctftime_response_json)
    for ctf in raw_ctfs_info:
        if (ctf["title"] or "").lower() == name.lower(): return ctf
    return ""