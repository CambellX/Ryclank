import os
import discord
from discord.ext import commands
from pathlib import Path
import asyncio
import re
import json
import chatbot
import tuning
import imageParsing
import conversationParsing

#Contains all the history and member details
serversDirectory = "discord_servers"
USER_ADAPTER = None

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='!', intents = intents, help_command=None)

# If the bot was being spammed, the conversation context would break. 
# Make lock for sending messages so everything happens one at a time
lock = asyncio.Lock()

# Only activates ryclank in the channels that explicitly activate it
ACTIVATED_CHANNELS= []
# Stores the temp convo of all the activated channels
ACTIVATED_CHANNEL_HISTORY = {}

def DeactivateAllModels():
    global ACTIVATED_CHANNELS, ACTIVATED_CHANNEL_HISTORY
    imageParsing.unloadImagingModel()
    conversationParsing.unloadConversationModel()
    chatbot.unloadLLM()
    ACTIVATED_CHANNELS = []
    ACTIVATED_CHANNEL_HISTORY = {}

# ended up tossing everything into different directories for oraganization
# so this function keeps track of all of that. Easier to change in the future too.
def getServerPaths(guild_id):
    guildDir = f"{serversDirectory}/{guild_id}"
    return {
        "guild": guildDir,
        "history": f"{guildDir}/history.json",
        "users": f"{guildDir}/users.json",
        "tuning": f"{guildDir}/training_data",
        "adapters": f"{guildDir}/adapters"
    }

# Just checks of all the directories actually exist
def checkDirectories(ctx):
    try:
        os.makedirs(serversDirectory, exist_ok=True)
        os.makedirs(getServerPaths(ctx.guild.id)['guild'], exist_ok=True)
        os.makedirs(getServerPaths(ctx.guild.id)['tuning'], exist_ok=True)
        os.makedirs(getServerPaths(ctx.guild.id)['adapters'], exist_ok=True)
        return True
    except:
        return False

# Checks if the message has a reply. Useful when building conversation chains
def getMessageReply(message):
    if not message.reference:
        return None
    
    if message.reference.message_id:
        return message.reference.message_id
    
    return None

# Skips bad data and prevents it from building data on itself. 
# Also sorts out all the pings after the grand tragedy The Great Pinging of `26
pingsRegex = r"^<@\d+>$"
def cleanMessages(message):
    content = message.content.strip()
    
    if (content.startswith("!") or message.author.name == "RyClank" or re.match(pingsRegex, content) or len(content) < 5):
        return False
    return True

# Checks if the user exists in the discord server
def isValidUser(guild_id, username):
    usersPath = getServerPaths(guild_id)['users']
    if not Path(usersPath).is_file():
        return False
    with open (usersPath, "r") as f:
        users = json.load(f)
    return username in users.values()

# Gets all the users in a discord server
async def getAllUsers(ctx):
    usersLocation = getServerPaths(ctx.guild.id)['users']
    await ctx.send("Getting list of users")
    usersToID = {}
    for member in ctx.guild.members:
        usersToID.update(
            {str(member.id): str(member.name)}
            )
    with open (usersLocation, "w") as f:
        json.dump(usersToID, f)
    return usersToID

# Converts a user id to a string. Used so the bot can map an ID to a username
# Raw discord id looks like <@123812312312312>
IDRegex = r"<@(\d+)>"
def IDtoUsername(text, IDtoUsername):
    def replace(match):
        # match just the pinging part
        user_id = match.group(1)
        # Replace it with a username
        return f"@{IDtoUsername.get(user_id, 'UNKNOWN')}"
    
    # Replaces the discord ID with @username
    return re.sub(IDRegex, replace, text)

# treat replies as a tree for future parsing in convos. Decided to do this as a first pass.
# Just uses discord's reply feature to make the convo paths. Its the easiest and people 
# always refer back on previous conversations.
def getAllReplies(messages):
    #hashmap store all child : parents
    allRepliedMessages = {}
    for message in messages:
        # IF the message replies to another message, set the message's parent to the replied to message.
        parentID = message.get("replyTo")
        if parentID:
            allRepliedMessages[message["messageId"]] = parentID
    return allRepliedMessages

# Given a message, check if it exists in the replies tree.
# If so, add it to the corresponding conversation it replies to.
def addRepliesToConversation(idToConversation, currentMessage, replies):
    parentID = replies.get(currentMessage["messageId"])

    if parentID not in idToConversation:
        return False

    #go through and check if the new message replies to any of the conversation's messages
    conversation = idToConversation[parentID]
    conversation.append(currentMessage)
    idToConversation[currentMessage["messageId"]] = conversation
    
    return True

#Tested with this:
# 匚尺丹乙ㄚ [Ado],  — 4:51 PM
# yeah
# i assume you have TWO cables with your psu
# for the gpu
# both of which are y split
# so you have 4 total 8pins for the gpu
# unammed123 — 4:52 PM
# saw a youtube short and they were playing the stupid rolly polly music from newport
# Яisky — 4:52 PM
# Only 1 going into the gpu but that turns into 2
#
# When i spoke, i made a new conversation topic. The bot should successfully be able to separate
# The conversation between the other two whilst separating me talking to myself as a new convo.

# After testing, this splits into 3 convos, but I dont thinkit will affect the eventual tuning data.

def separateConversations(messages):
    if not messages:
        print("ERRROR: Cant find messages to separate")
        return []
    
    #message to parent hashmap
    replies = getAllReplies(messages)

    # stores all the messages split up into convos
    conversations = []
    currentConversation = [messages[0]]

    # Maps every message ID to a conversation
    idToConversation = {}
    idToConversation[messages[0]["messageId"]] = currentConversation

    # Iterate through all messages in the discord server and separate them into conversations
    for i in range(1, len(messages)):
        nextMessage = messages[i]

        # get the latest conversation which is the most likely conversation the next message will reference
        latestContext = currentConversation[-1]
        
        # If it replies to a previous message thne directly add it to the corresponding convo
        if addRepliesToConversation(idToConversation, nextMessage, replies):
            continue

        # When we run out of messages in a channel, just end the conversation, rinse, and repeat.
        if nextMessage["channel"] != latestContext["channel"]:
            conversations.append(currentConversation)
            currentConversation = [nextMessage]
            idToConversation[nextMessage["messageId"]] = currentConversation
            continue   
        
        #otherwise use the conversationParsing llm to determine if convo continues or not
        result = conversationParsing.separateConversations(currentConversation, nextMessage)

        # LLM returns CONTINUE, UNCERTAIN, or NEW. More documentation on that file.
        if result == "CONTINUE":
            # if the new messsage continues the convo, append it to the current conversation.
            currentConversation.append(nextMessage)
            idToConversation[nextMessage["messageId"]] = currentConversation
        # ill add more functionality for uncertain later
        else:
            # otherwise add the conversation to the map and start a new one.
            conversations.append(currentConversation)
            currentConversation = [nextMessage]
            idToConversation[nextMessage["messageId"]] = currentConversation
    # append the last conversation
    conversations.append(currentConversation)
    return conversations

# Uses an image reading llm to describe the image so Ryclank can view images as well
async def parseAllAttachments(attachments, message):
    for attachment in message.attachments:
        contentType = attachment.content_type or ""
        if contentType.startswith("image/"):
            try:
                imageDescription = await asyncio.to_thread(imageParsing.processImage, attachment.url)
            except Exception as error:
                print(f"ERROR: {attachment.filename}: {error}")
                imageDescription = None

            attachments.append({
                "url": attachment.url,
                "description": imageDescription
            })
        else:
            print(f"Skipped {contentType}")

# Turns conversations into a more standardized and readable format
# A helper to setup where the entire chat history is saved into a file
async def buildMessageData(rawMessages, usersToID, message, channel):
    #get any message replies if there is any
    if not message.attachments and not cleanMessages(message):
        return
    
    message_id = getMessageReply(message)

    rawMessage = message.content.strip()
    rawMessage = IDtoUsername(rawMessage, usersToID)


    attachments = []
    await parseAllAttachments(attachments, message)
    # Format the data
    rawMessages.append({
        "messageId": message.id,
        "channel": channel.name,
        "author": message.author.name,
        "timestamp": message.created_at.isoformat(),
        "content": rawMessage,
        "replyTo": message_id,
        "attachments": attachments,
    })

# Writes the pulled data into the corresponding directory
def writeConvoToFile(conversations, historyLocation):
    with open (historyLocation, "w") as f:
        for i, conversation in enumerate(conversations):
            f.write(json.dumps({
                "conversationNum": i + 1,
                "messages": conversation
                }) + "\n")

# Sets up ryclank in the current server. Pulls all the history, attachments, and 
# builds the environment to tune on people.
@bot.command()
async def setup(ctx):
    # Make sure all the required directories are there
    checkDirectories(ctx)

    # Obtain all the users in the server
    usersToID = await getAllUsers(ctx)

    await asyncio.to_thread(imageParsing.loadImagingModel)
    # save different histories into different files as the id of each server
    await ctx.send(f"doing setup")
    rawMessages = []
    for channel in ctx.guild.text_channels:
        try:
            # If the channel is the testing channel, ignore it because it has a ton of bad data.
            if channel.name == "ryclank-testing-grounds":
                continue
            async for message in channel.history(limit=None, oldest_first=True):
                # Build the rawMessages array which contains all of a server's messages
                await buildMessageData(rawMessages, usersToID, message, channel)

        except Exception as e:
            print(f"Error: {e}")

    await asyncio.to_thread(imageParsing.unloadImagingModel)
    await ctx.send("done with collection")

    await asyncio.to_thread(conversationParsing.loadConversationModel)

    # Separate all the convos and then finally write to file
    conversations = await asyncio.to_thread(separateConversations, rawMessages)
    await asyncio.to_thread(conversationParsing.unloadConversationModel)
    await ctx.send("Done with separation")
    historyLocation = getServerPaths(ctx.guild.id)['history']
    writeConvoToFile(conversations, historyLocation)
    await ctx.send("Finished setup")

# # # # # # # # # # # # # # # # # # # # # # # # #
# Clauded this. Couldnt get the format right after billions of years.

LINK_REGEX = re.compile(r"https?://\S+")
MENTION_SIGIL_REGEX = re.compile(r"@([A-Za-z0-9_.-]+)")

def clean(text):
    text = LINK_REGEX.sub(" ", text)
    text = MENTION_SIGIL_REGEX.sub(r"\1", text)
    return " ".join(text.split())

def formatToShareGPT(conversations, target_username, context_window=20):
    dataset = []
    for convo in conversations:
        lines = []
        for message in convo['messages']:
            content = clean(m['content'])

            for attachment in message.get('attachments', []):
                if attachment.get('description'):
                    content = f"{content} [image: {clean(attachment['description'])}]".strip()

            # Messages that were nothing but a link are stripped
            if not content:
                continue

            lines.append({"author": message['author'], "content": content})

        # if a convo doesnt have the targetted username talking in it, skip it because its not 
        # important to train on
        for i in range(1, len(lines)):
            if lines[i]['author'] != target_username:
                continue

            target_content = lines[i]['content']
            if len(target_content) < 4:
                continue

            start = max(0, i - context_window)
            context = "\n".join(f"{m['author']}: {m['content']}" for m in lines[start:i])
            if not context:
                continue

            dataset.append({
                "conversations": [
                    {"from": "system", "value": chatbot.makeSystemPrompt(target_username)},
                    {"from": "human", "value": chatbot.makeUserPrompt(context, target_username)},
                    {"from": "gpt", "value": f"{target_username}: {target_content}"}
                ]
            })
    return dataset

# # # # # # # # # # # # # # # # # # # # # # # # #

# Given a history JSON file, rebuild conversations from what was saved into a conversations string
def parseConversationFile(historyLocation, conversations):
    with open(historyLocation, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conversations.append(json.loads(line))

# Just save a bunch of datasets into a tuning location. Saving it is better
# because i can debug the training data later.
def saveToTuningLocation(tuningLocation, dataset):
    with open(tuningLocation, "w") as f:
            json.dump(dataset, f, indent=2)

# Converts the discord history into training data specifically for one username
async def generateTrainingData(ctx, targetUsername):
    historyLocation = getServerPaths(ctx.guild.id)['history']
    if not Path(historyLocation).is_file():
        await ctx.send("Need to first run setup")
        return False
    
    if not isValidUser(ctx.guild.id, targetUsername):
        await ctx.send("Can't find user. Run setup again or use someone else")
        return False

    # Get all of the conversations
    conversations = []

    # Convert the JSON file of conversations back into a string
    parseConversationFile(historyLocation, conversations)

    # format the data into the ShareGPT format for tuning later
    dataset = await asyncio.to_thread(formatToShareGPT, conversations, targetUsername)

    # save that data locally for debugging and in case tuning fails somehow
    tuningLocation = f"{getServerPaths(ctx.guild.id)['tuning']}/{targetUsername}_training_data.json"
    saveToTuningLocation(tuningLocation, dataset)

    await ctx.send(f"{targetUsername}: {len(dataset)} pairs")
    return True


# tune LoRA adapters based off the generated training data.
@bot.command()
async def tune(ctx, username):
    result = await generateTrainingData(ctx, username)
    if not result:
        await ctx.send("Tuning failed")
        return
    # stop all generation for more GPU resources
    DeactivateAllModels()

    try:
        # Tune adapters
        await asyncio.to_thread(tuning.tuneModel, ctx.guild.id, username)
        await ctx.send(f"Finished tuning on {username}")
    except Exception as e:
        await ctx.send("Something failed")
        print(e)

# Lists all of the adapters available for a specific discord server
@bot.command()
async def listAdapters(ctx):
    adaptersLocation = getServerPaths(ctx.guild.id)['adapters']
    if not Path(adaptersLocation).is_dir():
        await ctx.send("No adapters found")
        return
    
    adapters = []
    # Look through the adapters directory, then check which adapters were made.
    for p in Path(adaptersLocation).iterdir():
        if p.is_dir():
            adapters.append(p.name)

    if not adapters:
        await ctx.send("No adapters generated yet. Use !tune")
        return

    await ctx.send("Trained adapters:" + "\n".join(f"- {a}" for a in adapters))

# Makes the LLM use a pretrained LoRA adapter.
@bot.command()
async def equipAdapter(ctx, username):
    global USER_ADAPTER

    if not checkDirectories(ctx):
        await ctx.send("Can't find directories. Run setup first")
        return
    # Remove the previous adapter if there is one
    chatbot.unloadLLM()

    # Equip the adapter
    await ctx.send(f"Equipping {username}")
    result = await asyncio.to_thread(chatbot.equipAdapter, ctx.guild.id, username)
    if not result:
        await ctx.send(f"No adapter found for {username}")
        return
    
    USER_ADAPTER = username

    # Get the user's information.
    member = discord.utils.find(
        lambda m: m.name == username or m.display_name == username or str(m) == username,
        ctx.guild.members
    )
    
    if not member:
        await ctx.send("Yea ur stupid")
        return

    #copy their avatar
    copiedAvatar = await member.display_avatar.read()
    await bot.user.edit(avatar=copiedAvatar)

    #copy their name
    await ctx.guild.me.edit(nick=member.display_name)
    await ctx.send(f"Equipped adapter for {username}")

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# Allows a bot to speak in the current channel.
@bot.command()
async def activateChannel(ctx):
    if ctx.channel.id not in ACTIVATED_CHANNELS:
        ACTIVATED_CHANNELS.append(ctx.channel.id)
        ACTIVATED_CHANNEL_HISTORY[ctx.channel.id] = []
        # async for msg in ctx.channel.history(limit=10, oldest_first=True):
        #         if not msg.author.bot and isValidMessage(msg):
        #             ACTIVATED_CHANNEL_HISTORY[ctx.channel.name].append(f"{msg.author.name}: {msg.content.strip()}")
        await asyncio.to_thread(conversationParsing.loadConversationModel)
        await ctx.send("Activated Channel")
        return
    await ctx.send("Already activated channel")

# Stops a bot from speaking in the current channel
@bot.command()
async def deactivateChannel(ctx):
    if ctx.channel.id in ACTIVATED_CHANNELS:
        ACTIVATED_CHANNELS.remove(ctx.channel.id)
        ACTIVATED_CHANNEL_HISTORY.pop(ctx.channel.id)
        await ctx.send("Deactivated Channel")
        return
    await ctx.send("Channel not activated")

# for debug
@bot.command()
async def ping(ctx):
    print (ctx.author)
    await ctx.send(f"Current Adapter: {USER_ADAPTER}")
    if ctx.channel.id in ACTIVATED_CHANNELS:
        await ctx.send("Channel Activated")
    else:
        await ctx.send("Channel Deactivated")

# Determine whether Ryclank should respond a message.
def isValidMessage(message):

    # Ensure bot doesnt respond to itself
    if message.author == bot.user or message.author.bot: return False

    # return if no adapter is equipped
    if not USER_ADAPTER: return False

    # if the bot is pinged but not in an active channel, then return true.
    if bot.user in message.mentions: return True

    if message.channel.id not in ACTIVATED_CHANNELS: return False

    if not cleanMessages(message): return False

    return True

async def sendMessage(message, history):
    async with lock:
        conversation = "\n".join(history)
        try:
            async with message.channel.typing():
                response = await asyncio.wait_for(
                    asyncio.to_thread(chatbot.sendMessage, conversation, message.author.name, message.content.strip(),
                    ),
                    timeout=45,
                )
        except asyncio.TimeoutError:
            print("TIMEDOUT")
            response = "error"
        except Exception as e:
            print(f"Random Exception Happened: {e}")
            response = "error"
        await message.channel.send(response)

        for line in response.splitlines():
            if line.strip():
                history.append(f"{USER_ADAPTER}: {line.strip()}")

async def pingBehavior(message):
    recent = []
    async for msg in message.channel.history(limit=10):
        recent.append(msg)
    recent.reverse()

    history = []
    for msg in recent:
        if msg.content.startswith("!"): continue

        # If the bot was the one that spoke, change it to the user adapter.
        name = None
        if msg.author == bot.user and USER_ADAPTER:
            name = USER_ADAPTER
        else:
            name = msg.author.name

        # Send the message
        content = msg.content.strip()
        if content:
            history.append(f"{name}: {content}")

    await sendMessage(message, history)

# Runs whenever someone sends a message.
@bot.event
async def on_message(message):
    # Check if the message was a command or not valid
    await bot.process_commands(message)

    if not isValidMessage(message): return

    # If the bot was pinged in another channel thats not activated, then let the bot run using recent history
    if bot.user in message.mentions and message.channel.id not in ACTIVATED_CHANNELS:
        await pingBehavior(message)
        return

    # Use the most recent history to determine the bot's next response.
    nextMessage = f"{message.author.name}: {message.content.strip()}"
    history = ACTIVATED_CHANNEL_HISTORY[message.channel.id]

    # Get the conversation bot to decide whether the next message continues the conversation or not
    try:
        await asyncio.to_thread(conversationParsing.loadConversationModel)
        result = await asyncio.to_thread(
            conversationParsing.separateConversations,
            history,
            nextMessage,
        )
    except Exception as e:
        result = "NEW"
        print(e)
    # If the newest message creates a new conversation, then clear the recent history and move on.
    if result == "CONTINUE":
        history.append(nextMessage)
    # Otherwise create a new conversation
    else:
        ACTIVATED_CHANNEL_HISTORY[message.channel.id] = [nextMessage]
        history = ACTIVATED_CHANNEL_HISTORY[message.channel.id]
    # Send the message using the most recent message and the conversation history.
    await sendMessage(message, history)

# Unloads adapters and stops the bot.
@bot.command()
async def stop(ctx):
    global USER_ADAPTER
    USER_ADAPTER = None

    chatbot.unloadLLM()
    
    await ctx.send("De-equipped adapter")

# # # # # # # # # # # # # # # # # # # # # # # # #
@bot.command()
async def help(ctx):
    await ctx.send(
    """**RyClank Bot Commands:**
    `!(activate/deactivate)Channel` - talk/mute in current channel. Channels deactivated by default.
    `!setup` - Fetch all users and chat history from the server, parse images & separate convos. Takes like 5 hours for 30k messages
    `!tune <username>` - Tunes ryclank on someone
    `!equipAdapter <username>` - Equips the adapter of the username
    `!listAdapters` - lists all the available ppl to equip adapters on
    `!ping` - shows equipped adapter and activated channels
    `!stop` - :ryslow:
    """
    )