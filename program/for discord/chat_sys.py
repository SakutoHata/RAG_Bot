# チャットボット層

#ライブラリのインポート
import os
import platform
import openai
import chromadb
import langchain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import app as gpt_sys
## 以下、discrd用
from discord.ext import commands
import discord


# Botのトークンを設定します。
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message):
        content = message.content
        # Openai API setting
        openai.api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview") #gpt-3.5-turbo-0125") #gpt-4-0125-preview") #

        # forder setting
        folder_path = "RAG_Bot/program/for discord/PDF_files"
        file_path =   folder_path + "/" +"PDF's file name Here"


        ## Talk_GPTを実行後、返答
        Sys_RAG = gpt_sys.Talk_GPT()
        response = f"{message.author.mention}\n"+Sys_RAG.merge(file_path,llm,content)
        await message.channel.send(response)

# 実行
bot.run(TOKEN)
