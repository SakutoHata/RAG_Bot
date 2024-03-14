# アプリケーション層

#ライブラリのインポート
import os
import platform
import re
import pandas as pd

import openai
import chromadb
import langchain

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def update_page_content(self, new_content):
        self.page_content = new_content

class Talk_GPT:
    #knowledge chunking
    def chunking_data(self,file_path):
        """
        #URLを使う場合はこちら
        urls = ["",
                ]

        loader = UnstructuredURLLoader(urls=urls)
        """

        #PDFデータを使う場合はこちら
        loader = PyPDFLoader(file_path)


        documents = loader.load()

        # cleaned_dataの内容をDocumentオブジェクトのpage_contentに更新する
        for i, doc in enumerate(documents):
            #  まず、各Documentオブジェクトのpage_contentから不要な部分を削除します。
            cleaned_content = doc.page_content.replace("\n", "").replace("不許複製・禁無断転載", "").replace("   ", "")
            #  文頭に数字＋空白がある場合にそれを削除
            cleaned_content = re.sub(r"^\d+\s", "", cleaned_content)
            #  ここでは、cleaned_dataの内容を使用して、各Documentオブジェクトのpage_contentを更新します。
            doc.page_content = cleaned_content

        #デバッグ用
        #print(documents)

        knowledge = documents

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 200,   # チャンクの文字数
            chunk_overlap  = 100,  # チャンクオーバーラップの文字数
        )
        
        splitted_documents = text_splitter.split_documents(knowledge)
        
        #デバッグ用
        #print(f"before:{len(knowledge)}")
        #print(f"after:{len(splitted_documents)}")
        #print("chunking_data OK!")
        
        return splitted_documents

    #Creating Vector Store
    def Vector_Store(self,splitted_documents):
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(splitted_documents,embedding=embeddings, persist_directory=".")
        #print("Vector_Store OK!")
        return vectorstore

    def save_Vector_Store(self,vectorstore):
        store_data = vectorstore.get()
        # documentsのテキストデータを取得
        texts = store_data['documents']
        # metadatasからページ番号を取得
        #pages = [metadata['page'] for metadata in store_data['metadatas']]
        #  テキストデータとページ番号を使用してDataFrameを作成
        #df = pd.DataFrame({'Pages':pages, 'Text':texts})
        df = pd.DataFrame({'Text':texts})
        # CSVファイルにDataFrameを保存
        df.to_csv('RAG_Bot/program/for discord/documents_table.csv', index=False)
        return df 

    #User_question2Answer
    def User_question(self,vectorstore,llm,question):
        vectorstore.persist()
        pdf_qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
        query = question
        chat_history = []
        result = pdf_qa.invoke({"question": query, "chat_history": chat_history})

        Answer = result["answer"]
        line_feed = "\n"+"*******************************************"+"\n"
        
        #デバッグ用
        #source = result['source_documents']
        #print(result['source_documents'])
        
        # Document  オブジェクトの page_content  属性を使用して文字列に変換
        #　PDFを使う場合は以下
        #source = "\n".join(f"＜参照情報＞\n内容：{doc.page_content}\n記載箇所： {doc.metadata['source']}\n該当ページ： {doc.metadata['page']}\n" for doc in result['source_documents'])
        #　URLから取得する場合は以下
        source = "\n".join(f"＜参照情報＞\n内容：{doc.page_content}\n記載箇所： {doc.metadata['source']}\n" for doc in result['source_documents'])

        return_content = Answer + line_feed + source
        #print("User_question OK!")
        return return_content
    
    #Merge
    def merge(self,file_path,llm,question):
        chunk_data = self.chunking_data(file_path)
        vector_store = self.Vector_Store(chunk_data)
        question2answer = self.User_question(vector_store,llm,question)

        #デバッグ用
        save_data = self.save_Vector_Store(vector_store)
        print(save_data)

        return question2answer

#使用例
if __name__ == "__main__":
    Sys_RAG = Talk_GPT()

    # Openai API setting
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview") #gpt-3.5-turbo-0125") #gpt-4-0125-preview") #

    # forder setting
    folder_path = "RAG_Bot/program/for discord/PDF_files"
    file_path =   folder_path + "/" +"PDF's file name Here"

    # ユーザーからの質問をコマンドラインから入力
    question = input("質問をどうぞ：")
    print(Sys_RAG.merge(file_path,llm,question))