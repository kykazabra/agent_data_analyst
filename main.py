from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, CombinedMemory
from langchain_openai import ChatOpenAI
from langchain.callbacks import FileCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_experimental.agents import create_pandas_dataframe_agent
from typing import Dict, Any
import nbformat as nbf
import json
import pandas as pd
import os


class NotebookOperator:
    def __init__(self, notebook_path: str) -> None:
        if os.path.isfile(notebook_path):
            os.remove(notebook_path)
        self.notebook_path = notebook_path
        self.nb = nbf.v4.new_notebook()

    def refresh_notebook(self) -> None:
        with open(self.notebook_path, 'w') as f:
            nbf.write(self.nb, f)

    def add_code_cell(self, code: str) -> None:
        self.nb['cells'].append(nbf.v4.new_code_cell(code))
        self.refresh_notebook()

    def add_markdown_cell(self, text: str) -> None:
        self.nb['cells'].append(nbf.v4.new_markdown_cell(text))
        self.refresh_notebook()


class CodeCallbackHandler(BaseCallbackHandler):
    def __init__(self, notebook: NotebookOperator) -> None:
        self.notebook = notebook

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        if serialized['name'] == 'python_repl_ast':
            self.notebook.add_code_cell(input_str)
            print('1')


class DataAnalyst:
    def __init__(self, df_path: str, model: str = 'gpt-3.5-turbo', cred_path: str = 'credentials.json',
                 log_path: str = 'agent_logs.log', notebook_path: str = 'agent_results.ipynb', logging: bool = True,
                 notebook: bool = True) -> None:
        self.df = self.read_data(df_path)

        with open(cred_path, 'r') as f:
            data = json.load(f)
            openai_api_key = data['openai_api_key']
            base_url = data['base_url']

        self.agent_llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model,
            base_url=base_url,
            temperature=0
        )

        self.memory = self.prepare_memory(
            llm=ChatOpenAI(
                openai_api_key=openai_api_key,
                model_name=model,
                base_url=base_url)
        )

        self.callbacks = list()

        if logging:
            self.callbacks.append(FileCallbackHandler(log_path))

        if notebook:
            self.notebook = NotebookOperator(notebook_path)
            self.callbacks.append(CodeCallbackHandler(self.notebook))

        prefix = """
        You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
        Answer only in Russian language.
        You should use the tools below to answer the question posed of you:

        Summary of the whole conversation:
        {chat_history_summary}

        Last few messages between you and user:
        {chat_history_buffer}
        """

        self.agent = create_pandas_dataframe_agent(
            llm=self.agent_llm,
            df=self.df,
            prefix=prefix,
            agent_executor_kwargs={
                "memory": self.memory,
                "callbacks": self.callbacks}
        )

    @staticmethod
    def prepare_memory(llm: ChatOpenAI, window_size: int = 5) -> CombinedMemory:
        chat_history_buffer = ConversationBufferWindowMemory(
            k=window_size,
            memory_key="chat_history_buffer",
            input_key="input"
        )

        chat_history_summary = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history_summary",
            input_key="input"
        )

        return CombinedMemory(memories=[chat_history_buffer, chat_history_summary])

    @staticmethod
    def read_data(df_path: str) -> pd.DataFrame:
        try:
            if df_path[-3:] == 'csv':
                df = pd.read_csv(df_path, delimiter=';')

            elif df_path[-4:] == 'xlsx':
                df = pd.read_csv(df_path)

            else:
                raise TypeError('К сожалению, такой тип входных данных не поддерживается.')

        except pd.errors.ParserError:
            print('Ошибка в чтении данных. Возможно, формат вашего файла не соответствует ожидаемому.')
            raise

        return df

    def talk(self, user_input: str) -> str:
        if 'notebook' in self.__dict__.keys():
            self.notebook.add_markdown_cell(user_input)

        reply = self.agent.invoke(user_input)['output']

        if 'notebook' in self.__dict__.keys():
            self.notebook.add_markdown_cell(reply)

        return reply

