# 导入必要的库
import os
import sys
from LLM import InternLM_LLM
import requests
import gradio as gr
from BCEmbedding.tools.langchain import BCERerank
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever
from openxlab.model import download

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
os.makedirs('model', exist_ok=True)
os.makedirs('internlm2-chat-7b', exist_ok=True)
#download(model_repo='OpenLMLab/internlm2-chat-7b', output='internlm2-chat-7b')

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

hf_token = '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'

os.system(f'huggingface-cli download --force-download internlm/internlm2-chat-7b --local-dir internlm2-chat-7b --token {hf_token}')
os.system(f'huggingface-cli download --force-download maidalun1020/bce-embedding-base_v1 --local-dir model/bce-embedding-base_v1 --token {hf_token}')
os.system(f'huggingface-cli download --force-download maidalun1020/bce-reranker-base_v1 --local-dir model/bce-reranker-base_v1 --token {hf_token}')


def load_chain():
    # 加载问答链
    # 加载本地索引
    embedding_model_name =  './model/bce-embedding-base_v1'  # 'maidalun1020/bce-embedding-base_v1'
    embedding_model_kwargs = {'device': 'cuda:0'}
    embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}
    embeddings = HuggingFaceEmbeddings(
                      model_name=embedding_model_name,
                      model_kwargs=embedding_model_kwargs,
                      encode_kwargs=embedding_encode_kwargs
                )
    loaded_index = FAISS.load_local('./faiss_index', embeddings)
    # 构建检索器
    reranker_args = {'model': './model/bce-reranker-base_v1', 'top_n': 50, 'device': 'cuda:0'}
    reranker = BCERerank(**reranker_args)
    retriever = loaded_index.as_retriever(search_type="similarity", search_kwargs={"score_threshold": 0.3, "k": 50})
    compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
    llm = InternLM_LLM(model_path="internlm2-chat-7b")

    template = """使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
    问题: {question}
    可参考的上下文：
    ···
    {context}
    ···
    如果给定的上下文无法让你做出回答，请回答你不知道。
    有用的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=template)

    # 运行 chain
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(llm,
                                        retriever=retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    
    return qa_chain


class Model_center():
    """
    存储问答 Chain 的对象 
    """
    def __init__(self):
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用不带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"query": question})["result"]))
            return "", chat_history
        except Exception as e:
            return e, chat_history


model_center = Model_center()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            gr.Markdown("""<h1><center>InternLM</center></h1>
                <center>你的专属量刑助手</center>
                """)
        # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
                
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])
        
    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
# threads to consume the request
gr.close_all()
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch()
