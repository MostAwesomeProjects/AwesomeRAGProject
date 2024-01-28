# 首先导入所需第三方库
from BCEmbedding.tools.langchain import BCERerank

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import MarkdownHeaderTextSplitter  # markdown分割器

tar_path = "./law_data/刑法.md"

with open(tar_path, 'r', encoding='utf-8') as file:
    loaded_text = file.read()

# 准备分割的标题
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# 文档分割器
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# 分割文档
split_docs = markdown_splitter.split_text(loaded_text)

# print(split_docs)

# 构建向量数据库
embedding_model_name =  './model/bce-embedding-base_v1'  # 'maidalun1020/bce-embedding-base_v1'
embedding_model_kwargs = {'device': 'cuda:0'}
embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}
embeddings = HuggingFaceEmbeddings(
  model_name=embedding_model_name,
  model_kwargs=embedding_model_kwargs,
  encode_kwargs=embedding_encode_kwargs
)

faiss_index = FAISS.from_documents(split_docs, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)

# 保存索引到磁盘
faiss_index.save_local('./faiss_index')

# # 在将来需要的时候加载索引
# loaded_index = FAISS.read_index('path_to_saved_index')
# retriever = loaded_index.as_retriever(search_type="similarity", search_kwargs={"score_threshold": 0.3, "k": 10})

