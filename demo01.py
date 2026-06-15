import sys
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("=========================================")
print("🚀 正在初始化本地电商 AI 客服系统...")
print("=========================================")

try:
    # 1. 初始化 270MB 的超轻量向量模型 (用于文本数字化)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # 2. 初始化 390MB 的超轻量对话大模型 (用于思考和组织语言)
    # temperature=0 代表让大模型严格按照事实回答，不瞎编
    llm = ChatOllama(model="qwen2.5:0.5b", temperature=0)
except Exception as e:
    print(f"❌ 初始化模型失败，请确保 Ollama.app 已打开！错误信息: {e}")
    sys.exit(1)

# 3. 准备私有的电商规则知识库（大数据工程师的洗数据成果）
print("\n📦 正在加载电商私有知识库...")
ecommerce_knowledge = [
    Document(page_content="【极速退款规章】尊享会员在下单后 2 小时内取消订单，享有极速退款绿色通道，款项秒到账。"),
    Document(page_content="【生鲜退换货规章】生鲜类商品（如新鲜水果、冷冻海鲜、车厘子）由于易腐烂，平台不支持 7 天无理由退货。"),
    Document(page_content="【商品详情】商品名称：智利进口车厘子 JJ 级别 2kg 礼盒装。特征：冷链直达，适合节日送礼，单果果径 28-30mm。")
]

# 4. 将知识库写入内存级别的 Chroma 向量数据库
print("💾 正在将文本转化为语义向量并存入数据库...")
try:
    vector_db = Chroma.from_documents(ecommerce_knowledge, embeddings)
    # 将向量库转为检索器，k=1 代表只捞取最相关的一条规章，节省算力
    retriever = vector_db.as_retriever(search_kwargs={"k": 1})
except Exception as e:
    print(f"\n❌ 向量化失败！你大概率漏掉了向量模型。请在终端执行: ollama pull nomic-embed-text")
    print(f"原始报错信息: {e}")
    sys.exit(1)

# 5. 设计 RAG 专用的 Prompt（提示词模板）
# 告诉大模型：你只能根据我喂给你的【Context】来回答，不能自己瞎编
prompt_template = """你是一个电商平台的智能客服助理。请根据下面提供的【核心规章上下文】来专业、礼貌地回答用户的问题。
如果你不知道答案，就直接说不知道，绝对不能自己编造。

【核心规章上下文】：
{context}

用户提问：{question}
智能客服官方回答："""

prompt = ChatPromptTemplate.from_template(prompt_template)

# 6. 用 LangChain 表达式（LCEL）把整个 RAG 管道串联起来
# 数据流：用户提问 -> 检索器查知识 -> 填充提示词 -> 喂给大模型 -> 打印出文本
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. 模拟用户的高级语义提问
# 注意：问题里没有出现“生鲜”或“规章”等字眼，完全考验模型的语义联想能力
user_question = "我昨天在你们这买的智利车厘子礼盒，收到后不想要了，能走 7 天无理由退货吗？"

print(f"\n👤 用户提问: '{user_question}'")
print("🔍 AI 正在翻阅向量数据库并思考中...")

try:
    # 激活管道，获取答案
    ai_response = rag_chain.invoke(user_question)
    
    print("\n=================== 🤖 AI 客服在线回复 ===================")
    print(ai_response)
    print("=========================================================\n")
except Exception as e:
    print(f"❌ 管道运行出错: {e}")

print("🎉 链路测试完全闭环！")
