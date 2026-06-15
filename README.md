# LocalAI
尝试学习部署本地LLM模型

## Day1 实现一个小型的本地LLM

### 本地开发环境准备
``` shell

alias pip3="python3 -m pip"
alias pip="python3 -m pip"

# 1. 在你的电商 Agent 项目目录下，创建一个专属的 3.14 独立虚拟环境空间
python3 -m venv .venv

python3 -m pip install langchain chromadb

# 2. 激活这个虚拟环境
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# 3. 【重点】激活后，你会发现终端前面多了一个 (.venv)
# 此时，你不需要配任何环境变量，直接敲 `pip` 或 `pip3` 就能原地复活并使用了！
pip install langchain langchain-ollama chromadb


# 启动ollama后下载模型，大概300MB左右
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text
# curl确认模型下载成功
curl http://localhost:11434/api/tags

```

### HelloWorld
``` python

```

### 基于内存

## Day2 实现一个基于RAPG库的LLM

#### 
