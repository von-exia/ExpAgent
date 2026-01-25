# ReAcTree - 智能决策树代理系统

## 项目简介

ReAcTree 是一个基于行为树（Behavior Tree）架构的智能决策代理系统，它结合了语言模型（LLM）与多类型任务执行能力，能够自主进行复杂任务分解和执行。该项目通过 ReAcTree 算法实现了一个层次化的决策框架，使代理能够在思考（Think）、行动（Act）和扩展（Expand）之间动态切换，从而解决复杂的多步骤问题。

## 核心特性

### 1. 行为树架构
- **TreeNode**: 基础节点类，支持树形结构构建
- **ControlFlowNode**: 控制流节点，支持三种控制流程：
  - **Sequence（序列）**: 顺序执行子任务，任一失败则中断
  - **Fallback（回退）**: 按序尝试子任务，任一成功则停止
  - **Parallel（并行）**: 并行执行子任务，独立处理失败情况

### 2. 智能决策机制
- **AgentNode**: 智能代理节点，负责决策制定
- 支持动态决策：思考（Think）、行动（Act）、扩展（Expand）
- 自适应任务分解与执行策略

### 3. 多模态工具集成
- **Shell 工具**: 执行系统命令
- **Wikipedia 搜索**: 实时知识检索
- **浏览器工具**: 网页浏览和交互
- **Python 代码执行**: 动态代码运行
- **文件查看工具**: 文本文件读取

### 4. 高级功能
- **技能系统（Skills）**: 可扩展的功能模块
- **实时 RAG**: 基于检索的增强生成
- **环境评估**: 任务完成度自动评估
- **层级规划**: 支持最多 10 层深度的任务分解

## 项目目录结构

```
ReAcTree/
├── __pycache__/              # Python 缓存文件目录
├── .vscode/                  # VSCode 配置文件目录
├── act/                      # 动作和工具模块
│   ├── __pycache__/          # Python 缓存文件
│   ├── actions/              # 基础动作定义（如 Answer）
│   ├── skills/               # 技能系统相关文件
│   ├── tools/                # 各种工具实现（Shell、Wikipedia搜索、浏览器等）
│   └── act_loader.py         # 动作加载器
├── agent_model/              # 代理模型核心模块
│   ├── __pycache__/          # Python 缓存文件
│   ├── agent_model.py        # 主代理类，包含核心逻辑
│   ├── env.py               # 环境管理类
│   └── rag.py               # RAG（检索增强生成）系统
├── debug/                    # 调试相关文件目录
├── hf_cache/                 # HuggingFace 模型缓存目录
├── planner/                  # 规划器模块
│   ├── ReAct/               # ReAct 算法实现（备用）
│   └── ReAcTree/            # ReAcTree 算法核心实现
│       ├── __pycache__/      # Python 缓存文件
│       ├── __init__.py       # 模块初始化文件
│       ├── config.py         # 配置类定义
│       ├── reactree.py       # ReAcTree 核心算法实现
│       └── system_prompt.txt # 系统提示词模板
├── Qwen2_5-7B-Instruct-MNN/  # Qwen2.5-7B 模型文件目录
├── Qwen3-0.6B-embedding/     # Qwen3-0.6B 嵌入模型文件目录
├── Qwen3-4B-MNN/            # Qwen3-4B 模型文件目录（默认使用）
├── Qwen3-8B-MNN/            # Qwen3-8B 模型文件目录
├── .gitignore               # Git 忽略文件配置
├── cmd.sh                   # Shell 命令脚本
├── download.py              # 模型和数据下载脚本
├── py_test.py               # Python 测试文件
├── README.md                # 项目说明文档
└── run.py                   # 主程序入口文件
```

## 架构设计

### 核心模块说明

- **act/**: 动作和工具模块，提供各种可执行的动作和外部工具接口
- **agent_model/**: 代理模型核心，包含代理的主要逻辑和环境交互
- **planner/ReAcTree/**: ReAcTree 规划算法的核心实现
- **Qwen*/**: 不同版本的语言模型文件

## 配置参数

- `max_steps`: 最大执行步数（默认 10）
- `max_decisions`: 最大决策次数（默认 10）
- `max_depth`: 最大树深度（默认 10）
- `max_think`: 最大思考次数（默认 1）
- `max_expand`: 最大扩展次数（默认 1）

## 使用方法

### 环境准备
```bash
# 安装依赖
pip install MNN agentscope

# 下载模型（根据需要选择）
# 默认使用 Qwen3-4B-MNN 模型
```

### 运行示例
```python
from agent_model.agent_model import AgentModel
from agent_model.env import Environment
from planner.ReAcTree import Config, ReAcTreePlanner
from act import *

# 配置代理
config_path = "./Qwen3-4B-MNN/"
agent = AgentModel(config_path)

# 创建规划器
cfg = Config()
env = Environment(critique=agent)
planner = ReAcTreePlanner(cfg=cfg, agent=agent, env=env)

# 执行查询
query = "Run the py_test.py file."
result = planner.collect(query)
print(result)
```

### 支持的模型
- Qwen3-4B-MNN (推荐)
- Qwen3-8B-MNN
- Qwen2_5-7B-Instruct-MNN

## 工作流程

1. **任务接收**: 接收用户查询或任务
2. **决策制定**: 代理决定是思考、行动还是扩展
3. **任务分解**: 如需扩展，将任务分解为子任务
4. **执行控制**: 根据控制流（序列/回退/并行）执行子任务
5. **结果反馈**: 返回执行结果或继续迭代

## 应用场景

- 自动化任务执行
- 复杂问题求解
- 知识检索与整合
- 系统管理与运维
- 研究辅助工具

## 依赖项

- Python 3.8+
- MNN (用于模型推理)
- Agentscope (代理框架)
- Hugging Face Datasets (可选，用于数据加载)

## 开发计划

- [ ] 增强浏览器工具功能
- [ ] 添加更多内置工具
- [ ] 优化性能和内存使用
- [ ] 支持更多语言模型
- [ ] 图形化界面开发

## 许可证

请参阅 LICENSE 文件获取详细信息。