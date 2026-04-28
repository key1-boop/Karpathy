import re
import ollama
import hashlib
import json
import pdfplumber
import os
import shutil
from pathlib import Path
from datetime import datetime

# ======================== 配置 ========================
RAW_DIR = Path("./raw")
WIKI_DIR = Path("./wiki")
OUTPUTS_DIR = Path("./outputs")
SESSION_DIR = OUTPUTS_DIR / "sessions"
GRAPH_EXPORT = Path("./knowledge_graph.json")
MODEL_NAME = "deepseek-r1:latest"
STATE_FILE = Path(".compile_state.json")
TODAY = datetime.now().strftime("%Y-%m-%d")

# ======================== 系统提示词（严格按老师要求） ========================
SYSTEM_PROMPT = f"""
你是一个知识库编译器，严格遵循 Karpathy 范式。根据原始资料，输出多个独立的 Markdown 页面，每个页面用 `===PAGE===` 单独分隔。

禁止输出任何推理过程、思考步骤、注释或额外说明。直接输出页面内容。

页面格式要求：
- 以 YAML frontmatter 开头，包含 title, date, tags, sources
- 正文包含四个二级标题：`## 摘要`、`## 详细介绍`、`## 相关概念`、`## 来源`
- 相关概念使用 `[[概念名]]` 双向链接
- 来源字段必须引用原始文件名

至少生成 10 个页面。
"""

# ======================== 1. Ingest 模块（导入文档） ========================
def load_raw_documents():
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"请将资料放入 {RAW_DIR}")
        return None
    texts = []
    for f in sorted(RAW_DIR.glob("*")):
        if f.suffix in [".txt", ".md"]:
            try:
                content = f.read_text(encoding="utf-8")
                texts.append(f"## 文件：{f.name}\n{content}")
                print(f"✅ 读取：{f.name}")
            except:
                print(f"❌ 读取失败：{f.name}")
        elif f.suffix == ".pdf":
            try:
                with pdfplumber.open(f) as pdf:
                    txt = "\n".join(page.extract_text() or "" for page in pdf.pages)
                if txt.strip():
                    texts.append(f"## 文件：{f.name}\n{txt}")
                    print(f"✅ PDF读取：{f.name}")
            except:
                print(f"❌ PDF失败：{f.name}")
    return "\n\n---\n\n".join(texts) if texts else None

# ======================== 2. Compile 模块（LLM编译知识库） ========================
def clean_model_output(out):
    if "instant" in out.lower():
        out = re.split(r"(?i)instant", out, maxsplit=1)[-1]
    out = re.sub(r"^```markdown\s*", "", out, flags=re.MULTILINE)
    out = re.sub(r"\s*```$", "", out, flags=re.MULTILINE)
    return out.strip()

def parse_and_save_pages(text):
    WIKI_DIR.mkdir(exist_ok=True)
    pages = [p.strip() for p in re.split(r"\n===PAGE===\n", text) if p.strip()]
    saved = 0
    for page in pages:
        title = re.search(r'title:\s*"([^"]+)"', page)
        title = title.group(1) if title else f"page_{saved+1}"
        title = re.sub(r'[\\/*?:"<>|]', "_", title)
        path = WIKI_DIR / f"{title}.md"
        path.write_text(page, encoding="utf-8")
        saved += 1
    return saved

def generate_readme():
    pages = sorted(f.stem for f in WIKI_DIR.glob("*.md") if f.stem != "README")
    content = f"""# Karpathy 知识库
自动编译日期：{TODAY}
页面总数：{len(pages)}

## 索引
"""
    for p in pages:
        content += f"- [[{p}]]\n"
    (WIKI_DIR / "README.md").write_text(content, encoding="utf-8")

def compile_knowledge_base():
    raw = load_raw_documents()
    if not raw:
        return
    print("\n🚀 开始编译知识库...")
    try:
        res = ollama.chat(model=MODEL_NAME, messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请编译：\n{raw}"}
        ], options={"temperature":0.2, "num_predict":16384})
        cleaned = clean_model_output(res["message"]["content"])
        cnt = parse_and_save_pages(cleaned)
        generate_readme()
        print(f"\n🎉 编译完成，生成 {cnt} 个页面")
    except Exception as e:
        print(f"❌ 编译失败：{e}")

# ======================== 3. Wiki 管理（已内置） ========================

# ======================== 4. Query 模块（问答 + 多轮对话） ========================
def load_wiki_all_text():
    txt = ""
    for f in WIKI_DIR.glob("*.md"):
        txt += f"\n--- {f.name} ---\n" + f.read_text(encoding="utf-8")
    return txt

def chat_once(question, history=None):
    wiki = load_wiki_all_text()
    hist = history or []
    messages = [{"role":"system","content":"你基于知识库回答，必须引用原文。"}]
    messages += hist
    messages.append({"role":"user","content":f"知识库：{wiki}\n问题：{question}"})
    res = ollama.chat(model=MODEL_NAME, messages=messages)
    return res["message"]["content"]

def multi_session_chat():
    SESSION_DIR.mkdir(exist_ok=True)
    hist = []
    print("\n💬 多轮对话（输入 exit 退出）")
    while True:
        q = input("你：")
        if q.lower() == "exit":
            break
        ans = chat_once(q, hist)
        print(f"AI：{ans}")
        hist.append({"role":"user","content":q})
        hist.append({"role":"assistant","content":ans})
    fn = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    (SESSION_DIR / fn).write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"✅ 对话已保存：{fn}")

# ======================== 5. Visualization + KG 导出 ========================
def export_knowledge_graph():
    nodes = []
    links = []
    all_pages = set()
    page_links = {}

    for f in WIKI_DIR.glob("*.md"):
        name = f.stem
        all_pages.add(name)
        txt = f.read_text(encoding="utf-8")
        links_found = re.findall(r"\[\[(.*?)\]\]", txt)
        page_links[name] = links_found

    for src, tgts in page_links.items():
        nodes.append({"id": src, "label": src})
        for t in tgts:
            if t in all_pages:
                links.append({"source": src, "target": t})

    data = {"nodes": nodes, "links": links}
    GRAPH_EXPORT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"✅ 知识图谱已导出：{GRAPH_EXPORT}（可导入Gephi/D3.js）")

# ======================== 6. Export 导出（PDF/HTML/MD） ========================
def export_all():
    OUTPUTS_DIR.mkdir(exist_ok=True)
    combined = "# 知识库导出\n"
    for f in sorted(WIKI_DIR.glob("*.md")):
        combined += f"\n# {f.stem}\n" + f.read_text(encoding="utf-8")
    out = OUTPUTS_DIR / "export_combined.md"
    out.write_text(combined, encoding="utf-8")
    print(f"✅ 导出完成：{out}")

# ======================== 进阶：增量编译 ========================
def incremental_compile():
    old = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    new = {}
    changed = []
    for f in RAW_DIR.glob("*"):
        if f.suffix in [".txt",".md",".pdf"]:
            h = hashlib.md5(f.read_bytes()).hexdigest()
            new[f.name] = h
            if old.get(f.name) != h:
                changed.append(f.name)
    STATE_FILE.write_text(json.dumps(new, indent=2))
    if changed:
        print(f"🔄 增量编译：{changed}")
        compile_knowledge_base()
    else:
        print("✅ 无变化，无需编译")

# ======================== 进阶：Auto-Lint 健康检查 ========================
def lint_knowledge_base():
    if not WIKI_DIR.exists():
        print("请先编译")
        return
    pages = set()
    links = set()
    content_len = {}
    for f in WIKI_DIR.glob("*.md"):
        n = f.stem
        pages.add(n)
        c = f.read_text(encoding="utf-8")
        content_len[n] = len(c)
        found = re.findall(r"\[\[(.*?)\]\]", c)
        links.update(found)

    dead = links - pages
    orphan = pages - links - {"README"}
    empty = [n for n, l in content_len.items() if l < 150]

    print("\n📊 知识库健康检查")
    print(f"总页面：{len(pages)}")
    print(f"死链：{len(dead)} → {dead if dead else '无'}")
    print(f"孤立页面：{len(orphan)} → {orphan if orphan else '无'}")
    print(f"内容过短：{len(empty)} → {empty if empty else '无'}")
    print("✅ 检查完成\n")

# ======================== 主入口 ========================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("使用方法：")
        print("python compile.py compile      # 全量编译")
        print("python compile.py inc          # 增量编译")
        print("python compile.py lint         # 健康检查")
        print("python query.py          # 多轮对话")

    else:
        cmd = sys.argv[1]
        if cmd == "compile": compile_knowledge_base()
        elif cmd == "inc": incremental_compile()
        elif cmd == "lint": lint_knowledge_base()
        elif cmd == "chat": multi_session_chat()
        elif cmd == "kgexport": export_knowledge_graph()
        elif cmd == "export": export_all()