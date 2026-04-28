import ollama
from pathlib import Path

WIKI_DIR = Path("./wiki")

def load_entire_wiki():
    """加载整个wiki文件夹的所有Markdown文件"""
    all_content = []
    if not WIKI_DIR.exists():
        print("错误：没有找到 './wiki' 文件夹。请先运行 compile.py 来创建知识库。")
        return None
        
    for md_file in WIKI_DIR.glob("*.md"):
        content = md_file.read_text(encoding='utf-8')
        all_content.append(f"# {md_file.stem}\n{content}")
        print(f"已加载: {md_file.name}")
    
    return "\n\n---\n\n".join(all_content)

def ask_question(question, wiki_context):
    """基于整个知识库回答问题"""
    system_prompt = f"""
    你是一位知识渊博的助手。你将基于以下完整的知识库内容来回答用户的问题。
    你的回答必须准确，并严格遵守以下规则：
    1.  **引用来源**：在你回答的每个关键点后面，使用`[[文件名]]`的格式来标注信息来源。例如：`Transformer模型的核心是注意力机制[[Attention_Is_All_You_Need]]`。
    2.  **综合回答**：如果问题涉及多个概念，请综合知识库中不同页面的信息来组织一个完整的答案。
    
    ### 知识库内容：
    {wiki_context}
    """
    
    try:
        response = ollama.chat(
            model='deepseek-r1:latest',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            options={
                'temperature': 0.5,      # 稍微提高一点创造性，使回答更自然
            }
        )
        return response['message']['content']
    except Exception as e:
        return f"问答过程中发生错误: {e}"

if __name__ == "__main__":
    wiki_content = load_entire_wiki()
    if wiki_content is None:
        exit()
        
    print("\n知识库加载完成！请输入你的问题。")
    while True:
        user_question = input("\n你: ")
        if user_question.lower() == 'exit':
            print("退出问答系统。")
            break
        print("\nAI: ", end="")
        answer = ask_question(user_question, wiki_content)
        print(answer)