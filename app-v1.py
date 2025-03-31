from flask import Flask, request, jsonify, render_template
import os
import re
from datetime import datetime
from PyPDF2 import PdfReader
from docx import Document
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def split_text(text, chunk_size=800, overlap=100):
    """将长文本分块处理"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start >= end:
            break
    return chunks


def extract_text_from_file(file_path):
    """从文件中提取文本内容"""
    try:
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = ''.join(page.extract_text() for page in reader.pages if page.extract_text())
        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError("不支持的文件格式")
        return text.replace('\n', ' ').strip()  # 清理换行符
    except Exception as e:
        raise ValueError(f"文件读取失败：{e}")


def format_model_response(response):
    """格式化大模型的回复"""
    lines = response.split('\n')
    problems = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith(("招标文件", "投标技术方案")):
            cleaned_line = re.sub(r'^\d+[\.、]?\s*', '', line)
            if cleaned_line:
                problems.append(cleaned_line)
    return problems


@app.route('/', methods=['GET'])
def index():
    return render_template('index-v1.html')


@app.route('/check', methods=['POST'])
def check_files():
    try:
        tender_file = request.files.get('tenderFile')
        proposal_file = request.files.get('proposalFile')

        if not tender_file or not proposal_file:
            return jsonify({"error": "请上传招标文件和投标技术方案！"}), 400

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        folder_name = f"{tender_file.filename.split('.')[0]}_{timestamp}"
        folder_path = os.path.join(UPLOAD_FOLDER, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        tender_path = os.path.join(folder_path, tender_file.filename)
        proposal_path = os.path.join(folder_path, proposal_file.filename)
        tender_file.save(tender_path)
        proposal_file.save(proposal_path)

        # 提取并分块文本
        tender_text = extract_text_from_file(tender_path)
        proposal_text = extract_text_from_file(proposal_path)

        # 分块参数设置（可根据模型实际输入限制调整）
        tender_chunks = split_text(tender_text, chunk_size=1000, overlap=200)
        proposal_chunks = split_text(proposal_text, chunk_size=1000, overlap=200)

        all_problems = []
        for t_chunk in tender_chunks[:5]:  # 限制最多处理前5个招标块
            for p_chunk in proposal_chunks[:5]:  # 限制最多处理前5个投标块
                input_prompt = (
                    f"请严格分析以下招标内容与投标内容的对应关系，列出投标中不符合招标要求的具体问题：\n"
                    f"【招标内容开始】\n{t_chunk}\n【招标内容结束】\n\n"
                    f"【投标内容开始】\n{p_chunk}\n【投标内容结束】\n\n"
                    f"要求：\n1. 只返回具体问题\n2. 不要编号\n3. 确保问题明确具体"
                )

                try:
                    response = client.chat.completions.create(
                        model="qwen-max-latest",
                        messages=[
                            {"role": "system", "content": "你是一个严谨的招标文件分析专家，只返回检查发现的具体问题"},
                            {"role": "user", "content": input_prompt}
                        ],
                        max_tokens=2000,
                        temperature=0.3  # 降低随机性
                    )
                    raw_response = response.choices[0].message.content.strip()
                    all_problems.extend(format_model_response(raw_response))
                except Exception as e:
                    print(f"API调用失败：{str(e)}")
                    continue

        # 高级去重（基于语义相似度的简单实现）
        seen = set()
        unique_problems = []
        for problem in all_problems:
            # 简化的相似度判断（实际应使用更复杂的方法）
            key = problem[:50].lower().replace(' ', '')  # 取前50字符作为近似判断
            if key not in seen:
                seen.add(key)
                unique_problems.append(problem)

        result_file_path = os.path.join(folder_path, "analysis_result.txt")
        with open(result_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(unique_problems))

        return jsonify({
            "problems": unique_problems,
            "result_file": result_file_path
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"服务器内部错误：{str(e)}"}), 500


if __name__ == '__main__':
    app.run(port=5001)