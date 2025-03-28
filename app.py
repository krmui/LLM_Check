from flask import Flask, request, jsonify, render_template
import os
import re  # 新增导入
from datetime import datetime
from PyPDF2 import PdfReader
from docx import Document
from openai import OpenAI

app = Flask(__name__)

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 文件保存路径
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
        return text
    except Exception as e:
        raise ValueError(f"文件读取失败：{e}")

def format_model_response(response):
    """格式化大模型的回复，提取问题列表"""
    lines = response.split('\n')
    problems = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith(("招标文件", "投标技术方案")):
            # 使用正则表达式移除序号
            cleaned_line = re.sub(r'^\d+\.\s*', '', line)
            problems.append(cleaned_line)
    return problems

@app.route('/', methods=['GET'])
def index():
    """渲染前端页面"""
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_files():
    try:
        # 获取上传的文件
        tender_file = request.files.get('tenderFile')
        proposal_file = request.files.get('proposalFile')

        if not tender_file or not proposal_file:
            return jsonify({"error": "请上传招标文件和投标技术方案！"}), 400

        # 生成唯一的子文件夹名称
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        folder_name = f"{tender_file.filename.split('.')[0]}_{timestamp}"
        folder_path = os.path.join(UPLOAD_FOLDER, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # 保存文件到子文件夹中
        tender_path = os.path.join(folder_path, tender_file.filename)
        proposal_path = os.path.join(folder_path, proposal_file.filename)
        tender_file.save(tender_path)
        proposal_file.save(proposal_path)

        # 提取文本内容
        tender_text = extract_text_from_file(tender_path)
        proposal_text = extract_text_from_file(proposal_path)

        # 构造输入提示词
        input_prompt = (
            f"请分析以下招标文件和投标技术方案的内容，检查投标技术方案是否符合招标文件的要求。\n"
            f"如果发现问题，请列出具体问题，仅返回问题列表，不要包含其他信息。\n\n"
            f"招标文件内容：\n{tender_text}\n\n"
            f"投标技术方案内容：\n{proposal_text}"
        )

        # 调用大模型进行分析
        response = client.chat.completions.create(
            model="qwen-max-latest",
            messages=[
                {"role": "system", "content": "你是一个专业的招标文件分析助手，仅返回问题列表。"},
                {"role": "user", "content": input_prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        raw_response = response.choices[0].message.content.strip()

        # 格式化大模型的回复
        problems = format_model_response(raw_response)

        # 去重并保存结果（保持顺序）
        seen = set()
        unique_problems = []
        for problem in problems:
            if problem not in seen:
                seen.add(problem)
                unique_problems.append(problem)

        result_file_path = os.path.join(folder_path, "analysis_result.txt")
        with open(result_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(unique_problems))

        # 返回结果
        return jsonify({
            "problems": unique_problems,
            "result_file": result_file_path
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"服务器内部错误：{str(e)}"}), 500

if __name__ == '__main__':
    app.run()