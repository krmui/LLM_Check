from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import re
from datetime import datetime
from PyPDF2 import PdfReader
from docx import Document
from openai import OpenAI
import hashlib
from difflib import SequenceMatcher
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def extract_structured_text(file_path):
    """提取带结构的文本内容"""
    try:
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            structured_text = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    for line in lines:
                        if line.strip():
                            structured_text.append({
                                "text": line.strip(),
                                "page": page_num + 1,
                                "type": "line"
                            })
            return structured_text
        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            return [{"text": para.text.strip(), "type": "paragraph"}
                    for para in doc.paragraphs if para.text.strip()]
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return [{"text": line.strip(), "type": "line"}
                        for line in f if line.strip()]
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        raise ValueError(f"文件解析失败：{str(e)}")

def get_embedding(text):
    """获取文本语义向量"""
    try:
        response = client.embeddings.create(
            model="text-embedding-v3",
            input=text,
            dimensions=1024,  # 使用1024维向量
            encoding_format="float"
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Embedding API Error: {str(e)}")
        return None

def add_markers(structured_data):
    """为结构化文本添加定位标记"""
    marked_data = []
    for idx, item in enumerate(structured_data):
        marker = f"[LOC_{idx}]"
        marked_item = {
            "original": item,
            "marked_text": f"{marker}{item['text']}",
            "marker": marker,
            "index": idx
        }
        marked_data.append(marked_item)
    return marked_data


def parse_problems(response_text):
    """解析模型响应中的问题定位"""
    problems = []
    pattern = r'\[LOC_(\d+)\]'

    lines = response_text.split('\n')
    for line in lines:
        line = line.strip()
        if line:
            matches = re.findall(pattern, line)
            if matches:
                locations = list(set([int(m) for m in matches]))
                clean_text = re.sub(pattern, '', line).strip()
                problems.append({
                    "description": clean_text,
                    "locations": locations
                })
    return problems


def deduplicate_problems(problems, similarity_threshold=0.85):
    """支持语义相似度判重的去重算法"""
    seen_hashes = set()
    semantic_cache = {}
    unique_problems = []

    for problem in problems:
        # 标准化处理
        sorted_locs = sorted(problem['locations'])
        loc_str = '-'.join(map(str, sorted_locs))
        clean_desc = re.sub(r'\s+', ' ', problem['description']).strip().lower()

        # 生成哈希键
        content_hash = hashlib.md5(clean_desc.encode()).hexdigest()
        composite_key = f"{content_hash}|{loc_str}"

        # 快速哈希过滤
        if composite_key in seen_hashes:
            continue

        # 获取语义向量
        semantic_vec = get_embedding(clean_desc)

        # 降级策略：当API失败时仅用哈希判断
        if semantic_vec is None:
            if composite_key not in seen_hashes:
                seen_hashes.add(composite_key)
                unique_problems.append(problem)
            continue

        # 语义相似度校验
        is_duplicate = False
        for uid, data in semantic_cache.items():
            if sorted_locs == data['locations']:
                similarity = cosine_similarity(
                    [semantic_vec],
                    [data['embedding']]
                )[0][0]
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            seen_hashes.add(composite_key)
            semantic_cache[f"prob_{len(semantic_cache)}"] = {
                "embedding": semantic_vec,
                "locations": sorted_locs
            }
            unique_problems.append(problem)

    return unique_problems

@app.route('/', methods=['GET'])
def index():
    return render_template('index-v3.html')


@app.route('/check', methods=['POST'])
def check_files():
    try:
        # 文件上传处理
        tender_file = request.files.get('tenderFile')
        proposal_file = request.files.get('proposalFile')

        if not tender_file or not proposal_file:
            return jsonify({"error": "请上传招标文件和投标技术方案！"}), 400

        # 创建存储目录
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        folder_name = f"{tender_file.filename.split('.')[0]}_{timestamp}"
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # 保存文件
        tender_path = os.path.join(folder_path, tender_file.filename)
        proposal_path = os.path.join(folder_path, proposal_file.filename)
        tender_file.save(tender_path)
        proposal_file.save(proposal_path)

        # 处理招标文件
        tender_structure = extract_structured_text(tender_path)
        marked_tender = add_markers(tender_structure)
        tender_text = " ".join([item["marked_text"] for item in marked_tender])

        # 处理投标文件
        proposal_structure = extract_structured_text(proposal_path)
        marked_proposal = add_markers(proposal_structure)
        proposal_text = " ".join([item["marked_text"] for item in marked_proposal])

        # 分块处理
        chunk_size = 1000
        overlap = 200
        tender_chunks = [tender_text[i:i + chunk_size]
                         for i in range(0, len(tender_text), chunk_size - overlap)]
        proposal_chunks = [proposal_text[i:i + chunk_size]
                           for i in range(0, len(proposal_text), chunk_size - overlap)]

        all_problems = []
        for t_chunk in tender_chunks[:3]:  # 限制处理块数
            for p_chunk in proposal_chunks[:3]:
                prompt = f"""请严格分析投标内容是否符合招标要求：
                【招标内容】{t_chunk}
                【投标内容】{p_chunk}
                要求：
                1. 指出具体不符合的条目
                2. 在问题后标注来源招标文件位置（如：[LOC_123]）
                3. 只返回问题内容，每个问题不要带编号"""

                try:
                    response = client.chat.completions.create(
                        model="qwen-max-latest",
                        messages=[{
                            "role": "system",
                            "content": "你是一个严谨的招投标文件分析专家"
                        }, {
                            "role": "user",
                            "content": prompt
                        }],
                        temperature=0.3
                    )
                    raw_response = response.choices[0].message.content
                    all_problems.extend(parse_problems(raw_response))
                except Exception as e:
                    print(f"API错误：{str(e)}")

        # 去重处理
        unique_problems = deduplicate_problems(all_problems)

        return jsonify({
            "tender": [item["original"] for item in marked_tender],
            "proposal": [item["original"] for item in marked_proposal],
            "problems": unique_problems
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/uploads/<path:filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(port=5003)