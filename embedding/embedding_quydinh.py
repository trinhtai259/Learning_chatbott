from sentence_transformers import SentenceTransformer
import spacy
from tqdm.auto import tqdm # thanh trạng thái
import os
import getpass
import re
import weaviate
import pdfplumber

def extract_text_and_tables(pdf_path):
    text_segments = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:

            if page.extract_tables():
                page_tables = page.extract_tables()
                tables_text = tables_reader(page_tables)
                for table in tables_text:
                    table_text = " ".join(table)+"."
                    text_segments.append(table_text)
            # Trích xuất văn bản
            page_text = page.extract_text()
            text_segments.append(page_text)
            
        text = " ".join(text_segments)
        return text

def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip()
    cleaned_text = cleaned_text.replace("÷"," đến ")
    cleaned_text= re.sub(r'\.+', '.', cleaned_text)
    cleaned_text = cleaned_text.replace(":"," ")
    cleaned_text = cleaned_text.replace("<","nhỏ hơn")
    cleaned_text = cleaned_text.replace("≥","lớn hơn bằng")
    cleaned_text = cleaned_text.replace(">","lớn hơn")
    cleaned_text = cleaned_text.replace("≤","nhỏ hơn bằng")
    return cleaned_text

def tables_reader(tables):
    table_content = []
    for table in tables:
        if len(table[0]) >= 5:
            table = list(map(list, zip(*table)))
        titles = table.pop(0)
        table_content.append([])
        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j] == None:
                    table[i][j] = table[i-1][j]
                table[i][j] = titles[j] + " " + table[i][j]

            table_content[-1].append(" ".join(table[i]).replace('\n'," ")+". ")
    return table_content

# tách list thành các đoạn với độ dài mong muốn
def split_chunk(input_list: list) -> list[list[str]]:
    slice_size = 2
    list_sentece = [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]
    chunks = []
    # print(slice_size)
    for chunk in list_sentece:            
        chunk = "".join(chunk).replace("  ", " ").strip()
        if len(chunk) > seq_len:
            nlp = spacy.blank("vi")
            nlp.add_pipe("sentencizer")
            chunk = list(nlp(chunk).sents)
            chunk = [str(sentence) for sentence in chunk]
            chunks.extend(chunk)
        else:
            chunks.append(chunk)
    return chunks

def sentences_split(texts):
    nlp = spacy.blank("vi")
    nlp.add_pipe("sentencizer") # Thêm pipeline sentencizer giúp phân đoạn câu
    texts = list(nlp(texts).sents) # gán vào pipeline
    texts = [str(sentence) for sentence in texts]
    for sentence in texts:
        if len(sentence) > seq_len:
            texts.remove(sentence)
            sentence = sentence.replace(";", ".")
            sentence = list(nlp(sentence).sents) # gán vào pipeline
            for item in sentence:
                texts.append(str(item))
    texts = split_chunk(texts)
    return texts

# Tách văn bản quy định thành các phần dựa trên từ khóa 'Điều' --> dễ tìm kiếm hơn
def split_regulation(pdf_path: str)-> list[dict]:

    pages_and_texts = []
    sections=[{'title':"",'content':""}]
    text = extract_text_and_tables(pdf_path)
    cur_main_sec = ""

    for line in text.split('\n'):
        # Tìm tiêu đề (dòng đầu tiên)
        title_match = re.match(r"(Điều\s\d+\.)", line)
        
        if not title_match:
            if cur_main_sec == "":
                sections[-1]['content'] += line + " "
            else:
                sections[-1]['content'] += " " + line   
        else:
            cur_main_sec = line
            sections.append({})
            sections[-1]['title'] = cur_main_sec
            sections[-1]['content'] = ""

    for section in sections:
            cleaned_content = text_formatter(section['content'])
            len_content = len(cleaned_content)
            
            if len_content > seq_len:
                cleaned_content = sentences_split(cleaned_content)
            else:
                cleaned_content = [cleaned_content]
            for i in range (len(cleaned_content)):
                pages_and_texts.append({"char_count": len(cleaned_content[i])+len(section['title']),
                                        "token_count": (len(cleaned_content[i])+len(section['title'])) / 4,  # 1 token = ~4 chars
                                        "text": f"{section['title']}\n {cleaned_content[i]}"})
    return pages_and_texts

# embedding văn bản
def embedding(filename,pages_and_texts):
    embedding_model = SentenceTransformer('dangvantuan/vietnamese-embedding', device='cpu')
    i=0
    for item in tqdm(pages_and_texts):
        text = f"Văn bản trong {filename} có nội dung như sau:\n {item['text']}"
        print(text)
        i+=1
        if i > 5:
            break
        item["embedding"] = embedding_model.encode(text)
    
        client.data_object.create(data_object={"title":filename,
                                               "content":text},
                                  class_name="Document",
                                  vector = item["embedding"]
                                  )

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
        return os.environ[var]
    
if __name__ == "__main__":
    
    WEAVIATE_URL = _set_env("WEAVIATE_URL")
    WEAVIATE_API_KEY = _set_env("WEAVIATE_API_KEY")
    client = weaviate.Client(url=WEAVIATE_URL,
                             auth_client_secret=weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY),)
    
    # Kiểm tra kết nối
    if client.is_ready():
        pdf_folder_path = "DA2\Trợ lý ảo\Documents\Quy định"
        seq_len = 400
        # Lặp qua các file PDF để thực hiện embedding và lưu văn bản
        for filename in os.listdir(pdf_folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(pdf_folder_path, filename)
                file_path = file_path.replace("\\", "/")
                print(f"Đang đọc tệp: {file_path}")
                
                pages_and_texts = split_regulation(file_path)
                embedding(filename.strip(".pdf"),pages_and_texts)