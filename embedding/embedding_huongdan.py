from sentence_transformers import SentenceTransformer
import spacy
from tqdm.auto import tqdm # thanh trạng thái
import os
import getpass
import re
import weaviate
import pdfplumber

def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text


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

# Tách văn bản thành các mục con để dễ tìm kiếm
def split_section(file_path):
    sections=[{'title':"",'content':""}]
    text = ""
    cur_main_sec = ""
    cur_sub_main_sec = ""
    cur_sub_sub_main_sec = ""
    pages_and_texts = []
    # Đọc nội dung từ file PDF
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
        
        # Tìm các mục bằng regex
        for line in text.split('\n'):
            section_match = re.match(r'^([A-Z]+\.)*((\d+\.)(\d+\.)?(\d+\.)?)',line)
            if not section_match:
                if cur_main_sec == "":
                    sections[-1]['content'] += line + " "
                else:
                    sections[-1]['content'] += " " + line
            elif section_match.group(1):
                cur_main_sec = line
                sections.append({})
                sections[-1]['title'] = cur_main_sec
                sections[-1]['content'] = ""
            elif section_match.group(5):
                cur_sub_sub_main_sec = cur_main_sec +" "+ line
                sections.append({})
                sections[-1]['title'] = cur_sub_sub_main_sec
                sections[-1]['content'] = ""
            elif section_match.group(4):
                cur_sub_main_sec = cur_main_sec +" "+ line
                sections.append({})
                sections[-1]['title'] = cur_sub_main_sec
                sections[-1]['content'] = ""
            elif section_match.group(3):
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

    for item in tqdm(pages_and_texts):
        text = f"Văn bản trong {filename} có nội dung như sau:\n{item['text']}"
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

        pdf_folder_path = "DA2\Trợ lý ảo\Documents\Hướng dẫn"
        seq_len = 500
        # Lặp qua các file PDF để thực hiện embedding và lưu văn bản
        for filename in os.listdir(pdf_folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(pdf_folder_path, filename)
                file_path = file_path.replace("\\", "/")
                print(f"Đang đọc tệp: {file_path}")
                
                pages_and_texts = split_section(file_path)
                embedding(filename.strip(".pdf"),pages_and_texts)