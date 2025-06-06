from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import os
import getpass
import weaviate
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
import json
import operator
from typing import List, Annotated
from langgraph.graph import END
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState

class GraphState(MessagesState):
    question: str
    generation: str
    web_search: str
    max_retries: int
    answers: int # yes hoặc no
    loop_step: Annotated[int, operator.add]
    documents: List[str]
    web_doc: List[str]
    summary: str

# Tìm kiếm tài liệu bằng Weaviate
def search(query,k):
  query_embedding = embedding_model.encode(query)
  result =(
    client.query
    .get("Document", ["title","content"])
    .with_hybrid(
        query=query,
        vector=query_embedding
    )
    .with_limit(k)
    .do()
)
  docs=[]
  
  for content in result["data"]["Get"]["Document"]:
    docs.append(Document(page_content= content['content']))
  return docs

# Định dạng lại tài liệu
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Tìm kiếm tài liệu
def retrieve(state):
    question =  state["messages"][-1].content

    documents = search(question,10)
    return {"documents": documents}

# Tạo câu trả lời
def generate(state):
    # Promt tạo câu trả lời
    rag_prompt = """Bạn là một cố vấn học tập cho nhiệm vụ trả lời câu hỏi và giao tiếp sinh viên Đại học Bách khoa Hà Nội (ĐHBKHN) hay hay Hanoi University of Science and Technology (HUST).
    Nếu là câu hỏi, thì hãy xem xét các yêu cầu sau
    Ngữ cảnh từ hội thoại:
    "{summary}"
    Ngữ cảnh từ tài liệu của đại học:
    "{context}"
    Ngữ cảnh từ web:
    "{web_context}"
    Bây giờ, hãy xem câu hỏi của người dùng:
    "{question}?"
    Hãy dành thời gian suy nghĩ kỹ (nhất là các câu về số) ngữ cảnh trên, ta cần ưu tiên ngữ cảnh từ tài liệu của đại học và hội thoại, sau đó mới cung cấp câu trả lời cho câu hỏi này.
    Giữ cho câu trả lời ngắn gọn, dễ hiểu, phù hợp với đối tượng sinh viên, trả lời câu hỏi và kết thúc việc hỏi sinh viên còn câu hỏi nào không.
    Lưu ý trả lời là Đại học Bách khoa Hà Nội, không có chữ trường.
    Trả lời:
    """

    question = state["messages"][-1].content
    documents = state["documents"]
    websearch = state.get("web_doc", "")
    loop_step = state.get("loop_step", 0)
    summary = state.get("summary", "")

    # RAG generation
    docs_txt = format_docs(documents)
    web_txt = format_docs(websearch)
    rag_prompt_formatted = rag_prompt.format(summary = summary,context=docs_txt,web_context = web_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}

def normal_conversation(state):
    normal_conversation_instruction = """Bạn là cố vấn học tập của Đại học Bách khoa Hà Nội (Hanoi University of Science and Technology), khi gặp các câu giao tiếp thông thường nhớ nói bạn là ai và sẵn sàng giúp đỡ nhé!
    Lưu ý là đại học chứ không phải trường đại học"""
    summary = state.get("summary", "")
    question = [state["messages"][-1]]
    # If there is summary, then we add it
    if summary:
        # Add summary to system message
        summary_message = f"Tóm tắt cuộc hội thoại từ trước: {summary}"

        messages = [SystemMessage(content = summary_message + normal_conversation_instruction)] + question
    else:
        messages = [SystemMessage(content = normal_conversation_instruction)] + question

    generation = llm.invoke(messages)
    return {"messages": generation}

# Kiểm tra từng tài liêu phù hợp không
def grade_documents(state):
    # Prompt chấm điểm tài liệu
    doc_grader_instructions = """Bạn là người chấm điểm đánh giá mức độ liên quan của một tài liệu đã truy xuất đến câu hỏi của người dùng.
    Nếu tài liệu chứa từ khóa hoặc ý nghĩa ngữ nghĩa liên quan đến câu hỏi, hãy chấm điểm là có liên quan."""

    doc_grader_prompt = """Đây là tài liệu đã truy xuất: \n\n {document} \n\n Đây là câu hỏi của người dùng: \n\n {question}?.
    Kết hợp với bản tóm tắt hội thoại: "{summary}". Đánh giá cẩn thận và khách quan xem tài liệu có chứa thông tin liên quan đến câu hỏi và hội thoại hay không.
    Trả về JSON với khóa "binary_score" là đánh giá chỉ gồm 'YES' hoặc 'NO' để xem liệu tài liệu có chứa thông tin liên quan đến câu hỏi hay không và một khoá giải thích tại sao."""

    question = state["messages"][-1].content
    documents = state["documents"]
    summary = state.get("summary", "")
    filtered_docs = []
    web_search = "NO"
    
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            summary=summary,document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]  
        
        if grade == "YES":
            filtered_docs.append(d)
        else:
            web_search = "YES"
            continue
    return {"documents": filtered_docs, "web_search": web_search}


def web_search(state):
    # Web search
    web_search_tool = TavilySearchResults(k=2)
    
    question = state["messages"][-1].content
    summary = state.get("summary", "")
    key = "hoặc HUST - Đại học Bách khoa Hà Nội"
    # Web search
    summary_doc = web_search_tool.invoke({"query": summary+key})
    docs = web_search_tool.invoke({"query": question+key})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = [Document(metadata={},page_content=web_results)]
    return {"web_doc": web_results}

# Xác định sử dụng web search hay tài liệu để trả lời
def route_question(state):
    # Promt định tuyến
    router_instruction = """Bạn là chuyên gia trong việc định tuyến câu hỏi của người dùng đến vectorstore hoặc tìm kiếm trên web hoặc giao tiếp thông thường.
    Vectorstore chứa các tài liệu liên quan đến quy định và học tập của Đại học Bách khoa Hà Nội (ĐHBKHN) hay Hanoi University of Science and Technology (HUST).
    Sử dụng vectorstore cho các câu hỏi về các chủ đề này. Đối với tất cả các chủ đề khác, và đặc biệt là các sự kiện hiện tại, hãy sử dụng web-search. Đối với câu giao tiếp thông thường hãy sử dụng normal-conversation.
    Và hãy dựa thêm bản tóm tắt hội thoại (nếu có) sau để quyết định: "{summary}".
    Trả về JSON với khóa duy nhất là 'source' có giá trị chỉ gồm 'websearch' hoặc 'vectorstore' tùy thuộc vào câu hỏi (Nếu là câu giao tiếp thông thường thì trả về 'normal-conversation')"""
    
    summary = state.get("summary", "")
    
    router_instruction_formatted = router_instruction.format(summary=summary)
    
    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instruction_formatted)]
        + [state["messages"][-1]]
    )
    source = json.loads(route_question.content)["source"]
    if source == "websearch":
        return "websearch"
    elif source == "vectorstore":
        return "vectorstore"
    elif source == "normal-conversation":
        return "normal_conversation"

# Xác định xem có tạo câi trả lời không hay thêm websearch
def decide_to_generate(state):
    web_search = state["web_search"]

    if web_search == "YES":
        # Có tài liệu không phù hợp
        return "websearch"
    else:
        # Tất cả tài liệu phù hợp
        return "generate"
    
# Xác định xem câu trả lời có dựa vào tài liệu và trả lời câu hỏi không
def grade_generation_v_documents_and_question(state):
    # Promt kiểm tra ảo giác
    hallucination_grader_instructions = """
    Bạn là giáo viên chấm bài kiểm tra.
    Bạn sẽ được cung cấp SỰ THẬT và CÂU TRẢ LỜI CỦA HỌC SINH.
    Sau đây là tiêu chí chấm điểm cần tuân theo:

    (1) Đảm bảo CÂU TRẢ LỜI CỦA HỌC SINH dựa trên SỰ THẬT.
    (2) Đảm bảo CÂU TRẢ LỜI CỦA HỌC SINH không chứa thông tin "ảo giác" nằm ngoài phạm vi của SỰ THẬT.

    Điểm:
    Điểm 'YES' có nghĩa là câu trả lời của học sinh đáp ứng tất cả các tiêu chí. Đây là điểm cao nhất (tốt nhất).
    Điểm 'NO' có nghĩa là câu trả lời của học sinh không đáp ứng tất cả các tiêu chí. Đây là điểm thấp nhất có thể mà bạn có thể cho.
    """

    hallucination_grader_prompt = """Văn bản từ đại học: \n\n {documents} \n {websearch} \n\n Câu trả lời của học sinh: {generation}.

    Trả về JSON với hai khóa, binary_score là điểm 'YES' hoặc 'NO' để xem liệu CÂU TRẢ LỜI CỦA HỌC SINH có dựa trên văn bản hay không."""

    # Prompt chấm điểm câu trả lời
    answer_grader_instructions = """Bạn là giáo viên chấm bài kiểm tra.

    Bạn sẽ được đưa ra một CÂU HỎI và một CÂU TRẢ LỜI CỦA HỌC SINH cùng với ngữ cảnh để quyết định.
    Sau đây là tiêu chí chấm điểm cần tuân theo:

    (1) CÂU TRẢ LỜI CỦA HỌC SINH giúp trả lời CÂU HỎI dựa trên ngữ cảnh (nếu có)

    Điểm:
    Điểm 'YES' nghĩa là câu trả lời của học sinh đáp ứng tất cả các tiêu chí. Đây là điểm cao nhất (tốt nhất).
    Học sinh có thể nhận được điểm 'YES' nếu câu trả lời có chứa thông tin bổ sung không được yêu cầu rõ ràng trong câu hỏi.
    Điểm 'NO' có nghĩa là câu trả lời của học sinh không đáp ứng tất cả các tiêu chí. Đây là điểm thấp nhất có thể mà bạn có thể cho."""

    answer_grader_prompt = """QUESTION: \n\n {question}? \n\n STUDENT ANSWER: {generation}\n\n Ngữ cảnh: "{summary}".
    Trả về JSON với hai khóa, binary_score là chỉ gồm 'YES' hoặc 'NO' để xem liệu STUDENT ANSWER có đáp ứng tiêu chí hay không."""
    
    question = state["messages"][-1].content
    documents = state["documents"]
    summary = state.get("summary", "")
    
    if "web_doc" in state:
        websearch = state["web_doc"]
    else:
        websearch = ""
    generation = state["generation"]
    max_retries = state.get("max_retries", 1)

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents),websearch = format_docs(websearch), generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]

    # Kiểm tra ảo giác
    if grade == "YES":
        # Kiểm câu trả lời có giải quyết câu hỏi không
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            summary=summary,question=question, generation=generation.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "YES":
            # Câu trả lời giải quyết được câu hỏi
            return "useful"
        elif state["loop_step"] <= max_retries:
            # Câu trả lời không giải quyết được câu hỏi
            return "not useful"
        else:
            # Vượt mức thử lại
            return "max retries"
    elif state["loop_step"] <= max_retries:
        # Nếu câu trả lời không dựa vào tài liệu
        return "not supported"
    else:
        # Vượt mức thử lại
        return "max retries"

def summarize_conversation(state):

    generation = state.get("generation", "")
    if generation:
      state['messages'].append(generation)  
    summary = state.get("summary", "")

    if summary:
        summary_message = (f"Đây là tóm tắt cuộc hội thoại cho đến hiện tại: {summary}\n\n"
                           "Mở rộng bản tóm tắt bằng cách lưu ý đến các thông điệp mới ở trên và cả hiện tại, không cần nói gì thêm:")
    else:
        summary_message = "Tạo một bản tóm tắt của cuộc hội thoại trên và không cần nói gì thêm: "

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    if len(state["messages"]) > 6:
      delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
      return {"summary": response.content, "messages": delete_messages}
    return {"summary": response.content}

def reply(question):
    global graph
    # Chạy mô hình
    
    user_message = HumanMessage(content=question)
    model_inputs = {
            "messages": [user_message],
            "max_retries": 1,
        }

    model_response = graph.invoke(model_inputs,model_config)
    
    return model_response['messages'][-1].content

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
        return os.environ[var]
### ===============================================================Main=============================================================== ###

# Kết nối các API
_set_env("LANGCHAIN_API_KEY")
_set_env("TAVILY_API_KEY")
_set_env("GROQ_API_KEY")
WEAVIATE_URL = _set_env("WEAVIATE_URL")
WEAVIATE_API_KEY = _set_env("WEAVIATE_API_KEY")
# Kết nối langchain và websearch
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "QA-answering"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"


# Kết nối CSDL vector

client = weaviate.Client(url=WEAVIATE_URL                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ,
                        auth_client_secret=weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY))
# Tạo llm với chatgroq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2)

llm_json_mode = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    response_format={"type": "json_object"})

embedding_model = SentenceTransformer('dangvantuan/vietnamese-embedding', device='cpu')

workflow = StateGraph(GraphState)
# Định nghĩa các nút
workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("normal_conversation", normal_conversation)
workflow.add_node(summarize_conversation)

# Kết nối các nút
workflow.set_conditional_entry_point(
    route_question,
    {
        "normal_conversation": "normal_conversation",
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": "summarize_conversation",
        "not useful": "websearch",
        "max retries": "summarize_conversation",
    },
)

workflow.add_edge("normal_conversation","summarize_conversation")
workflow.add_edge("summarize_conversation", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
model_config = {"configurable": {"thread_id": "1"}}

# reply("")