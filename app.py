import os
import json
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

#---0. 환경 설정 ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY가 필요합니다.")
    st.stop()

#---1. Mock 데이터 ---
mork_crm_db={
    "C-1001":{"name":"김민준","level":"VIP","recent_order_id":"ORD-5678"},
    "C-2002":{"name":"이서연","level":"GOLD","recent_order_id":"ORD-5679"},
}

mock_knowledge_base={
    "환불":"VIP 고객은 구매 후 14일,일반 고객은 7일 이내에 환불 가능합니다.",
    "배송":"기본 배송 기간은 2-3일이며,주문 폭주 시 지연될수 있습니다."
}

#2.--- Tool 정의 ---
@tool
def search_crm(customer_id: str) -> str:
    """고객 ID를 사용하여 CRM 시스템에서 고객 정보(이름, 등급, 최근 주문 ID)를 검색합니다."""
    data=mork_crm_db.get(customer_id)
    return json.dumps(data,ensure_ascii=False) if data else "고객 정보를 찾을 수 없습니다."

@tool
def search_manual(query: str) -> str:
    """고객의 질문과 관련된 회사 정책을 내부 지식 베이스(메뉴얼)에서 검색합니다."""
    for keyword, content in mock_knowledge_base.items():
        if keyword in query:
            return content
    return "관련된 정보를 찾을 수 없습니다."

@tool
def process_refund(order_id: str,reason: str) ->str:
    """[다중 인자 도구] 특정 주문에 대해 환불을 처리합니다."""
    print(f"\nACTION: 환불 처리 API 호출(Order ID:{order_id},Reason: {reason})")
    return f"주문번호 '{order_id}에 대한 환불 처리가 성공적으로 완료되었습니다."

@tool
def check_shipping_status(order_id: str)->str:
    """배송 조회"""
    if order_id=="ORD-5679":
        return"오늘 도착 예정"
    return "배송 정보 없음"

@tool
def create_support_ticket(customer_id: str,issue_description: str)->str:
    """지원 티켓 생성 [다중 인자 도구] AI가 스스로 문제를 해결할 수 없을 때,인간 상담사에게 지원 티켓을 생성합니다."""
    print(f"\nACTION: 지원 티켓 생성 API 호출(Customer:{customer_id})")
    ticket_id="TICKET-9876"

    return f"문제를 해결할 수 없어 담당 팀에 지원 티켓을 생성했습니다.(티켓 번호:{ticket_id})"

tools=[
    search_crm,
    search_manual,
    process_refund,
    check_shipping_status,
    create_support_ticket
]

#---3.시스템 프롬프트---
def build_system_prompt(customer_id: str,user_query: str) ->str:
    customer=mork_crm_db.get(customer_id)
    knowledge=""

    for k, v in mock_knowledge_base.items():
        if k in user_query:
            knowledge = v

    role="공감 중심 상담원" if any(k in user_query for k in ["환불","불만","고장"])else "친절한 상담원"

    return f"""
            당신은 고객의 문제를 해결하는 '문제 해결 전문가'이며 {role}입니다.

            [고객 정보]
            {json.dumps(customer,ensure_ascii=False)}

            [지식]
            {knowledge}
            도구를 적절히 사용하여 고객 문제를 해결하세요.
            """

#---4. Agent 실행 함수---
def run_agent(llm_with_tools, customer_id, user_query, chat_history):

    system_prompt=build_system_prompt(customer_id,user_query)

    messages=[
        {"role":"system","content":system_prompt},
        *chat_history,
        {"role":"user","content":user_query}
    ]

    response=llm_with_tools.invoke(messages)

    #tool 호출 루프
    while hasattr(response,"tool_calls")and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name=tool_call["name"]
            tool_args=tool_call["args"]

            st.info(f"🔧 TOOL 호출:{tool_name} {tool_args}")

            tool_func=next(t for t in tools if t.name == tool_name)
            tool_result=tool_func.invoke(tool_args)

            messages.append(response)
            messages.append({
                "role":"tool",
                "content":tool_result,
                "tool_call_id":tool_call["id"]
            })

        response=llm_with_tools.invoke(messages)

    return response.content

#---5. Streamlit UI ---
st.set_page_config(page_title="AI_고객센터", layout="wide")

st.title("📞 AI 고객센터 (Langchain 최신 구조)")

#고객 선택
customer_id = st.selectbox(
    "고객 선택",
    ["C-1001","C-2002"]
)

#세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

if "messages" not in st.session_state:
    st.session_state.messages=[]

#LLM 초기화 (1회만)
if "llm" not in st.session_state:
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    st.session_state.llm_with_tools=llm.bind_tools(tools)

#채팅 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#사용자 입력
user_input=st.chat_input("문의 내용을 입력하세요...")

if user_input:
    #사용자 메시지 출력
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    #에이전트 실행
    with st.chat_message("assistant"):
        with st.spinner("답변 생성중..."):
            response=run_agent(
                st.session_state.llm_with_tools,
                customer_id,
                user_input,
                st.session_state.chat_history
            )
            st.markdown(response)

    #히스토리 저장
    st.session_state.messages.append({"role":"assistant","content":response})
    st.session_state.chat_history.append({"role":"user","content":user_input})
    st.session_state.chat_history.append({"role":"assistant","content":response})