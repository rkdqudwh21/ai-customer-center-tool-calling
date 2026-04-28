import os
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

#---0. 환경 설정 ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY 필요")

#---1. 가상 CRM Mock 데이터 ---
mock_crm_db={
    "C-1001":{"name":"김민준","level":"VIP","recent_order_id":"ORD-5678"},
    "C-2002":{"name":"이서연","level":"GOLD","recent_order_id":"ORD-5679"},
}

#가상 지식 베이스
mock_knowledge_base={
    "환불":"VIP 고객은 구매 후 14일,일반 고객은 7일 이내에 환불 가능합니다.",
    "배송":"기본 배송 기간은 2-3일이며, 주문 폭주시 지연될 수도 있습니다.",
}

#---2.Tool 정의---
@tool
def search_crm(customer_id: str) -> str:
    """고객 ID를 사용하여 CRM시스템에서 고객 정보(이름,등급,최근 주문 ID)를 검색합니다."""
    data=mock_crm_db.get(customer_id)
    return json.dumps(data,ensure_ascii=False) if data else "고객 정보를 찾을 수 없습니다."

@tool
def search_manual(query:str) -> str:
    """고객의 질문과 관련된 회사 정책을 내부 지식 베이스(메뉴얼)에서 검색합니다."""
    for keyword, content in mock_knowledge_base.items():
        if keyword in query:
            return content
    return "관련된 정보를 찾을 수 없습니다."

@tool
def process_refund(order_id: str, reason: str) -> str:
    """[다중 인자 도구] 특정 주문에 대해 환불을 처리합니다. """
    print(f"\nACTION: 환불 처리 API 호출(Order ID: {order_id},Reason: {reason})")
    return f"주문번호 '{order_id}'에 대한 환불처리가 성공적으로 완료되었습니다."

@tool
def check_shipping_status(order_id: str) ->str:
    """배송 조회"""
    if order_id=="ORD-5679":
        return "오늘 도착 예정"
    return"배송 정보 없음"

@tool
def create_support_ticket(customer_id: str,issue_description: str) -> str:
    """지원티켓 생성 [다중 인자 도구] AI가 스스로 문제를 해결할 수 없을 때 인간 상담사에게 지원 티켓을 생성합니다."""
    print(f"\nACTION: 지원 티켓 생성 API 호출(Customer:{customer_id})")
    ticket_id="TICKET-9876"

    return f"문제를 해결할 수 없어 담당 팀에게 지원 티켓을 생성했습니다.(티켓 번호:{ticket_id})"

tools=[
    search_crm,
    search_manual,
    process_refund,
    check_shipping_status,
    create_support_ticket
]

#---3. 컨텍스트 생성---

def build_system_prompt(customer_id: str,user_query: str) ->str:
    customer=mock_crm_db.get(customer_id)
    knowledge=""
    for k, v in mock_knowledge_base.items():
        if k in user_query:
            knowledge = v

    if any(k in user_query for k in["환불", "불만","고장"]):
        role="공감 중심 상담원"
    else:
        role="친절한 상담원"

    return f"""
            당신은 고객의 문제를 해결하는 '문제 해결 전문가' 이며 {role}입니다.

            [고객 정보]
            {json.dumps(customer,ensure_ascii=False)}

            [지식]
            {knowledge}

            도구를 적절히 사용하여 고객 문제를 해결하세요.
            """
#---4.Agent 클래스---
class CustomerServiceAgent:

    def __init__(self):
        self.lim=ChatOpenAI(model="gpt-5-nano",temperature=0)

        #최신 방식: tool 바인딩
        self.lim_with_tools=self.lim.bind_tools(tools)

        #대화 히스토리 직접 관리
        self.chat_history=[]

        print("최신 bind_tools 기반 Agent 준비 완료.")
    
    def run(self,customer_id: str,query: str):

        print(f"\n[요청] {customer_id}:{query}")

        system_prompt=build_system_prompt(customer_id,query)

        #메시지 구성
        messages=[
            {"role":"system","content":system_prompt},
            *self.chat_history,
            {"role":"user","content":query}
        ]

        #1차 호출
        response=self.lim_with_tools.invoke(messages)

        #tool 호출 처리 루프
        while hasattr(response,"tool_calls")and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name=tool_call["name"]
                tool_args=tool_call["args"]

                print(f"\n[TOOL 호출] {tool_name} {tool_args}")

                # tool 실행
                tool_func=next(t for t in tools if t.name==tool_name)
                tool_result=tool_func.invoke(tool_args)

                #결과 메시지 추가
                messages.append(response)
                messages.append({
                    "role":"tool",
                    "content":tool_result,
                    "tool_call_id":tool_call["id"]
                })

                #다시 LLM 호출
                response=self.lim_with_tools.invoke(messages)

                #결괴 출력
                print("\n[응답]")
                print(response.content)

                #히스토리 저장
                self.chat_history.append({"role":"user","content":query})
                self.chat_history.append({"role":"assistant","content":response.content})


#---5.실행 ---
if __name__=="__main__":
    agent=CustomerServiceAgent()

    agent.run("C-1001","8일 전에 산 상품 환불하고 싶어요.")
    print("\n"+"="*50)

    agent.run("C-2002","배송 언제 와요?")