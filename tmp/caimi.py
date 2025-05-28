import random
import openai

# 角色列表
characters = [
    "爱莉希雅",
    "雷电芽衣",
    "布洛妮娅",
    "琪亚娜",
    "符华",
    "德丽莎",
    "姬子",
    "希儿",
    "罗莎莉亚",
    "莉莉娅"
]

# 随机选择一个角色
secret_character = random.choice(characters)

# 设置OpenAI API密钥
openai.api_key = "你的API密钥"

def get_ai_response(user_input):
    prompt = f"""
    你作为人物猜谜游戏主持人，谜底是{secret_character}。
    请严格遵循:
    1.当用户输入与谜底完全一致时(标点符号不影响)，立即回复『推理正确!游戏结束』
    2.用户输入包含"提示"时，给出1个特征(如:她与「真我』概念相关)
    3.其他情况:
    完全匹配特征 →「是』
    部分相关 →『可以这么说』
    无关内容 →『不是』
    特别注意:
    必须严格匹配文字(如『爱莉希雅』≠『爱丽希雅』)
    用户直接输入谜底时立即终止游戏
    不要主动补充说明
    当前用户输入: {user_input}
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip()

def main():
    print("欢迎来到人物猜谜游戏！")
    print("我已经想好了一个角色，你可以通过提问来猜测TA是谁。")
    print("输入'提示'可以获取一个特征提示。")
    print("----------------------------------")
    
    while True:
        user_input = input("你的问题或猜测: ").strip()
        
        if not user_input:
            print("请输入有效内容")
            continue
            
        response = get_ai_response(user_input)
        print(response)
        
        if "推理正确!游戏结束" in response:
            break

if __name__ == "__main__":
    main()