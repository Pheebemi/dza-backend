import os
from google import genai
from google.genai import types


def get_gemini_response(system_prompt: str, conversation_history: list, user_message: str) -> str:
    client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))

    # Convert history to new SDK format
    formatted_history = []
    for msg in conversation_history:
        role = msg.get('role', 'user')
        if role == 'assistant':
            role = 'model'
        formatted_history.append(
            types.Content(role=role, parts=[types.Part(text=msg.get('content', ''))])
        )

    chat = client.chats.create(
        model='gemini-2.5-flash',
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        ),
        history=formatted_history,
    )

    response = chat.send_message(user_message)
    return response.text
