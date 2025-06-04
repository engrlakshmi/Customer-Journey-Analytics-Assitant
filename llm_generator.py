
import requests
import os
import json

# Constants
TOKEN_FILE = "hf_token.json"
API_URL = "xxxx"
TOKEN_REFRESH_URL = "xxx"

# Auth headers for token refresh GET request
TOKEN_REFRESH_HEADERS = {
    "x-api-key": "xx" 
}

def load_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            data = json.load(f)
            print("Existing_token", data.get("access_token"))
            return data.get("access_token")
    return None

def save_token(token):
    with open(TOKEN_FILE, "w") as f:
        json.dump({"access_token": token}, f)

def refresh_hf_token():
    try:
        response = requests.get(TOKEN_REFRESH_URL, headers=TOKEN_REFRESH_HEADERS, verify=False)
        response.raise_for_status()
        token = response.json().get("payload").get("accessToken")
        if not token:
            raise Exception("Token refresh failed: 'access_token' not found in response.")
        print("Refreshed token: ", token)
        save_token(token)
        return token 
    except Exception as e:
        raise Exception(f"Token refresh error: {str(e)}")

def analyze_journey_with_llm(journey_data, user_prompt):
    token = load_token()
    if not token:
        print("No token found. Refreshing...")
        token = refresh_hf_token()

    def send_request(current_token):
        headers = {
            "Authorization": f"{current_token}",
            "Content-Type": "application/json"
        }

        data = {
            "instruction": f"""
            {journey_data}
            """,
            "tone": {
                "mood": "neutral",
                "formality": "neutral",
                "politeness": "neutral",
                "engagement": "neutral"
            },
            "context": '''You are a fully functional AI chatbot for customer journey analytics.
You may receive prior conversation history in the format:
User: [user message]
Assistant: [bot response]
Continue the conversation based on this context. Answer the latest user message accurately and helpfully, while keeping the full conversation in mind. Only return your response.'''
        }

        return requests.post(API_URL, headers=headers, json=data, verify=False)

    try:
        response = send_request(token)
        response_json = response.json()
        print(response_json)

        # Check for token expiration/denial message
        if "Message" in response_json:
            msg = response_json["Message"]
            print(msg)
            if "not authorized" in msg.lower() and "explicit deny" in msg.lower():
                print("Token expired or unauthorized. Refreshing token...")
                token = refresh_hf_token()
                response = send_request(token)
                response_json = response.json()
                print(response_json)

        result = response_json.get("payload", {}).get("message")
        return result if result else f"No useful output. Raw response: {response_json}"

    except Exception as e:
        return f"API call failed: {str(e)}"
