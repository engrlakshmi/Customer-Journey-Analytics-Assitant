from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from journey_mixer import process_journey_file  # ML logic here
from llm_generator import analyze_journey_with_llm  # AI logic here
import os
from datetime import datetime
from flask_session import Session
from apply_event_model import predict_and_add_events

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Step 1: Add Predicted Events using trained model (or train if missing)
        try:
            predicted_df = predict_and_add_events(filepath)
            # Optional: save file with predictions (if you want to persist it)
            predicted_file_path = filepath.replace(".csv", "_with_predicted_events.csv")
            predicted_df.to_csv(predicted_file_path, index=False)
        except Exception as e:
            return jsonify({'success': False, 'message': f'Event prediction failed: {str(e)}'})

        # Step 2: Run ML analysis
        ml_output = process_journey_file(predicted_file_path)

        # Store in session
        session['ml_output'] = ml_output
        session['chat_history'] = []
        session['first_prompt_sent'] = False  # flag to track if system prompt was sent

        return jsonify({'success': True, 'message': 'File processed successfully.'})
    return jsonify({'success': False, 'message': 'No file uploaded.'})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    ml_output = session.get('ml_output', '')
    chat_history = session.get('chat_history', [])
    first_prompt_sent = session.get('first_prompt_sent', False)

    messages = []

    # Only for the first prompt
    if not first_prompt_sent:
        combined_user_message = f"""
        User: {user_message}
I'm building an AI-powered solution to optimize customer payment journeys by analyzing behavioral clusters, identifying key milestones, detecting drop-offs or loops, and recommending visual flows.

Please assist with the following tasks based on the machine learning derived insights.

ML Output Summary
Analyze the following ML output, which includes insights from clustering and behavioral journey data. Key sections are:

Number of Customers and Cluster Distribution: How users are grouped across behavioral clusters with proportions.

Top Transitions per Cluster: Frequent user transitions in each cluster, indicating typical behavior patterns.

Success Count per Cluster: Number of users who reached the goal (e.g., payment) in each cluster.

Top Paths Without Goal: Common paths for users who didn't convert — useful for spotting churn or confusion.

Top Paths to Payment Confirmed: Most frequent successful conversion paths.

Top Paths to Payment Submitted: Common intermediate steps for users who initiated payment but didn't complete it.

Last Event Summary: Highlights where users dropped off or ended their journeys.

The ML insights are as follows:
{ml_output}

✅ Tasks
Based on the event transition data and clustering results, give me a structured analysis with the following sections:

1. 📍 Critical Milestones per Cluster

For each cluster, list the key event transitions that indicate forward movement toward conversion.

Format transitions using arrows (→), e.g.:
Event A → Event B → Event C

2. 🔁 Drop-Offs and Loops

Identify the top drop-off points (events where users exit).

Mention any frequent loop transitions (repetitive patterns like A → A or A → B → A).

3. 📈 Suggested Events for Sankey Diagram

Recommend a filtered list of high-impact events to include in a Sankey diagram for journey visualization.

Prioritize conversion-relevant and decision-point events. Ignore technical or redundant steps if they don't affect decision making.

4. 📌 Representative Journey Flows (2-3)

Provide 2-3 generalized user journeys that reflect common behaviors across clusters, excluding outliers or noise.

Format using arrows (→), e.g.:
Start → Login → View Plans → Submit Payment → Payment Confirmed

If certain events (like Feedback Declined) appear as the start but originate from previous sessions or unrelated actions, ignore or skip them.

Ensure the journeys are representative and could explain the behavior of most users (focus on high-frequency, high-conversion paths).

5. 💡 Recommendations

Suggest 2-3 actionable improvements to the journey, based on drop-off points, loops, or missing milestones.

Example: "Introduce a reminder email after event X", or "Simplify steps between Y and Z".


"""

        messages.append({
            "role": "user",
            "content": combined_user_message
        })
        session['first_prompt_sent'] = True

    else:
        # Add full chat history for later turns
        for msg in chat_history:
            messages.append({"role": "user", "content": msg['user']})
            messages.append({"role": "assistant", "content": msg['bot']})

        # Append the current message
        messages.append({"role": "user", "content": user_message})

    # Debug
    print("Messages sent to LLM:", messages)

    # Call your LLM
    bot_response = analyze_journey_with_llm(messages, '')

    # Save to session
    chat_entry = {
        'user': user_message,
        'bot': bot_response,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    chat_history.append(chat_entry)
    session['chat_history'] = chat_history

    return jsonify({'response': bot_response, 'timestamp': chat_entry['timestamp']})

if __name__ == '__main__':
    app.run(debug=True)
