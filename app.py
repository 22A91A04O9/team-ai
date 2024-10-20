from flask import Flask, render_template, Response
import cv2
import base64
import requests
import json

app = Flask(__name__)

# API endpoint (replace with the actual endpoint of NVIDIA NeMo model)
api_url = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Function to send a frame and get the detection result
def send_frame_to_model(frame):
    # Encode frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    
    # Convert to base64 string
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    # API Key (replace with your actual API key)
    api_key = "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"  # Replace with your actual API key

    # Create JSON payload
    json_payload = {
        "model": 'meta/llama-3.2-90b-vision-instruct',
        "messages": [
            {
                "role": "user",
                "content": f'What is in this image? <img src="data:image/png;base64,{img_base64}" />'
            }
        ],
        "max_tokens": 512,
        "temperature": 1.00,
        "top_p": 1.00,
        "stream": False
    }

    # Send POST request to the API
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        response = requests.post(api_url, json=json_payload, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Response: {response.status_code} - {response.json()}")
            return {"error": f"Failed to get response, status code: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# Function to generate video frames
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize the frame for processing
        frame = cv2.resize(frame, (640, 480))

        # Send the frame to the model and get the detection result
        result = send_frame_to_model(frame)

        # Process and display detection results
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "No detection")

            # Mark detected potholes (example logic, adjust according to your model's output)
            if "pothole" in text.lower():
                # Draw a rectangle around detected pothole (example coordinates)
                # Replace with actual coordinates if provided by the model
                cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)  # Adjust coordinates as needed
                cv2.putText(frame, "Pothole Detected!", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as a response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
