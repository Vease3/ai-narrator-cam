import os
import cv2
import base64
import requests
from pathlib import Path
from elevenlabs import set_api_key, generate, play

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def send_image_to_openai(image_path):
    api_key = "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"  # Replace with your OpenAI API key
    elevenlabs_api_key = "a6a62e0e6d115505ba87abd0bb0b1c2f"  # Replace with your ElevenLabs API key
    base64_image = encode_image_to_base64(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe in a short statement this photo as if you were Sir David Frederick Attenborough describing an animal in a nature documentary."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        text_response = response.json()['choices'][0]['message']['content']
        print(text_response)

        # ElevenLabs Text to Speech Conversion
        set_api_key(elevenlabs_api_key)
        audio = generate(text_response, voice="Sir Narrator")  # Change "Emma" to your desired voice
        play(audio)
    else:
        print("Error: ", response.text)

def capture_when_person_detected():
    folder_path = '/Users/veasey/Desktop/narrator-script/pics'  # Replace with your folder path
    
    # Ensure the directory exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Load the body detection model
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = body_cascade.detectMultiScale(gray, 1.1, 4)

        if len(bodies) > 0:
            image_path = os.path.join(folder_path, 'person.jpg')
            cv2.imwrite(image_path, frame)
            break

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Send the image to OpenAI and play the response
    send_image_to_openai(image_path)

capture_when_person_detected()
