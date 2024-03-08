import replicate
import boto3
from flask import Flask, request, jsonify, render_template
import tempfile

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

bucket_name = 'BUCKET NAME'
aws_access_key = 'KEY'
aws_secret_key = 'SECRET'

s3 = boto3.client(
    "s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
)

@app.route('/process-audio', methods=["POST"])
def process_audio_data():
  audio_data = request.files["audio"].read()

  with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
    temp_audio.write(audio_data)
    temp_audio.flush()
    s3.upload_file(temp_audio.name, bucket_name, temp_audio.name)
    temp_audio_uri = f"https://{bucket_name}.s3.amazonaws.com/{temp_audio.name}"


    output = replicate.run(
        "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
        input={
            "task": "transcribe",
            "audio": temp_audio_uri,
            "language": "None",
            "timestamp": "chunk",
            "batch_size": 64,
            "diarise_audio": False
        }
    )
    print(output)

    return jsonify({"transcript": output['text']})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)