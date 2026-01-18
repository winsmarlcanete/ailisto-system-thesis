from flask import Flask, render_template, jsonify#Response
# from pipelines.video_stream import generate_frames

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/live-session")
def liveSession():
    return render_template("live-session.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/profile")
def profile():
    return render_template("profile.html")

@app.route("/settings")
def settings():
    return render_template("settings.html")

@app.route("/about")
def about():
    return render_template("about.html")

# @app.route("/video_feed")
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)

    
