const startBtn = document.getElementById("start-btn");
const pauseBtn = document.getElementById("pause-btn");
const endBtn = document.getElementById("end-btn");
const saveBtn = document.getElementById("save-btn");

const annotatedVideo = document.getElementById("annotated-video-recorded");

let isPaused = false;

/* ---------- START ---------- */
startBtn.addEventListener("click", () => {
    annotatedVideo.src = "/video_feed";
    annotatedVideo.style.display = "block";

    startBtn.disabled = true;
    pauseBtn.disabled = false;
    endBtn.disabled = false;
    saveBtn.disabled = false;
});

/* ---------- PAUSE ---------- */
pauseBtn.addEventListener("click", () => {
    if (!isPaused) {
        fetch("/pause_stream");
        pauseBtn.innerText = "Resume";
    } else {
        fetch("/resume_stream");
        pauseBtn.innerText = "Pause";
    }
    isPaused = !isPaused;
});

/* ---------- END ---------- */
endBtn.addEventListener("click", () => {
    fetch("/stop_stream");

    annotatedVideo.src = "";
    annotatedVideo.style.display = "none";

    startBtn.disabled = false;
    pauseBtn.disabled = true;
    endBtn.disabled = true;
    saveBtn.disabled = true;

    pauseBtn.innerText = "Pause";
    isPaused = false;
});

/* ---------- SAVE ---------- */
saveBtn.addEventListener("click", () => {
    alert("Session video saved successfully.");
});
