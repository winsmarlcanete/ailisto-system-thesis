document.addEventListener('DOMContentLoaded', function() {
    
    function startSession() {
        
    }

    function pauseSession() {
        
    }

    function endSession() {
        
    }

    function saveSession() {
        
    }

    
    document.getElementById('start-btn').addEventListener('click', startSession);
    document.getElementById('pause-btn').addEventListener('click', pauseSession);
    document.getElementById('end-btn').addEventListener('click', endSession);
    document.getElementById('save-btn').addEventListener('click', saveSession);

    
    document.addEventListener('cameraReady', () => {
        document.getElementById('start-btn').disabled = false;
        document.getElementById('pause-btn').disabled = false;
        document.getElementById('end-btn').disabled = false;
        document.getElementById('save-btn').disabled = false;
    });
});