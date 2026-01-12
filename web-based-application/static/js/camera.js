document.addEventListener('DOMContentLoaded', function() {
    const selectBtn = document.getElementById('select-camera-btn');
    const cameraSelect = document.getElementById('user-camera-select');
    const videoElement = document.getElementById('user-video');

    async function getUserMedia(constraints = { video: true }) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            videoElement.srcObject = stream;
            return stream;
        } catch (error) {
            console.error('Error accessing camera: ', error);
            throw error;
        }
    }

    
    async function populateCameraOptions() {
        const devices = await navigator.mediaDevices.enumerateDevices();
        cameraSelect.innerHTML = '';
        devices.forEach(device => {
            if (device.kind === 'videoinput') {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Camera ${cameraSelect.length + 1}`;
                cameraSelect.appendChild(option);
            }
        });
    }

    
    cameraSelect.addEventListener('change', async () => {
        const selectedCameraId = cameraSelect.value;
        await getUserMedia({ video: { deviceId: selectedCameraId } });
    });

    selectBtn.addEventListener('click', async () => {
        try {
            await getUserMedia();
            selectBtn.style.display = 'none';
            cameraSelect.style.display = 'block';
            document.querySelector('.video-container').style.display = 'block';
            await populateCameraOptions();
            document.dispatchEvent(new CustomEvent('cameraReady'));
        } catch (error) {
            // Handle error
        }
    });
});