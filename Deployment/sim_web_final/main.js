console.log("Dashboard loaded - waiting for Play button...");

let socket = null;
let autoModeActive = false;
let steeringWheelMesh = null;

// ======== Three.js Setup (3D Steering Wheel) ========
const container = document.getElementById('steering-wheel-container');
if (container) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
        75,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
    );
    camera.position.z = 30;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 5);
    directionalLight.position.set(0, 50, 350);
    scene.add(directionalLight);

    // Load 3D steering wheel model
    new THREE.GLTFLoader().load('assets/models/steering_wheel.glb', (gltf) => {
        steeringWheelMesh = gltf.scene;
        steeringWheelMesh.scale.set(4, 4, 4);
        steeringWheelMesh.position.set(-20.5, -10.5, 0);
        steeringWheelMesh.rotation.y = Math.PI * 1.5;
        scene.add(steeringWheelMesh);
        console.log("Steering wheel model loaded successfully");
    });

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }
    animate();
}

// Smoothly rotate the 3D steering wheel based on steering angle
function updateSteeringWheel(angle) {
    if (!steeringWheelMesh) return;

    const maxRotationDegrees = 35;
    const targetDegrees = -angle * maxRotationDegrees;

    const currentDegrees = steeringWheelMesh.rotation.x * (180 / Math.PI);
    const smoothedDegrees = currentDegrees + (targetDegrees - currentDegrees) * 0.25;

    steeringWheelMesh.rotation.x = smoothedDegrees * (Math.PI / 180);
}

// ======== Socket.IO Connection ========
function connectToServer() {
    if (socket && socket.connected) return;

    socket = io('http://127.0.0.1:4567', {
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionAttempts: 10,
        reconnectionDelay: 1000,
        timeout: 20000
    });

    socket.on('connect', () => {
        console.log("Successfully connected to server");
    });

    socket.on('web_telemetry', (data) => {
        if (!autoModeActive) return;

        // Update road camera view
        document.getElementById('road-frame').src = 
            "data:image/jpeg;base64," + data.image_b64;

        // Update steering angle display
        const angle = parseFloat(data.steering).toFixed(3);
        document.getElementById('angle-display').textContent = angle;

        // Optional: show speed and throttle
        if (data.speed !== undefined) {
            document.getElementById('speed-display').textContent = 
                parseFloat(data.speed).toFixed(1);
        }
        if (data.throttle !== undefined) {
            document.getElementById('throttle-display').textContent = 
                parseFloat(data.throttle).toFixed(3);
        }

        // Rotate 3D steering wheel
        updateSteeringWheel(parseFloat(data.steering));
    });

    socket.on('connect_error', (err) => {
        console.error("Connection failed:", err.message);
    });

    socket.on('disconnect', () => {
        console.log("Disconnected from server");
    });
}

// ======== Play / Stop Button ========
document.getElementById('start-button').addEventListener('click', () => {
    autoModeActive = !autoModeActive;
    const btn = document.getElementById('start-button');

    if (autoModeActive) {
        connectToServer();
        btn.textContent = "Stop Auto Mode";
        btn.style.background = "linear-gradient(145deg, #ff4444, #cc0000)";
        btn.style.boxShadow = "0 8px 15px rgba(255,0,0,0.4)";
    } else {
        if (socket) socket.disconnect();
        btn.textContent = "Start Auto Mode";
        btn.style.background = "linear-gradient(145deg, #00ffaa, #00b377)";
        btn.style.boxShadow = "0 8px 15px rgba(0,255,170,0.4)";

        // Reset displays
        document.getElementById('road-frame').src = "assets/images/placeholder.jpg";
        document.getElementById('angle-display').textContent = "0.000";
    }
});