/**
 * 3D Viewer — renders Anny mesh with measurement overlays using Three.js (MIT).
 *
 * Anny uses Z-up coordinate system. The backend sends vertices in cm with Z as height.
 * Three.js uses Y-up, so we swap Y↔Z when building the mesh.
 *
 * Listens for "body-result" custom event dispatched by upload.js.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// Measurement overlay colors
const COLORS = {
    chest: 0xff6b6b,
    waist: 0x4ecdc4,
    hip: 0xffe66d,
    height: 0x6c8cff,
    inseam: 0xa78bfa,
    shoulder_width: 0xf97316,
    sleeve_length: 0x22d3ee,
};

// Z-fractions matching backend measure.py (calibrated for Anny T-pose)
const HEIGHT_FRACTIONS = {
    chest: 0.68,
    waist: 0.62,
    hip: 0.48,
    crotch: 0.45,
    shoulder: 0.82,
};

let scene, camera, renderer, controls;
let meshGroup = null;
let overlayGroup = null;
let currentUnit = "cm";
let currentResult = null;

// ===== Init Three.js scene =====

function initScene() {
    const canvas = document.getElementById("three-canvas");
    const container = document.getElementById("viewer-container");

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f1117);

    camera = new THREE.PerspectiveCamera(
        45,
        container.clientWidth / container.clientHeight,
        0.1,
        2000
    );
    camera.position.set(0, 100, 300);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(50, 200, 100);
    scene.add(dirLight);

    const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
    backLight.position.set(-50, 100, -100);
    scene.add(backLight);

    // Orbit controls
    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 85, 0);
    controls.update();

    // Grid
    const grid = new THREE.GridHelper(400, 40, 0x2d3140, 0x1a1d27);
    scene.add(grid);

    window.addEventListener("resize", onResize);
    animate();
}

function onResize() {
    const container = document.getElementById("viewer-container");
    if (!container || !camera || !renderer) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls?.update();
    renderer?.render(scene, camera);
}

// ===== Build mesh =====
// Anny vertices come as [x, y, z] where Z is up.
// Three.js uses Y-up, so we swap: Three(x, y, z) = Anny(x, z, -y)

function swapYZ(v) {
    return [v[0], v[2], -v[1]];
}

function buildMesh(vertices, faces) {
    if (meshGroup) {
        scene.remove(meshGroup);
        meshGroup.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        });
    }

    meshGroup = new THREE.Group();

    const geometry = new THREE.BufferGeometry();

    // Swap Y↔Z for Three.js
    const positions = new Float32Array(vertices.length * 3);
    for (let i = 0; i < vertices.length; i++) {
        const sv = swapYZ(vertices[i]);
        positions[i * 3] = sv[0];
        positions[i * 3 + 1] = sv[1];
        positions[i * 3 + 2] = sv[2];
    }
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const indices = [];
    for (let i = 0; i < faces.length; i++) {
        indices.push(faces[i][0], faces[i][1], faces[i][2]);
    }
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    // Blue semi-transparent material
    const material = new THREE.MeshPhysicalMaterial({
        color: 0x4a90d9,
        roughness: 0.3,
        metalness: 0.1,
        transparent: true,
        opacity: 0.6,
        side: THREE.DoubleSide,
        depthWrite: false,
    });

    const mesh = new THREE.Mesh(geometry, material);
    meshGroup.add(mesh);

    // Subtle wireframe overlay for structure
    const wireMat = new THREE.MeshBasicMaterial({
        color: 0x6cb4ff,
        wireframe: true,
        transparent: true,
        opacity: 0.08,
    });
    const wireMesh = new THREE.Mesh(geometry.clone(), wireMat);
    meshGroup.add(wireMesh);

    scene.add(meshGroup);

    // Center camera on mesh
    const bbox = new THREE.Box3().setFromBufferAttribute(geometry.getAttribute("position"));
    const center = bbox.getCenter(new THREE.Vector3());
    const size = bbox.getSize(new THREE.Vector3());
    controls.target.copy(center);
    camera.position.set(center.x, center.y, center.z + size.y * 1.5);
    controls.update();
}

// ===== Measurement overlays =====

function buildOverlays(vertices, measurements) {
    if (overlayGroup) {
        scene.remove(overlayGroup);
        overlayGroup.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        });
    }

    overlayGroup = new THREE.Group();

    // Compute mesh bounds in Three.js Y-up space
    let yMin = Infinity, yMax = -Infinity;
    let xMin = Infinity, xMax = -Infinity;
    let zMin = Infinity, zMax = -Infinity;
    for (const v of vertices) {
        const sv = swapYZ(v);
        if (sv[1] < yMin) yMin = sv[1];
        if (sv[1] > yMax) yMax = sv[1];
        if (sv[0] < xMin) xMin = sv[0];
        if (sv[0] > xMax) xMax = sv[0];
        if (sv[2] < zMin) zMin = sv[2];
        if (sv[2] > zMax) zMax = sv[2];
    }
    const height = yMax - yMin;
    const xCenter = (xMin + xMax) / 2;
    const zCenter = (zMin + zMax) / 2;

    // Circumference rings (chest, waist, hip) — horizontal in Y-up space
    for (const name of ["chest", "waist", "hip"]) {
        const frac = HEIGHT_FRACTIONS[name];
        const yLevel = yMin + frac * height;
        const color = COLORS[name] || 0xffffff;

        // Find vertices near this Y level to determine radius
        const nearby = [];
        for (const v of vertices) {
            const sv = swapYZ(v);
            if (Math.abs(sv[1] - yLevel) < height * 0.02) {
                nearby.push(sv);
            }
        }
        if (nearby.length < 3) continue;

        let maxR = 0;
        for (const sv of nearby) {
            const dx = sv[0] - xCenter;
            const dz = sv[2] - zCenter;
            const r = Math.sqrt(dx * dx + dz * dz);
            if (r > maxR) maxR = r;
        }

        const ring = createRing(xCenter, yLevel, zCenter, maxR, color);
        overlayGroup.add(ring);

        const value = measurements[name];
        if (value) {
            const text = formatValue(value);
            const label = createLabel(text, xCenter + maxR + 3, yLevel, zCenter, color);
            overlayGroup.add(label);
        }
    }

    // Height line (vertical)
    if (measurements.height) {
        const lineColor = COLORS.height;
        const x = xMax + 5;
        const line = createLine(
            new THREE.Vector3(x, yMin, zCenter),
            new THREE.Vector3(x, yMax, zCenter),
            lineColor
        );
        overlayGroup.add(line);

        const label = createLabel(
            formatValue(measurements.height),
            x + 3, (yMin + yMax) / 2, zCenter,
            lineColor
        );
        overlayGroup.add(label);
    }

    // Inseam line
    if (measurements.inseam) {
        const lineColor = COLORS.inseam;
        const crotchY = yMin + HEIGHT_FRACTIONS.crotch * height;
        const line = createLine(
            new THREE.Vector3(xCenter, yMin, zCenter + 3),
            new THREE.Vector3(xCenter, crotchY, zCenter + 3),
            lineColor
        );
        overlayGroup.add(line);

        const label = createLabel(
            formatValue(measurements.inseam),
            xCenter + 3, (yMin + crotchY) / 2, zCenter + 3,
            lineColor
        );
        overlayGroup.add(label);
    }

    // Shoulder width line
    if (measurements.shoulder_width) {
        const lineColor = COLORS.shoulder_width;
        const shoulderY = yMin + HEIGHT_FRACTIONS.shoulder * height;
        const halfW = measurements.shoulder_width.value_cm / 2;
        const line = createLine(
            new THREE.Vector3(xCenter - halfW, shoulderY, zCenter),
            new THREE.Vector3(xCenter + halfW, shoulderY, zCenter),
            lineColor
        );
        overlayGroup.add(line);

        const label = createLabel(
            formatValue(measurements.shoulder_width),
            xCenter, shoulderY + 3, zCenter,
            lineColor
        );
        overlayGroup.add(label);
    }

    scene.add(overlayGroup);
}

function createRing(cx, y, cz, radius, color) {
    const segments = 64;
    const points = [];
    for (let i = 0; i <= segments; i++) {
        const angle = (i / segments) * Math.PI * 2;
        points.push(
            new THREE.Vector3(
                cx + Math.cos(angle) * radius,
                y,
                cz + Math.sin(angle) * radius
            )
        );
    }
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color, linewidth: 2 });
    return new THREE.Line(geometry, material);
}

function createLine(start, end, color) {
    const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
    const material = new THREE.LineBasicMaterial({ color, linewidth: 2 });
    return new THREE.Line(geometry, material);
}

function createLabel(text, x, y, z, color) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = 256;
    canvas.height = 64;

    ctx.fillStyle = "rgba(15, 17, 23, 0.8)";
    ctx.roundRect(0, 0, canvas.width, canvas.height, 8);
    ctx.fill();

    ctx.fillStyle = "#" + color.toString(16).padStart(6, "0");
    ctx.font = "bold 28px -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(material);
    sprite.position.set(x, y, z);
    sprite.scale.set(12, 3, 1);
    return sprite;
}

function formatValue(m) {
    if (!m) return "--";
    if (currentUnit === "in") return m.value_in.toFixed(1) + '"';
    return m.value_cm.toFixed(1) + " cm";
}

// ===== Sidebar =====

function populateSidebar(measurements, calibration) {
    const list = document.getElementById("measurement-list");
    list.innerHTML = "";

    const labelMap = {
        height: "Height",
        chest: "Chest",
        waist: "Waist",
        hip: "Hip",
        inseam: "Inseam",
        shoulder_width: "Shoulder Width",
        sleeve_length: "Sleeve Length",
    };

    for (const [key, label] of Object.entries(labelMap)) {
        const m = measurements[key];
        if (!m) continue;

        const item = document.createElement("div");
        item.className = "measurement-item";
        item.innerHTML = `
            <span class="measurement-label">${label}</span>
            <span class="measurement-value" data-cm="${m.value_cm}" data-in="${m.value_in}">
                ${formatValue(m)}
            </span>
        `;
        list.appendChild(item);
    }

    // Calibration badge
    const badge = document.getElementById("calibration-badge");
    const badgeText = document.getElementById("calibration-text");
    const conf = calibration.confidence;
    badge.className = "calibration-badge";
    if (conf >= 0.8) badge.classList.add("high");
    else if (conf >= 0.5) badge.classList.add("medium");
    else badge.classList.add("low");
    badgeText.textContent = `Calibration: ${(conf * 100).toFixed(0)}% (${calibration.method_used})`;

    const warningEl = document.getElementById("calibration-warning");
    if (calibration.warning) {
        warningEl.textContent = calibration.warning;
        warningEl.classList.remove("hidden");
    } else {
        warningEl.classList.add("hidden");
    }
}

function updateUnits() {
    document.querySelectorAll(".measurement-value").forEach((el) => {
        const cm = parseFloat(el.dataset.cm);
        const inch = parseFloat(el.dataset.in);
        if (currentUnit === "in") {
            el.textContent = inch.toFixed(1) + '"';
        } else {
            el.textContent = cm.toFixed(1) + " cm";
        }
    });

    if (currentResult) {
        buildOverlays(currentResult.mesh_vertices, currentResult.measurements);
    }
}

// Unit toggle in sidebar
document.getElementById("unit-cm")?.addEventListener("click", () => {
    currentUnit = "cm";
    document.getElementById("unit-cm").classList.add("active");
    document.getElementById("unit-in").classList.remove("active");
    updateUnits();
});
document.getElementById("unit-in")?.addEventListener("click", () => {
    currentUnit = "in";
    document.getElementById("unit-in").classList.add("active");
    document.getElementById("unit-cm").classList.remove("active");
    updateUnits();
});

// ===== Listen for result =====

window.addEventListener("body-result", (e) => {
    const result = e.detail;
    currentResult = result;

    if (!scene) initScene();

    buildMesh(result.mesh_vertices, result.faces);
    buildOverlays(result.mesh_vertices, result.measurements);
    populateSidebar(result.measurements, result.calibration);

    setTimeout(onResize, 100);
});
