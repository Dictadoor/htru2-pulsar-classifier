import * as THREE from './vendor/three.module.js';

const container = document.querySelector('#universe');
const status = document.querySelector('#scene-status');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
} catch {
  status.textContent = 'Este navegador no pudo iniciar la escena 3D. Puedes acceder al clasificador con el botón de la portada.';
}

if (renderer) initializeScene();

function initializeScene() {
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
  renderer.setClearColor(0x05080e);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.append(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
  camera.position.set(0, 0, 9.5);
  const clock = new THREE.Clock();
  const target = new THREE.Vector2();
  const orientation = new THREE.Vector2();
  const group = new THREE.Group();
  const magnet = new THREE.Group();
  scene.add(group);
  group.add(magnet);
  magnet.rotation.z = -0.43;

  let paused = reducedMotion.matches;
  let time = 0;
  let dragging = false;
  let lastPointer = null;
  let spinOffset = 0;
  let verticalOffset = 0;
  let frameCount = 0;
  let running = false;
  let frameId = 0;

  // Generador determinista para conservar la composición del campo estelar.
  let seed = 42;
  function random() {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  }

  const positions = [];
  const colors = [];
  const sizes = [];
  for (let i = 0; i < 2600; i++) {
    positions.push((random() - .5) * 54, (random() - .5) * 36, -4 - random() * 45);
    const brightness = .18 + random() * .67;
    colors.push(brightness * .76, brightness * .85, brightness);
    sizes.push(.5 + random() * 1.8);
  }
  const starGeometry = new THREE.BufferGeometry();
  starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  starGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  starGeometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));
  const stars = new THREE.Points(starGeometry, new THREE.ShaderMaterial({
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    uniforms: { uPixelRatio: { value: renderer.getPixelRatio() } },
    vertexShader: `
      attribute float size;
      varying vec3 vColor;
      uniform float uPixelRatio;
      void main() {
        vColor = color;
        vec4 view = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * uPixelRatio * (16.0 / -view.z);
        gl_Position = projectionMatrix * view;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      void main() {
        float distanceToCenter = length(gl_PointCoord - .5);
        float opacity = 1.0 - smoothstep(.1, .5, distanceToCenter);
        gl_FragColor = vec4(vColor, opacity);
      }
    `
  }));
  scene.add(stars);

  // La superficie procedural y la corona se calculan en la GPU.
  const surfaceMaterial = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 } },
    vertexShader: `
      varying vec3 vPosition;
      varying vec3 vNormal;
      varying vec3 vWorld;
      void main() {
        vPosition = position;
        vNormal = normalize(mat3(modelMatrix) * normal);
        vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float uTime;
      varying vec3 vPosition;
      varying vec3 vNormal;
      varying vec3 vWorld;
      float hash(vec3 p) {
        p = fract(p * .3183099 + vec3(.1, .2, .3));
        p *= 17.0;
        return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
      }
      float noise(vec3 p) {
        vec3 cell = floor(p);
        vec3 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        return mix(mix(mix(hash(cell), hash(cell + vec3(1,0,0)), f.x),
                       mix(hash(cell + vec3(0,1,0)), hash(cell + vec3(1,1,0)), f.x), f.y),
                   mix(mix(hash(cell + vec3(0,0,1)), hash(cell + vec3(1,0,1)), f.x),
                       mix(hash(cell + vec3(0,1,1)), hash(cell + vec3(1,1,1)), f.x), f.y), f.z);
      }
      void main() {
        vec3 p = vPosition * 7.0 + vec3(0.0, uTime * .09, 0.0);
        float turbulence = noise(p) * .55 + noise(p * 2.3) * .3 + noise(p * 5.1) * .15;
        float filaments = pow(abs(sin(turbulence * 14.0 + vPosition.y * 10.0)), 8.0);
        vec3 color = mix(vec3(.035,.19,.44), vec3(.27,.67,1.0), turbulence);
        color += filaments * vec3(.27,.48,.65);
        vec3 viewDirection = normalize(cameraPosition - vWorld);
        float face = max(dot(normalize(vNormal), viewDirection), 0.0);
        float light = max(dot(normalize(vNormal), normalize(vec3(-.5,.7,1.0))), 0.0);
        color *= .52 + .63 * light;
        color += pow(1.0 - face, 3.0) * vec3(.25,.65,1.0);
        gl_FragColor = vec4(color, 1.0);
      }
    `
  });
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(.73, 80, 64), surfaceMaterial);
  magnet.add(sphere);

  const atmosphere = new THREE.Mesh(new THREE.SphereGeometry(.765, 64, 48), new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vWorld;
      void main() {
        vNormal = normalize(mat3(modelMatrix) * normal);
        vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec3 vNormal;
      varying vec3 vWorld;
      void main() {
        float rim = pow(1.0 - max(dot(normalize(vNormal), normalize(cameraPosition - vWorld)), 0.0), 2.2);
        gl_FragColor = vec4(.12,.49,1.0,rim * .72);
      }
    `
  }));
  magnet.add(atmosphere);

  const glowCanvas = document.createElement('canvas');
  glowCanvas.width = glowCanvas.height = 128;
  const context = glowCanvas.getContext('2d');
  const glowGradient = context.createRadialGradient(64, 64, 0, 64, 64, 64);
  glowGradient.addColorStop(0, 'rgba(45,135,255,0.5)');
  glowGradient.addColorStop(.3, 'rgba(30,95,210,0.17)');
  glowGradient.addColorStop(1, 'rgba(0,20,80,0)');
  context.fillStyle = glowGradient;
  context.fillRect(0, 0, 128, 128);
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(glowCanvas), blending: THREE.AdditiveBlending, transparent: true, depthWrite: false
  }));
  halo.scale.set(6.5, 6.5, 1);
  group.add(halo);

  const beams = [];
  for (const direction of [-1, 1]) {
    const geometry = new THREE.CylinderGeometry(.025, .55, 4.5, 48, 24, true);
    const material = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
      uniforms: { uTime: { value: 0 } },
      vertexShader: `varying vec2 vUv; void main() {vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
      fragmentShader: `
        varying vec2 vUv;
        uniform float uTime;
        void main() {
          float distanceFromPole = 1.0 - vUv.y;
          float fade = pow(1.0 - distanceFromPole, 1.7) * smoothstep(0.0, .07, distanceFromPole);
          float rays = .57 + .2 * sin(vUv.x * 62.83) + .1 * sin(vUv.x * 138.2);
          float pulse = .84 + .16 * sin(distanceFromPole * 20.0 - uTime * 3.0);
          vec3 color = mix(vec3(.10,.37,.85), vec3(.53,.82,1.0), pow(1.0 - distanceFromPole, 4.0));
          gl_FragColor = vec4(color, fade * rays * pulse * .3);
        }
      `
    });
    const beam = new THREE.Mesh(geometry, material);
    // El extremo estrecho nace en cada polo magnético.
    beam.rotation.z = direction === 1 ? Math.PI : 0;
    beam.position.y = direction * 2.94;
    magnet.add(beam);
    beams.push(beam);
  }

  const fieldLines = new THREE.Group();
  const particles = [];
  const particlePositions = [];
  for (let ring = 0; ring < 3; ring++) {
    const radius = 1.9 + ring * .6;
    const polar = .52 + ring * .12;
    for (let meridian = 0; meridian < 8; meridian++) {
      const azimuth = meridian / 8 * Math.PI * 2;
      const points = [];
      for (let step = 0; step <= 96; step++) {
        const theta = polar + (Math.PI - 2 * polar) * step / 96;
        const r = radius * Math.sin(theta) ** 2;
        points.push(new THREE.Vector3(r * Math.sin(theta) * Math.cos(azimuth), r * Math.cos(theta), r * Math.sin(theta) * Math.sin(azimuth)));
      }
      const curve = new THREE.CatmullRomCurve3(points);
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({
        color: 0x448fd3, transparent: true, opacity: .15 - ring * .028, depthWrite: false
      }));
      fieldLines.add(line);
      if (meridian % 2 === 0) {
        particles.push({ curve, phase: random() });
        particlePositions.push(0, 0, 0);
      }
    }
  }
  magnet.add(fieldLines);
  const particleGeometry = new THREE.BufferGeometry();
  particleGeometry.setAttribute('position', new THREE.Float32BufferAttribute(particlePositions, 3));
  const fieldParticles = new THREE.Points(particleGeometry, new THREE.PointsMaterial({
    color: 0x9ed8ff, size: .035, transparent: true, opacity: .85, blending: THREE.AdditiveBlending, depthWrite: false
  }));
  magnet.add(fieldParticles);

  function resize() {
    const width = innerWidth;
    const height = innerHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    const mobile = width <= 600;
    group.position.set(mobile ? .08 : 1.8, mobile ? 1.3 : 0, 0);
    const scale = mobile ? .72 : Math.min(1.16, width / 1440);
    group.scale.setScalar(scale);
  }

  document.addEventListener('keydown', event => {
    if (event.code === 'Space' && event.target === document.body) {
      event.preventDefault();
      paused = !paused;
    }
  });

  container.addEventListener('pointerdown', event => {
    dragging = true;
    lastPointer = { x: event.clientX, y: event.clientY };
    container.setPointerCapture(event.pointerId);
  });
  container.addEventListener('pointermove', event => {
    target.set((event.clientX / innerWidth - .5) * 2, (event.clientY / innerHeight - .5) * 2);
    if (dragging && lastPointer) {
      spinOffset += (event.clientX - lastPointer.x) * .006;
      verticalOffset = THREE.MathUtils.clamp(verticalOffset + (event.clientY - lastPointer.y) * .004, -.7, .7);
      lastPointer = { x: event.clientX, y: event.clientY };
    }
  });
  function endDrag() { dragging = false; lastPointer = null; }
  container.addEventListener('pointerup', endDrag);
  container.addEventListener('pointercancel', endDrag);
  container.addEventListener('lostpointercapture', endDrag);
  container.addEventListener('pointerleave', () => { if (!dragging) target.set(0, 0); });
  reducedMotion.addEventListener('change', event => {
    paused = event.matches;
  });
  window.addEventListener('resize', resize);

  function render() {
    if (!running) return;
    const delta = Math.min(clock.getDelta(), .05);
    if (!paused) time += delta;
    orientation.lerp(target, .035);
    group.rotation.y = spinOffset + orientation.x * .13;
    group.rotation.x = verticalOffset + orientation.y * .08;
    magnet.rotation.y = time * .13;
    magnet.rotation.z = -.43 + Math.sin(time * .17) * .12;
    sphere.rotation.y = time * .1;
    surfaceMaterial.uniforms.uTime.value = time;
    beams.forEach(beam => { beam.material.uniforms.uTime.value = time; });
    halo.material.opacity = .8 + Math.sin(time * 1.7) * .08;
    stars.rotation.z = time * .0012;
    stars.position.x = orientation.x * -.15;
    stars.position.y = orientation.y * .1;
    const points = particleGeometry.attributes.position;
    particles.forEach((particle, index) => {
      const point = particle.curve.getPointAt((particle.phase + time * .045) % 1);
      points.setXYZ(index, point.x, point.y, point.z);
    });
    points.needsUpdate = true;
    renderer.render(scene, camera);
    frameCount++;
    if (frameCount === 2) {
      status.hidden = true;
      document.body.dataset.scene = 'ready';
    }
    frameId = requestAnimationFrame(render);
  }

  function start() {
    if (running) return;
    running = true;
    clock.start();
    render();
  }
  document.addEventListener('visibilitychange', () => {
    running = !document.hidden;
    cancelAnimationFrame(frameId);
    if (running) { clock.start(); render(); }
  });
  renderer.domElement.addEventListener('webglcontextlost', event => {
    event.preventDefault();
    running = false;
    cancelAnimationFrame(frameId);
    status.hidden = false;
    status.textContent = 'La escena 3D se interrumpió. Recarga la página para recuperarla.';
  });
  resize();
  start();
}
