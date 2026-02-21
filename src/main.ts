import './style.css';
import cv from "@techstark/opencv-js";

/**
 * =====================================================
 *  LAPLACIAN PYRAMID BLENDING — Explicație pas cu pas
 * =====================================================
 *
 * Scopul: lipim o imagine mică (patch) peste una mare (fundal)
 * așa încât marginile să se îmbine smooth, fără cusătură vizibilă.
 *
 * Pași:
 *  1. Citim imaginea principală (fundal) și patch-ul uploadat.
 *  2. Plasăm patch-ul în centrul fundalului.
 *  3. Creăm o MASCĂ: alb unde e patch-ul, negru în rest.
 *  4. Construim piramida Gaussiană (blur progresiv) pentru:
 *     - fundal, patch (extins la dimensiunea fundalului), și mască.
 *  5. Din piramida Gaussiană extragem piramida Laplaciană
 *     (= detaliile/edge-urile la fiecare nivel de rezoluție).
 *  6. La FIECARE nivel, combinăm detaliile patch-ului cu cele
 *     ale fundalului, ghidați de masca blur-uită la acel nivel.
 *     → La nivelele mici (rezoluție mică), masca e foarte blur-uită,
 *       deci tranziția e graduală = FĂRĂ cusătură vizibilă.
 *  7. Reconstruim imaginea finală din piramida Laplaciană combinată.
 *
 * Rezultat: patch-ul apare natural pe fundal, cu margini smooth.
 * =====================================================
 */

const LEVELS = 6;

let cv2: any;

// ─── Piramida Gaussiană ───────────────────────────────
// Fiecare nivel e blur + resize la jumătate.
// Nivel 0 = imaginea originală, nivel N = cea mai blur-uită.
function buildGaussianPyramid(src: any, levels: number): any[] {
  const pyramid: any[] = [src.clone()];
  let current = pyramid[0];

  for (let i = 1; i < levels; i++) {
    const down = new cv2.Mat();
    cv2.pyrDown(current, down);
    pyramid.push(down);
    current = down;
  }
  return pyramid;
}

// ─── Piramida Laplaciană ──────────────────────────────
// Laplacian[i] = Gauss[i] - pyrUp(Gauss[i+1])
// = detaliile pierdute între nivel i și i+1
// Ultimul nivel = Gauss[N] (cel mai blur).
function buildLaplacianPyramid(gauss: any[]): any[] {
  const lap: any[] = [];

  for (let i = 0; i < gauss.length - 1; i++) {
    const up = new cv2.Mat();
    cv2.pyrUp(gauss[i + 1], up, new cv2.Size(gauss[i].cols, gauss[i].rows));
    const diff = new cv2.Mat();
    cv2.subtract(gauss[i], up, diff);
    lap.push(diff);
    up.delete();
  }
  // Ultimul nivel = imaginea cea mai blur-uită (baza piramidei)
  lap.push(gauss[gauss.length - 1].clone());
  return lap;
}

// ─── Blend la un singur nivel ─────────────────────────
// result = patch * mask + fundal * (1 - mask)
// La nivele blur-uite, mask e smooth → margini smooth
function blendWithMask(patch: any, background: any, mask: any): any {
  const mask3 = new cv2.Mat();
  const channels = new cv2.MatVector();
  channels.push_back(mask);
  channels.push_back(mask);
  channels.push_back(mask);
  cv2.merge(channels, mask3);
  channels.delete();

  const invMask3 = new cv2.Mat();
  const ones = new cv2.Mat(mask3.rows, mask3.cols, mask3.type(), new cv2.Scalar(1, 1, 1, 1));
  cv2.subtract(ones, mask3, invMask3);
  ones.delete();

  const part1 = new cv2.Mat();
  const part2 = new cv2.Mat();
  cv2.multiply(patch, mask3, part1);
  cv2.multiply(background, invMask3, part2);
  mask3.delete();
  invMask3.delete();

  const result = new cv2.Mat();
  cv2.add(part1, part2, result);
  part1.delete();
  part2.delete();
  return result;
}

// ─── Reconstrucție din piramida Laplaciană ─────────────
// Parcurgem de la cel mai blur nivel la cel mai detaliat:
// current = pyrUp(current) + Laplacian[i]
function reconstructFromLaplacian(pyramid: any[]): any {
  let current = pyramid[pyramid.length - 1].clone();

  for (let i = pyramid.length - 2; i >= 0; i--) {
    const up = new cv2.Mat();
    cv2.pyrUp(current, up, new cv2.Size(pyramid[i].cols, pyramid[i].rows));
    current.delete();
    const result = new cv2.Mat();
    cv2.add(up, pyramid[i], result);
    up.delete();
    current = result;
  }
  return current;
}

// ─── Normalizare Laplacian pt vizualizare ─────────────
// Valorile Laplacian pot fi negative; le aducem la [0,255]
function normalizeLaplacianLevel(lap: any): any {
  const channels = new cv2.MatVector();
  cv2.split(lap, channels);
  let gMin = Infinity, gMax = -Infinity;
  for (let c = 0; c < channels.size(); c++) {
    const mm = cv2.minMaxLoc(channels.get(c));
    if (mm.minVal < gMin) gMin = mm.minVal;
    if (mm.maxVal > gMax) gMax = mm.maxVal;
  }
  channels.delete();
  const range = gMax - gMin || 1;
  const offset = new cv2.Mat(lap.rows, lap.cols, lap.type(), new cv2.Scalar(-gMin, -gMin, -gMin));
  const shifted = new cv2.Mat();
  cv2.add(lap, offset, shifted);
  offset.delete();
  const normalized = new cv2.Mat();
  shifted.convertTo(normalized, cv2.CV_8UC3, 255.0 / range);
  shifted.delete();
  return normalized;
}

// ─── UI helpers ───────────────────────────────────────
function addToContainer(parent: HTMLElement, label: string, mat: any) {
  const div = document.createElement('div');
  div.style.cssText = 'margin-bottom:16px; display:inline-block; margin-right:16px; vertical-align:top;';

  const title = document.createElement('p');
  title.textContent = label;
  title.style.cssText = 'margin:0 0 4px; font-weight:bold; color:#eee; font-size:13px;';

  const canvas = document.createElement('canvas');
  const display = new cv2.Mat();
  const ch = mat.channels();
  if (ch === 1) {
    cv2.cvtColor(mat, display, cv2.COLOR_GRAY2RGBA);
  } else if (ch === 3) {
    cv2.cvtColor(mat, display, cv2.COLOR_RGB2RGBA);
  } else {
    mat.copyTo(display);
  }
  cv2.imshow(canvas, display);
  display.delete();
  canvas.style.cssText = 'max-width:480px; height:auto; border:1px solid #333;';

  div.appendChild(title);
  div.appendChild(canvas);
  parent.appendChild(div);
}

function addSection(container: HTMLElement, title: string, description: string): HTMLElement {
  const section = document.createElement('div');
  section.style.cssText = 'margin-bottom:32px;';
  section.innerHTML = `
    <h2 style="color:#eee; margin:0 0 4px;">${title}</h2>
    <p style="color:#999; margin:0 0 12px; font-size:14px; line-height:1.5;">${description}</p>
  `;
  container.appendChild(section);
  return section;
}

// ─── Main blending logic ──────────────────────────────
function blend(bgImg: HTMLImageElement, patchImg: HTMLImageElement) {
  const output = document.getElementById('output')!;
  output.textContent = 'Processing...';

  // Yield to browser so "Processing..." is visible
  setTimeout(() => {
    try {
      doBlend(bgImg, patchImg, output);
    } catch (error) {
      output.textContent = `Error: ${String(error)}`;
      console.error(error);
    }
  }, 50);
}

function doBlend(bgImg: HTMLImageElement, patchImg: HTMLImageElement, container: HTMLElement) {
  // 1. Read images
  // Browser/canvas pipeline: keep images in RGB.
  const bgRGBA = cv2.imread(bgImg);
  const bg = new cv2.Mat();
  cv2.cvtColor(bgRGBA, bg, cv2.COLOR_RGBA2RGB);
  bgRGBA.delete();

  const patchRGBA = cv2.imread(patchImg);
  const patchRaw = new cv2.Mat();
  cv2.cvtColor(patchRGBA, patchRaw, cv2.COLOR_RGBA2RGB);
  patchRGBA.delete();

  // 2. Resize patch to fit inside bg (max 40% of bg size, keep aspect ratio)
  const maxPatchW = Math.floor(bg.cols * 0.4);
  const maxPatchH = Math.floor(bg.rows * 0.4);
  const patchScale = Math.min(1, maxPatchW / patchRaw.cols, maxPatchH / patchRaw.rows);
  const pw = Math.round(patchRaw.cols * patchScale);
  const ph = Math.round(patchRaw.rows * patchScale);
  const patchResized = new cv2.Mat();
  cv2.resize(patchRaw, patchResized, new cv2.Size(pw, ph), 0, 0, cv2.INTER_AREA);
  patchRaw.delete();

  // 3. Place patch in center — create full-size "patch layer" and mask
  const cx = Math.floor((bg.cols - pw) / 2);
  const cy = Math.floor((bg.rows - ph) / 2);

  // patchFull = copy of bg, with patch region overwritten
  const patchFull = bg.clone();
  const roi = patchFull.roi(new cv2.Rect(cx, cy, pw, ph));
  patchResized.copyTo(roi);
  roi.delete();
  patchResized.delete();

  // HARD binary mask: 255 inside patch, 0 outside.
  const mask = cv2.Mat.zeros(bg.rows, bg.cols, cv2.CV_8UC1);
  const maskRoi = mask.roi(new cv2.Rect(cx, cy, pw, ph));
  maskRoi.setTo(new cv2.Scalar(255));
  maskRoi.delete();

  // Naive paste for 1:1 comparison
  const naivePaste = bg.clone();
  const naiveRoi = naivePaste.roi(new cv2.Rect(cx, cy, pw, ph));
  const patchRegion = patchFull.roi(new cv2.Rect(cx, cy, pw, ph));
  patchRegion.copyTo(naiveRoi);
  naiveRoi.delete();
  patchRegion.delete();

  // Convert to float [0,1]
  const bgF = new cv2.Mat();
  const patchFullF = new cv2.Mat();
  const maskF = new cv2.Mat();
  bg.convertTo(bgF, cv2.CV_32FC3, 1.0 / 255.0);
  patchFull.convertTo(patchFullF, cv2.CV_32FC3, 1.0 / 255.0);
  mask.convertTo(maskF, cv2.CV_32FC1, 1.0 / 255.0);

  // Compute effective pyramid levels
  const minDim = Math.min(bg.rows, bg.cols);
  const effectiveLevels = Math.max(2, Math.min(LEVELS, Math.floor(Math.log2(minDim)) - 2));
  console.log(`effectiveLevels=${effectiveLevels}, image=${bg.cols}x${bg.rows}`);

  // Build Gaussian pyramids for images
  const gaussBg = buildGaussianPyramid(bgF, effectiveLevels);
  const gaussPatch = buildGaussianPyramid(patchFullF, effectiveLevels);

  // Build mask pyramid with AGGRESSIVE independent blur at each level.
  // pyrDown alone only spreads the boundary ~2px per level (too subtle).
  // We blur each level with a kernel = 40% of the level width → very wide transition.
  const gaussMask: any[] = [];
  let currentMaskLevel = maskF.clone();
  for (let i = 0; i < effectiveLevels; i++) {
    // Apply strong Gaussian blur at this level
    const blurred = new cv2.Mat();
    const levelMinDim = Math.min(currentMaskLevel.rows, currentMaskLevel.cols);
    let ksize = Math.round(levelMinDim * 0.4) | 1; // 40% of dimension, must be odd
    if (ksize < 3) ksize = 3;
    cv2.GaussianBlur(currentMaskLevel, blurred, new cv2.Size(ksize, ksize), 0);
    // Apply blur multiple times for wider spread
    cv2.GaussianBlur(blurred, blurred, new cv2.Size(ksize, ksize), 0);
    cv2.GaussianBlur(blurred, blurred, new cv2.Size(ksize, ksize), 0);
    gaussMask.push(blurred);

    const stats = cv2.minMaxLoc(blurred);
    console.log(`gaussMask[${i}] ${blurred.cols}x${blurred.rows} ksize=${ksize} min=${stats.minVal.toFixed(4)} max=${stats.maxVal.toFixed(4)}`);

    // Downsample for next level
    if (i < effectiveLevels - 1) {
      const down = new cv2.Mat();
      cv2.pyrDown(currentMaskLevel, down);
      currentMaskLevel.delete();
      currentMaskLevel = down;
    }
  }
  currentMaskLevel.delete();

  // Build Laplacian pyramids
  const lapBg = buildLaplacianPyramid(gaussBg);
  const lapPatch = buildLaplacianPyramid(gaussPatch);

  // Blend each level: patch where mask≈1, bg where mask≈0, gradient in between
  const blendedPyramid: any[] = [];
  for (let i = 0; i < lapBg.length; i++) {
    const blended = blendWithMask(lapPatch[i], lapBg[i], gaussMask[i]);
    blendedPyramid.push(blended);
  }

  // Reconstruct final image from blended Laplacian pyramid
  const reconstructedF = reconstructFromLaplacian(blendedPyramid);
  const resultMat = new cv2.Mat();
  reconstructedF.convertTo(resultMat, cv2.CV_8UC3, 255.0);

  // FLOAT-level comparison with naive paste
  const naiveF = new cv2.Mat();
  naivePaste.convertTo(naiveF, cv2.CV_32FC3, 1.0 / 255.0);
  const diffF = new cv2.Mat();
  cv2.absdiff(naiveF, reconstructedF, diffF);
  const diffChans = new cv2.MatVector();
  cv2.split(diffF, diffChans);
  const diffStats = cv2.minMaxLoc(diffChans.get(0));
  console.log(`FLOAT |naive - pyramid| ch0: min=${diffStats.minVal.toFixed(6)}, max=${diffStats.maxVal.toFixed(6)}`);
  diffChans.delete();
  // Convert diff to visible: amplify 50× and scale to uint8
  const diffVis = new cv2.Mat();
  diffF.convertTo(diffVis, cv2.CV_8UC3, 255.0 * 50.0);
  diffF.delete();
  naiveF.delete();
  reconstructedF.delete();

  // ──── RENDER OUTPUT ────
  container.innerHTML = '';

  // Section 1: Source images
  const s1 = addSection(container,
    '1. Imagini sursă',
    'Imaginea principală (fundalul) și patch-ul uploadat.'
  );
  addToContainer(s1, 'Fundal', bg);
  const patchVis = patchFull.roi(new cv2.Rect(cx, cy, pw, ph));
  addToContainer(s1, `Patch (${pw}×${ph}px)`, patchVis);
  patchVis.delete();

  // Section 2: Comparație
  const s2 = addSection(container,
    '2. Comparație: Naive vs Laplacian Pyramid Blend',
    'Naive = inserare directă (hard edges). Laplacian = blend prin piramidă (smooth edges).<br>'
    + 'Diferența (×50) evidențiază zona de tranziție smooth în jurul patch-ului.'
  );
  addToContainer(s2, 'Naive paste (hard edges)', naivePaste);
  addToContainer(s2, 'Laplacian Blend (smooth)', resultMat);
  addToContainer(s2, '|Diferență| ×50 (float)', diffVis);

  // Section 3: Mask at each pyramid level
  const s3 = addSection(container,
    '3. Masca la fiecare nivel de piramidă',
    'Masca blur-uită agresiv (kernel = 40% din rezoluția nivelului, 3 treceri). '
    + 'La nivele mici, masca e foarte smooth → tranziție graduală.'
  );
  for (let i = 0; i < gaussMask.length; i++) {
    const vis = new cv2.Mat();
    gaussMask[i].convertTo(vis, cv2.CV_8UC1, 255.0);
    addToContainer(s3, `Nivel ${i} (${gaussMask[i].cols}×${gaussMask[i].rows})`, vis);
    vis.delete();
  }

  // Section 4: Blended Laplacian pyramid levels
  const s4 = addSection(container,
    '4. Piramida Laplaciană (blend-uită)',
    'Detalii combinate la fiecare scară.'
  );
  for (let i = 0; i < blendedPyramid.length; i++) {
    const vis = normalizeLaplacianLevel(blendedPyramid[i]);
    addToContainer(s4, `Nivel ${i}`, vis);
    vis.delete();
  }

  // Cleanup
  bg.delete(); patchFull.delete(); mask.delete(); naivePaste.delete();
  resultMat.delete(); diffVis.delete();
  bgF.delete(); patchFullF.delete(); maskF.delete();
  gaussBg.forEach((m: any) => m.delete());
  gaussPatch.forEach((m: any) => m.delete());
  gaussMask.forEach((m: any) => m.delete());
  lapBg.forEach((m: any) => m.delete());
  lapPatch.forEach((m: any) => m.delete());
  blendedPyramid.forEach((m: any) => m.delete());
}

// ─── Init ─────────────────────────────────────────────
const mainImg = document.getElementById('image') as HTMLImageElement;
const patchInput = document.getElementById('patchInput') as HTMLInputElement;
const patchNameSpan = document.getElementById('patchName') as HTMLElement;
const outputDiv = document.getElementById('output') as HTMLElement;

(cv as any).onRuntimeInitialized = () => {
  cv2 = cv;
  console.log('OpenCV ready');
  outputDiv.textContent = 'OpenCV încărcat. Încarcă o imagine patch pentru a începe.';

  patchInput.addEventListener('change', () => {
    const file = patchInput.files?.[0];
    if (!file) return;

    patchNameSpan.textContent = file.name;

    const patchImg = new Image();
    patchImg.onload = () => {
      if (mainImg.complete && mainImg.naturalWidth > 0) {
        blend(mainImg, patchImg);
      } else {
        mainImg.onload = () => blend(mainImg, patchImg);
      }
    };
    patchImg.src = URL.createObjectURL(file);
  });
};

