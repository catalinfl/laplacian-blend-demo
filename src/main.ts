import cv from "@techstark/opencv-js";
import "./style.css";

function waitForOpenCV(): Promise<void> {
  return new Promise((resolve) => {
    if ((cv as any).Mat) {
      resolve();
      return;
    }
    (cv as any).onRuntimeInitialized = () => resolve();
  });
}

function waitForImage(img: HTMLImageElement): Promise<void> {
  return new Promise((resolve) => {
    if (img.complete && img.naturalWidth > 0) {
      resolve();
      return;
    }
    img.onload = () => resolve();
  });

}
function buildGaussianPyramid(src: InstanceType<typeof cv.Mat>, levels: number) {
  const pyramid: InstanceType<typeof cv.Mat>[] = [src.clone()];
  for (let i = 1; i < levels; i++) {
    const prev = pyramid[i - 1];
    const down = new cv.Mat();
    // explicitly compute the target size (ceil to avoid losing pixels)
    const dstSize = new cv.Size(
      Math.ceil(prev.cols / 2),
      Math.ceil(prev.rows / 2)
    );
    cv.pyrDown(prev, down, dstSize);
    pyramid.push(down);
  }
  return pyramid;
}

function buildLaplacianPyramid(gaussPyr: InstanceType<typeof cv.Mat>[]) {
  const lapPyr: InstanceType<typeof cv.Mat>[] = [];

  for (let i = 0; i < gaussPyr.length - 1; i++) {
    // up-sample the next (smaller) level back to current size
    const up = new cv.Mat();
    cv.pyrUp(gaussPyr[i + 1], up, gaussPyr[i].size());

    // convert both to 64F before subtracting to preserve negatives
    const curF = new cv.Mat();
    const upF = new cv.Mat();
    gaussPyr[i].convertTo(curF, cv.CV_64FC3);
    up.convertTo(upF, cv.CV_64FC3);

    // laplacian up-sampled next level
    const lap = new cv.Mat();
    cv.subtract(curF, upF, lap);
    lapPyr.push(lap);
    up.delete();
    curF.delete();
    upF.delete();
  }
  // last level is low res, converted to 64F for consistency
  const lastF = new cv.Mat();
  gaussPyr[gaussPyr.length - 1].convertTo(lastF, cv.CV_64FC3);
  lapPyr.push(lastF);
  return lapPyr;
}

function blendLaplacianPyramids(lapA: InstanceType<typeof cv.Mat>[], lapB: InstanceType<typeof cv.Mat>[], maskPyr: InstanceType<typeof cv.Mat>[]) {
  const blended: InstanceType<typeof cv.Mat>[] = [];

  for (let i = 0; i < lapA.length; i++) {
    const m = new cv.Mat();
    maskPyr[i].convertTo(m, cv.CV_64FC3, 1.0 / 255.0);

    const aMasked = new cv.Mat();
    const oneMinusM = new cv.Mat();
    const bMasked = new cv.Mat();
    const result = new cv.Mat();

    cv.multiply(lapA[i], m, aMasked);

    const ones = new cv.Mat(m.rows, m.cols, m.type(), new cv.Scalar(1, 1, 1, 1));
    cv.subtract(ones, m, oneMinusM);
    cv.multiply(lapB[i], oneMinusM, bMasked);

    cv.add(aMasked, bMasked, result);

    blended.push(result);

    m.delete();
    aMasked.delete(); oneMinusM.delete(); bMasked.delete(); ones.delete();
  }
  return blended;
}

function reconstructFromLaplacian(lapPyr: InstanceType<typeof cv.Mat>[]) {
  let current = lapPyr[lapPyr.length - 1].clone();

  for (let i = lapPyr.length - 2; i >= 0; i--) {
    const up = new cv.Mat();
    cv.pyrUp(current, up, lapPyr[i].size());
    const next = new cv.Mat();
    cv.add(up, lapPyr[i], next);
    current.delete();
    up.delete();
    current = next;
  }
  return current;
}

function freePyramid(pyr: InstanceType<typeof cv.Mat>[]) {
  pyr.forEach((m) => m.delete());
}

function createMask(rows: number, cols: number) {
  const mask = new cv.Mat(rows, cols, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 128));
  const halfCols = Math.floor(cols / 2);

  // left half to white
  const roi = mask.roi(new cv.Rect(0, 0, halfCols, rows));
  roi.setTo(new cv.Scalar(255, 255, 255, 255));
  roi.delete();

  return mask;
}

function directConnection(imgA: InstanceType<typeof cv.Mat>, imgB: InstanceType<typeof cv.Mat>) {
  const result = imgB.clone();
  const halfCols = Math.floor(imgA.cols / 2);

  const srcRoi = imgA.roi(new cv.Rect(0, 0, halfCols, imgA.rows));
  const dstRoi = result.roi(new cv.Rect(0, 0, halfCols, result.rows));
  srcRoi.copyTo(dstRoi);
  srcRoi.delete();
  dstRoi.delete();

  return result;
}

async function main() {
  const statusEl = document.getElementById("status")!;

  statusEl.textContent = "w8 opencv";
  await waitForOpenCV();
  statusEl.textContent = "load images";

  const imgAppleEl = document.getElementById("imgApple") as HTMLImageElement;
  const imgOrangeEl = document.getElementById("imgOrange") as HTMLImageElement;
  await Promise.all([waitForImage(imgAppleEl), waitForImage(imgOrangeEl)]);

  statusEl.textContent = "Processing …";

  const levelsInput = document.getElementById("levels") as HTMLInputElement;
  const levelsVal = document.getElementById("levelsVal")!;

  const minDim = Math.min(imgAppleEl.naturalHeight, imgAppleEl.naturalWidth);
  const maxLevels = Math.max(1, Math.floor(Math.log2(minDim)));
  levelsInput.max = String(maxLevels);
  if (parseInt(levelsInput.value, 10) > maxLevels) {
    levelsInput.value = String(maxLevels);
  }

  function run() {
    const levels = parseInt(levelsInput.value, 10);
    levelsVal.textContent = String(levels);

    const matA = cv.imread(imgAppleEl);  
    const matB = cv.imread(imgOrangeEl); 

    if (matA.rows !== matB.rows || matA.cols !== matB.cols) {
      cv.resize(matB, matB, matA.size());
    }

    const a3 = new cv.Mat();
    const b3 = new cv.Mat();
    cv.cvtColor(matA, a3, cv.COLOR_RGBA2RGB);
    cv.cvtColor(matB, b3, cv.COLOR_RGBA2RGB);
    matA.delete();
    matB.delete();

    // pad images symmetrically to keep seam centered across levels
    const divisor = Math.pow(2, levels);
    const extraRows = (divisor - (a3.rows % divisor)) % divisor;
    const extraCols = (divisor - (a3.cols % divisor)) % divisor;
    const padTop = Math.floor(extraRows / 2);
    const padBottom = extraRows - padTop;
    const padLeft = Math.floor(extraCols / 2);
    const padRight = extraCols - padLeft;
    if (extraRows > 0 || extraCols > 0) {
      cv.copyMakeBorder(a3, a3, padTop, padBottom, padLeft, padRight, cv.BORDER_REFLECT);
      cv.copyMakeBorder(b3, b3, padTop, padBottom, padLeft, padRight, cv.BORDER_REFLECT);
    }
    // Original display dimensions and top-left crop offset in padded image
    const origRows = a3.rows - extraRows;
    const origCols = a3.cols - extraCols;

    // original images
    const a3Display = a3.roi(new cv.Rect(padLeft, padTop, origCols, origRows));
    const b3Display = b3.roi(new cv.Rect(padLeft, padTop, origCols, origRows));
    showOnCanvas("canvasApple", a3Display);
    showOnCanvas("canvasOrange", b3Display);
    a3Display.delete();
    b3Display.delete();

    // direct connection (no blending)
    const directResult = directConnection(a3, b3);
    const directCropped = directResult.roi(new cv.Rect(padLeft, padTop, origCols, origRows));
    showOnCanvas("canvasDirect", directCropped);
    directCropped.delete();
    directResult.delete();

    // laplacian pyramid blending
    const aF = new cv.Mat();
    const bF = new cv.Mat();
    a3.convertTo(aF, cv.CV_64FC3, 1.0 / 255.0);
    b3.convertTo(bF, cv.CV_64FC3, 1.0 / 255.0);

    const mask = createMask(a3.rows, a3.cols);

    const gpA = buildGaussianPyramid(aF, levels);
    const gpB = buildGaussianPyramid(bF, levels);
    const gpM = buildGaussianPyramid(mask, levels);

    const lpA = buildLaplacianPyramid(gpA);
    const lpB = buildLaplacianPyramid(gpB);

    const blendedPyr = blendLaplacianPyramids(lpA, lpB, gpM);

    const reconstructed = reconstructFromLaplacian(blendedPyr);

    // clip to [0, 255] and convert back to 8 bit for display
    const result8 = new cv.Mat();
    reconstructed.convertTo(result8, cv.CV_8UC3, 255.0);

    const resultCropped = result8.roi(new cv.Rect(padLeft, padTop, origCols, origRows));
    showOnCanvas("canvasPyramid", resultCropped);

    // cleanup 
    resultCropped.delete();
    result8.delete();
    reconstructed.delete();
    freePyramid(blendedPyr);
    freePyramid(lpA);
    freePyramid(lpB);
    freePyramid(gpA);
    freePyramid(gpB);
    freePyramid(gpM);
    mask.delete();
    aF.delete();
    bF.delete();
    a3.delete();
    b3.delete();

    statusEl.textContent = `level ${levels}`;
  }

  function showOnCanvas(canvasId: string, mat: InstanceType<typeof cv.Mat>) {
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    // Convert to RGBA for canvas display
    const rgba = new cv.Mat();
    if (mat.channels() === 3) {
      cv.cvtColor(mat, rgba, cv.COLOR_RGB2RGBA);
    } else {
      mat.copyTo(rgba);
    }
    // Ensure 8-bit
    if (rgba.type() !== cv.CV_8UC4) {
      const tmp = new cv.Mat();
      rgba.convertTo(tmp, cv.CV_8UC4);
      rgba.delete();
      cv.imshow(canvas, tmp);
      tmp.delete();
    } else {
      cv.imshow(canvas, rgba);
      rgba.delete();
    }
  }

  // Initial run
  run();

  // re-run when levels change
  levelsInput.addEventListener("input", () => run());
}

main().catch(console.error);
