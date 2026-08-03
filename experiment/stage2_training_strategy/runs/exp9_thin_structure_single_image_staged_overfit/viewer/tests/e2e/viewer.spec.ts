import { expect, test } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import path from "node:path";

test("switches experiments and preserves the Exp9 three-pane interactions", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "多实验点云对比器" }),
  ).toBeVisible();
  await expect(
    page.getByTestId(
      "experiment-exp12_hypersim_100_immediate_joint_finetuning",
    ),
  ).toBeVisible();
  await page
    .getByTestId(
      "experiment-exp9_thin_structure_single_image_staged_overfit",
    )
    .click();
  await expect(page.getByText("Exp9 · 单图极限过拟合").first()).toBeVisible();
  await expect(page.getByTestId("sample-5")).toBeVisible();
  await expect(page.getByTestId("sample-4")).toBeVisible();
  await expect(page.getByTestId("sample-6")).toBeVisible();
  await expect(page.getByTestId("viewer-left")).toContainText("训练前 · K=0");
  await expect(page.getByTestId("viewer-right")).toContainText("训练后 · K=3");
  await expect(page.getByTestId("viewer-ground-truth")).toContainText(
    "Hypersim Ground Truth",
  );
  await expect(page.locator("canvas")).toHaveCount(3);

  await page
    .getByTestId("left-interaction")
    .getByRole("button", { name: "平移" })
    .click();
  await expect(
    page
      .getByTestId("left-interaction")
      .getByRole("button", { name: "平移" }),
  ).toHaveAttribute("aria-pressed", "true");

  await page.getByTestId("sample-4").click();
  await expect(page.getByText("悬空楼梯踏板与支架").first()).toBeVisible();

  await page
    .getByTestId("left-k")
    .getByRole("button", { name: "K=5" })
    .click();
  await expect(page.getByTestId("viewer-left")).toContainText(
    "零初始化 SSR",
  );
  const groundTruthCanvas = page
    .getByTestId("viewer-ground-truth")
    .locator("canvas");
  await expect(groundTruthCanvas).toHaveAttribute(
    "data-scene-source",
    /04_ai_054_008_cam_00_frame\.0000\/ground_truth\.ply/,
  );
  const groundTruthSource = await groundTruthCanvas.getAttribute(
    "data-scene-source",
  );
  const leftCanvas = page.getByTestId("viewer-left").locator("canvas");
  const preservationBox = await leftCanvas.boundingBox();
  if (preservationBox) {
    await page.mouse.move(
      preservationBox.x + preservationBox.width * 0.55,
      preservationBox.y + preservationBox.height * 0.52,
    );
    await page.mouse.down();
    await page.mouse.move(
      preservationBox.x + preservationBox.width * 0.63,
      preservationBox.y + preservationBox.height * 0.47,
      { steps: 5 },
    );
    await page.mouse.up();
  }
  await page.waitForTimeout(500);
  const cameraBeforeStageChange = await leftCanvas.getAttribute(
    "data-camera-snapshot",
  );
  expect(cameraBeforeStageChange).toBeTruthy();
  const initialSceneSource = await leftCanvas.getAttribute(
    "data-scene-source",
  );
  expect(initialSceneSource).toBeTruthy();
  await page
    .getByTestId("left-stage")
    .getByRole("button", { name: "训练后" })
    .click();
  await expect(page.getByTestId("viewer-left")).toContainText("训练后 · K=5");
  await expect(leftCanvas).not.toHaveAttribute(
    "data-scene-source",
    initialSceneSource!,
  );
  await expect(leftCanvas).toHaveAttribute(
    "data-camera-snapshot",
    cameraBeforeStageChange!,
  );
  await expect(groundTruthCanvas).toHaveAttribute(
    "data-scene-source",
    groundTruthSource!,
  );

  const cameraBeforeStepChange = await leftCanvas.getAttribute(
    "data-camera-snapshot",
  );
  const finalK5SceneSource = await leftCanvas.getAttribute(
    "data-scene-source",
  );
  await page
    .getByTestId("left-k")
    .getByRole("button", { name: "K=1" })
    .click();
  await expect(page.getByTestId("viewer-left")).toContainText("训练后 · K=1");
  await expect(leftCanvas).not.toHaveAttribute(
    "data-scene-source",
    finalK5SceneSource!,
  );
  await expect(leftCanvas).toHaveAttribute(
    "data-camera-snapshot",
    cameraBeforeStepChange!,
  );

  await page
    .getByTestId("coordinate-mode")
    .getByRole("button", { name: "原始输出" })
    .click();
  await expect(page.locator(".raw-warning")).toContainText("相对尺度");

  await page
    .getByTestId("render-mode")
    .getByRole("button", { name: "SSR 体素壳" })
    .click();
  await expect(page.getByText(/round\(200·log Z\)/)).toBeVisible();

  await page
    .getByTestId("render-mode")
    .getByRole("button", { name: "彩色点云" })
    .click();
  await page
    .getByTestId("coordinate-mode")
    .getByRole("button", { name: "GT 对齐" })
    .click();
  await page
    .getByTestId("left-k")
    .getByRole("button", { name: "K=0" })
    .click();
  const cameraSync = page.getByRole("checkbox");
  if (!(await cameraSync.isChecked())) {
    await cameraSync.check();
  }
  await page
    .getByTestId("raster-scope")
    .getByRole("button", { name: "完整场景" })
    .click();
  await expect(page.locator(".canvas-error")).toHaveCount(0);
  const canvasBox = await leftCanvas.boundingBox();
  if (canvasBox) {
    await page.mouse.move(
      canvasBox.x + canvasBox.width * 0.55,
      canvasBox.y + canvasBox.height * 0.52,
    );
    await page.mouse.down();
    await page.mouse.move(
      canvasBox.x + canvasBox.width * 0.65,
      canvasBox.y + canvasBox.height * 0.48,
      { steps: 5 },
    );
    await page.mouse.up();
    await page.mouse.wheel(0, -180);
  }
  await page
    .getByTestId("raster-scope")
    .getByRole("button", { name: "细结构裁剪" })
    .click();
  const screenshotDirectory = path.resolve(
    process.cwd(),
    "../artifacts/viewer_acceptance",
  );
  await mkdir(screenshotDirectory, { recursive: true });
  for (const order of [5, 4, 6]) {
    await page.getByTestId(`sample-${order}`).click();
    await expect(page.locator("canvas")).toHaveCount(3);
    await page.waitForTimeout(800);
    if (process.env.UPDATE_ACCEPTANCE_SCREENSHOTS === "1") {
      await page.screenshot({
        path: path.join(screenshotDirectory, `sample_${order}_dual_view.png`),
        fullPage: true,
      });
    }
  }
});

test("shows a clear startup error when the manifest cannot be loaded", async ({
  page,
}) => {
  await page.route("**/data/experiments.json", (route) =>
    route.fulfill({ status: 503, body: "unavailable" }),
  );
  await page.goto("/");
  await expect(page.locator("main[role='alert']")).toContainText(
    "实验目录请求失败：HTTP 503",
  );
});

test("shows Exp29 improvements and degradations in every split", async ({
  page,
}) => {
  await page.goto(
    "/?experiment=exp29_smooth_bounded_residual_long_joint&split=train&sample=1&leftK=0&rightK=3",
  );
  await expect(
    page.getByText("Exp29 · 长程联合训练细杆案例").first(),
  ).toBeVisible();
  await expect(page.locator(".sample-switcher button")).toHaveCount(5);
  await expect(page.getByTestId("viewer-ground-truth")).toContainText(
    "Hypersim Ground Truth",
  );
  await expect(page.locator("canvas")).toHaveCount(3);
  const groundTruthCanvas = page
    .getByTestId("canvas-ground-truth")
    .locator("canvas");
  const leftPredictionCanvas = page.getByTestId("canvas-left").locator("canvas");
  const rightPredictionCanvas = page
    .getByTestId("canvas-right")
    .locator("canvas");
  await expect(groundTruthCanvas).toHaveAttribute(
    "data-camera-snapshot",
    /position/,
  );
  await expect(leftPredictionCanvas).toHaveAttribute(
    "data-camera-snapshot",
    /position/,
  );
  await expect(rightPredictionCanvas).toHaveAttribute(
    "data-camera-snapshot",
    /position/,
  );
  await page
    .getByTestId("viewer-ground-truth")
    .getByRole("button", { name: "适配视野" })
    .click();
  await page.waitForTimeout(300);
  const synchronizedCamera = await groundTruthCanvas.getAttribute(
    "data-camera-snapshot",
  );
  expect(synchronizedCamera).toBeTruthy();
  await expect(leftPredictionCanvas).toHaveAttribute(
    "data-camera-snapshot",
    synchronizedCamera!,
  );
  await expect(rightPredictionCanvas).toHaveAttribute(
    "data-camera-snapshot",
    synchronizedCamera!,
  );
  await expect(page.getByTestId("sample-1")).toContainText("改善样本");
  await expect(page.getByTestId("sample-4")).toContainText("退化样本");
  await expect(page.getByTestId("viewer-left")).toContainText(
    "最佳检查点 · K=0",
  );
  await expect(page.getByTestId("viewer-right")).toContainText(
    "最佳检查点 · K=3",
  );
  await expect(page.locator("[data-testid='left-stage']")).toHaveCount(0);
  await page.getByTestId("sample-1").hover();
  await expect(
    page.getByTestId("sample-1").locator(".sample-hover-preview"),
  ).toBeVisible();
  await page.mouse.move(0, 0);
  const screenshotDirectory = path.resolve(
    process.cwd(),
    "../../../../stage3_stability_normalization/runs/exp29_smooth_bounded_residual_long_joint/results/viewer_acceptance",
  );
  if (process.env.UPDATE_ACCEPTANCE_SCREENSHOTS === "1") {
    await mkdir(screenshotDirectory, { recursive: true });
  }
  for (const split of ["val", "test", "train"] as const) {
    await page.getByTestId(`split-${split}`).click();
    await expect(page.locator(".sample-switcher button")).toHaveCount(5);
    await expect(page.getByTestId("sample-1")).toContainText("改善样本");
    await expect(page.getByTestId("sample-4")).toContainText("退化样本");
    if (process.env.UPDATE_ACCEPTANCE_SCREENSHOTS === "1") {
      for (const [sample, outcome] of [
        [1, "improved"],
        [4, "degraded"],
      ] as const) {
        await page.getByTestId(`sample-${sample}`).click();
        await page.mouse.move(0, 0);
        await page.waitForTimeout(800);
        await page.screenshot({
          path: path.join(
            screenshotDirectory,
            `exp29_${split}_${outcome}_triple_view.png`,
          ),
          fullPage: true,
        });
      }
    }
  }
  await expect(page.locator(".canvas-error")).toHaveCount(0);
});

test("loads Exp12 train, validation, and test samples", async ({ page }) => {
  await page.goto("/");
  await page
    .getByTestId(
      "experiment-exp12_hypersim_100_immediate_joint_finetuning",
    )
    .click();
  await expect(page.getByText("Exp12 · 100 张联合微调").first()).toBeVisible();
  await expect(page.getByTestId("sample-1")).toContainText("训练样本");
  await page.getByTestId("split-val").click();
  await expect(page.getByTestId("sample-2")).toContainText("验证样本");
  await page.getByTestId("split-test").click();
  await expect(page.getByTestId("sample-3")).toContainText("测试样本");
  await page.getByTestId("split-train").click();
  await expect(page.getByTestId("viewer-left")).toContainText(
    "联合前 · K=0",
  );
  await expect(page.getByTestId("viewer-right")).toContainText(
    "联合后 · K=3",
  );
  await expect(page.getByTestId("viewer-ground-truth")).toContainText(
    "Hypersim Ground Truth",
  );
  await expect(page.locator("canvas")).toHaveCount(3);
  await page
    .getByTestId("left-k")
    .getByRole("button", { name: "K=5" })
    .click();
  await expect(page.getByTestId("viewer-left")).not.toContainText("资源别名");
  await expect(page.locator(".canvas-error")).toHaveCount(0);
  await page
    .getByTestId("left-k")
    .getByRole("button", { name: "K=0" })
    .click();

  if (process.env.UPDATE_ACCEPTANCE_SCREENSHOTS === "1") {
    const screenshotDirectory = path.resolve(
      process.cwd(),
      "../../exp12_hypersim_100_immediate_joint_finetuning/results/viewer_acceptance",
    );
    await mkdir(screenshotDirectory, { recursive: true });
    for (const [split, order] of [
      ["train", 1],
      ["val", 2],
      ["test", 3],
    ] as const) {
      await page.getByTestId(`split-${split}`).click();
      await page.getByTestId(`sample-${order}`).click();
      await expect(page.locator("canvas")).toHaveCount(3);
      await expect(page.locator(".canvas-error")).toHaveCount(0);
      await page.waitForTimeout(800);
      await page.screenshot({
        path: path.join(
          screenshotDirectory,
          `exp12_sample_${order}_dual_view.png`,
        ),
        fullPage: true,
      });
    }
  }
});

test("browses Exp20 by split, picture and K without resetting the camera", async ({
  page,
}) => {
  await page.goto(
    "/?experiment=exp20_stateless_ssr_batch_statistics&split=train&sample=1&leftStage=initial&rightStage=final&leftK=0&rightK=3",
  );
  await expect(page.getByText("Exp20 · 训练域最佳推理配置").first()).toBeVisible();
  await expect(page.getByTestId("sample-1")).toContainText(
    "ai_013_001_cam_00_frame.0000",
  );
  await expect(page.locator(".sample-switcher button")).toHaveCount(5);
  await expect(page.getByTestId("viewer-left")).toContainText(
    "训练前 · K=0",
  );
  await expect(page.getByTestId("viewer-right")).toContainText(
    "训练后 · K=3",
  );
  await expect(page.getByTestId("viewer-ground-truth")).toContainText(
    "Hypersim Ground Truth",
  );
  await expect(page.locator("canvas")).toHaveCount(3);

  const groundTruthCanvas = page
    .getByTestId("viewer-ground-truth")
    .locator("canvas");
  await expect(groundTruthCanvas).toHaveAttribute(
    "data-scene-source",
    /ground_truth\.ply/,
  );
  const groundTruthSource = await groundTruthCanvas.getAttribute(
    "data-scene-source",
  );
  const leftCanvas = page.getByTestId("viewer-left").locator("canvas");
  await expect(leftCanvas).toHaveAttribute("data-camera-snapshot", /position/);
  const cameraBefore = await leftCanvas.getAttribute("data-camera-snapshot");
  await page
    .getByTestId("left-k")
    .getByRole("button", { name: "K=1" })
    .click();
  await expect(page.getByTestId("viewer-left")).toContainText(
    "训练前 · K=1",
  );
  await expect(page.getByTestId("viewer-left")).toContainText("资源别名");
  await expect(leftCanvas).toHaveAttribute(
    "data-camera-snapshot",
    cameraBefore!,
  );
  await expect(page).toHaveURL(/leftK=1/);

  const cameraBeforeStageChange = await leftCanvas.getAttribute(
    "data-camera-snapshot",
  );
  const initialSource = await leftCanvas.getAttribute("data-scene-source");
  await page
    .getByTestId("left-stage")
    .getByRole("button", { name: "训练后" })
    .click();
  await expect(page.getByTestId("viewer-left")).toContainText(
    "训练后 · K=1",
  );
  await expect(leftCanvas).not.toHaveAttribute(
    "data-scene-source",
    initialSource!,
  );
  await expect(leftCanvas).toHaveAttribute(
    "data-camera-snapshot",
    cameraBeforeStageChange!,
  );
  await expect(groundTruthCanvas).toHaveAttribute(
    "data-scene-source",
    groundTruthSource!,
  );
  await expect(page).toHaveURL(/leftStage=final/);

  for (const split of ["val", "test", "train"] as const) {
    await page.getByTestId(`split-${split}`).click();
    await expect(page.locator(".sample-switcher button")).toHaveCount(5);
    await page.getByTestId("sample-5").click();
    await expect(page).toHaveURL(new RegExp(`split=${split}.*sample=5`));
    await expect(page.locator(".canvas-error")).toHaveCount(0);
  }

  if (process.env.UPDATE_ACCEPTANCE_SCREENSHOTS === "1") {
    const screenshotDirectory = path.resolve(
      process.cwd(),
      "../../../../stage3_stability_normalization/runs/exp20_stateless_ssr_batch_statistics/results/viewer_acceptance",
    );
    await mkdir(screenshotDirectory, { recursive: true });
    await page
      .getByTestId("left-k")
      .getByRole("button", { name: "K=0" })
      .click();
    await page
      .getByTestId("left-stage")
      .getByRole("button", { name: "训练前" })
      .click();
    await page
      .getByTestId("right-stage")
      .getByRole("button", { name: "训练后" })
      .click();
    await page
      .getByTestId("right-k")
      .getByRole("button", { name: "K=3" })
      .click();
    for (const split of ["train", "val", "test"] as const) {
      await page.getByTestId(`split-${split}`).click();
      await page.getByTestId("sample-1").click();
      await page.waitForTimeout(800);
      await page.screenshot({
        path: path.join(screenshotDirectory, `${split}_dual_view.png`),
        fullPage: true,
      });
    }
  }
});
