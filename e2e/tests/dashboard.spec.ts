import { test, expect, Page } from '@playwright/test';

type Problems = {
  consoleErrors: string[];
  pageErrors: string[];
  failedRequests: string[];
  serverErrors: string[];
};

/** Collects everything the browser complains about, so failures are readable in CI. */
function watchProblems(page: Page): Problems {
  const problems: Problems = {
    consoleErrors: [],
    pageErrors: [],
    failedRequests: [],
    serverErrors: [],
  };

  page.on('console', (msg) => {
    if (msg.type() === 'error') problems.consoleErrors.push(msg.text());
  });
  page.on('pageerror', (err) => problems.pageErrors.push(err.message));
  page.on('requestfailed', (req) => {
    problems.failedRequests.push(`${req.method()} ${req.url()} - ${req.failure()?.errorText ?? 'failed'}`);
  });
  page.on('response', (res) => {
    if (res.status() >= 500) problems.serverErrors.push(`${res.status()} ${res.url()}`);
  });

  return problems;
}

function dump(label: string, problems: Problems) {
  const lines: string[] = [];
  if (problems.pageErrors.length) lines.push(`uncaught exceptions:\n  ${problems.pageErrors.join('\n  ')}`);
  if (problems.serverErrors.length) lines.push(`5xx responses:\n  ${problems.serverErrors.join('\n  ')}`);
  if (problems.failedRequests.length) lines.push(`failed requests:\n  ${problems.failedRequests.join('\n  ')}`);
  if (problems.consoleErrors.length) lines.push(`console errors:\n  ${problems.consoleErrors.join('\n  ')}`);
  if (lines.length) console.log(`\n[${label}] browser problems:\n${lines.join('\n')}\n`);
}

test('dashboard loads and the websocket connects through the proxy', async ({ page }) => {
  const problems = watchProblems(page);

  // Registered before navigation: the app opens the socket on mount.
  const socketPromise = page.waitForEvent('websocket', { timeout: 60_000 });

  await page.goto('/', { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('heading', { name: 'PredPump Radar' })).toBeVisible();

  const socket = await socketPromise;
  console.log(`websocket url: ${socket.url()}`);
  expect(socket.url()).toContain('/ws');
  expect(socket.isClosed(), 'handshake was rejected (check WS_ALLOWED_ORIGINS and the nginx /ws location)').toBe(false);

  // The header reflects the real socket state, so this is an end-to-end check
  // of nginx -> api-gateway -> browser.
  await expect(page.getByText(/WS Status: connected/i)).toBeVisible({ timeout: 60_000 });

  dump('websocket', problems);
  expect(problems.pageErrors, 'uncaught exceptions on the dashboard').toEqual([]);
});

test('every tab renders without crashing', async ({ page }) => {
  const problems = watchProblems(page);
  await page.goto('/', { waitUntil: 'domcontentloaded' });

  const tabs = [
    'Market Monitor',
    'ML Signals',
    'Signals History',
    'Learning Dashboard',
    'ML Metrics',
    'Trader Mind',
  ];

  for (const tab of tabs) {
    const button = page.getByRole('button', { name: new RegExp(tab, 'i') }).first();
    await expect(button, `tab "${tab}" is missing`).toBeVisible();
    await button.click();
    // React unmounts the whole tree on an unhandled render error, so the header
    // still being there is a meaningful liveness check.
    await expect(page.getByRole('heading', { name: 'PredPump Radar' })).toBeVisible();
    await page.waitForTimeout(1500);
    await page.screenshot({ path: `screenshots/${tab.replace(/\s+/g, '-').toLowerCase()}.png`, fullPage: true });
  }

  dump('tabs', problems);
  expect(problems.pageErrors, 'uncaught exceptions while switching tabs').toEqual([]);
});

test('gateway REST endpoints answer through the frontend proxy', async ({ request }) => {
  const endpoints = [
    '/api/v1/market/pairs',
    '/api/v1/ml/metrics',
    '/api/v1/ml/signals/recent?limit=5&hours=24',
    '/api/v1/infrastructure/metrics',
  ];

  const broken: string[] = [];
  for (const endpoint of endpoints) {
    const res = await request.get(endpoint, { timeout: 30_000 });
    console.log(`${res.status()} ${endpoint}`);
    // 503 is acceptable while the ML engine is still cold; 5xx other than that
    // and any 4xx mean the route or the proxy is wrong.
    if (res.status() >= 500 && res.status() !== 503) broken.push(`${res.status()} ${endpoint}`);
    if (res.status() >= 400 && res.status() < 500) broken.push(`${res.status()} ${endpoint}`);
  }

  expect(broken, 'endpoints answering with an unexpected status').toEqual([]);
});
