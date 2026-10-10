import { defineConfig } from '@playwright/test'

// Runs against a fixture backend with synthetic data, so it works on a fresh clone.
// Set PYTHON if `python` on your PATH does not have the backend requirements installed.
const python = process.env.PYTHON ?? 'python'

export default defineConfig({
  testDir: 'e2e',
  timeout: 180_000,
  use: { baseURL: 'http://localhost:5174', headless: true },
  webServer: [
    { command: `${python} frontend/e2e/fixture_server.py`, cwd: '..', port: 5002, reuseExistingServer: false, timeout: 60_000 },
    { command: 'npm run dev -- --port 5174 --strictPort', env: { VITE_API_TARGET: 'http://localhost:5002' }, port: 5174, reuseExistingServer: false, timeout: 60_000 },
  ],
})
