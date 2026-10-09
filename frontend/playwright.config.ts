import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: 'e2e',
  timeout: 180_000,
  use: { baseURL: 'http://localhost:5173', headless: true },
  webServer: [
    { command: '/opt/anaconda3/bin/python -m backend', cwd: '..', port: 5001, reuseExistingServer: true, timeout: 30_000 },
    { command: 'npm run dev', port: 5173, reuseExistingServer: true, timeout: 30_000 },
  ],
})
