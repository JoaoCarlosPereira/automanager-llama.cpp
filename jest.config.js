/** @type {import('jest').Config} */
const config = {
  testEnvironment: 'jsdom',
  testMatch: ['**/static/js/*.test.js'],
  coverageDirectory: 'coverage/js',
  collectCoverageFrom: [
    'static/js/*.js',
    '!static/js/*.test.js',
    '!static/js/pacman_bg.js',
  ],
  coverageThreshold: {
    global: {
      statements: 90,
      branches: 90,
      functions: 90,
      lines: 90,
    },
  },
  moduleNameMapper: {
    '^(\\.{1,2}/.*)\\.js$': '$1',
  },
  transform: {},
  moduleFileExtensions: ['js'],
};

export default config;
