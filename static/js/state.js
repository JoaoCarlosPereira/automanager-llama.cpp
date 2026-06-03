/** Estado compartilhado da dashboard (objeto mutável para ESM + testes). */
export const state = {
    logStream: null,
    startTime: null,
    currentSelectedModel: null,
    currentRunningModelPath: null,
    manualGpuOverride: false,
    autoBalancePending: false,
    metricsTimer: null,
    downloadsTimer: null,
    modelsTimer: null,
    statusPollIntervalMs: 3000,
    statusPollTimer: null,
};
