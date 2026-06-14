/** Estado compartilhado da dashboard (objeto mutável para ESM + testes). */
export const state = {
    logStream: null,
    logStreamPort: null,
    logStreamTabId: null,
    logStreamSessionKey: null,
    startTime: null,
    currentSelectedModel: null,
    currentRunningModelPath: null,
    activeInstances: [],
    currentActivePort: 8085,
    manualGpuOverride: false,
    autoBalancePending: false,
    autoBalanceTabId: null,
    autoBalanceSeenActive: false,
    autoBalanceRunId: null,
    metricsTimer: null,
    downloadsTimer: null,
    modelsTimer: null,
    statusPollIntervalMs: 3000,
    statusPollTimer: null,
    activeTabs: [], // [{id, path, name}]
    currentTabId: null,
    initialTabsSynced: false,
    lastModelsList: [],
    lastConfig: {},
    modelLogs: {}, // {modelPath: 'log text...'}
};
