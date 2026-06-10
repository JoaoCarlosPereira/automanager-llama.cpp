import { state } from './state.js';
import { handleLogin, handleLogout, changePassword, apiFetch, handleShutdown, handleUpdate } from './auth.js';
import { cancelAutoBalance } from './gpu.js';
import {
    syncContextSizeCustomVisibility, onContextSizePresetChange, onContextSizeCustomInput,
    getContextSize, setContextSize, balanceWeights, updateTotal, resetToDefaults,
    hideAutoBalanceCapacityAlert, showMtpWarning, hideMtpWarning,
} from './gpu.js';
import {
    stopDashboardPolling, startDashboardPolling, renewToken, updateMetrics, updateStatus,
    updateDownloads, clearCompletedDownloads,
} from './metrics.js';
import {
    initDashboard, getModelButtonsHtml, selectModel, applyModelConfig, setDefaultModel,
    startModel, stopModel, renameModel, deleteModel, downloadModel, updateModels,
    saveModelsDir, openVisionImportModal, closeVisionImportModal, submitVisionImport,
    onMmprojChange,
} from './models.js';
import { checkForUpdates, dismissVersionModal } from './version.js';

window.modelConfigs = window.modelConfigs || {};

const win = window;
win.initDashboard = initDashboard;
win.startDashboardPolling = startDashboardPolling;
win.stopDashboardPolling = stopDashboardPolling;
win.handleLogin = handleLogin;
win.handleLogout = handleLogout;
win.changePassword = changePassword;
win.handleShutdown = handleShutdown;
win.handleUpdate = handleUpdate;
win.apiFetch = apiFetch;
win.cancelAutoBalance = cancelAutoBalance;
win.onContextSizePresetChange = onContextSizePresetChange;
win.onContextSizeCustomInput = onContextSizeCustomInput;
win.getContextSize = getContextSize;
win.setContextSize = setContextSize;
win.stopModel = stopModel;
win.startModel = startModel;
win.renameModel = renameModel;
win.deleteModel = deleteModel;
win.setDefaultModel = setDefaultModel;
win.selectModel = selectModel;
win.applyModelConfig = applyModelConfig;
win.renewToken = renewToken;
win.updateTotal = updateTotal;
win.balanceWeights = balanceWeights;
win.resetToDefaults = resetToDefaults;
win.hideAutoBalanceCapacityAlert = hideAutoBalanceCapacityAlert;
win.showMtpWarning = showMtpWarning;
win.hideMtpWarning = hideMtpWarning;
win.getModelButtonsHtml = getModelButtonsHtml;
win.updateModels = updateModels;
win.saveModelsDir = saveModelsDir;
win.updateStatus = updateStatus;
win.downloadModel = downloadModel;
win.openVisionImportModal = openVisionImportModal;
win.closeVisionImportModal = closeVisionImportModal;
win.submitVisionImport = submitVisionImport;
win.onMmprojChange = onMmprojChange;
win.updateMetrics = updateMetrics;
win.updateDownloads = updateDownloads;
win.clearCompletedDownloads = clearCompletedDownloads;
win.checkForUpdates = checkForUpdates;
win.dismissVersionModal = dismissVersionModal;

document.getElementById('chat-link').href = `http://${window.fixedIp}:8085/`;
document.getElementById('api-link').innerText = `http://${window.fixedIp}:8085/v1`;

if (document.getElementById('dashboard').style.display !== 'none') {
    initDashboard();
    startDashboardPolling();
}
