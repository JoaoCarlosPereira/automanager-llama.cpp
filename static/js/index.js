import { state } from './state.js';
import { handleLogin, handleLogout, changePassword, apiFetch, handleShutdown, handleUpdate } from './auth.js';
import { cancelAutoBalance } from './gpu.js';
import {
    syncContextSizeCustomVisibility,
    getContextSize, setContextSize, balanceWeights, updateTotal, resetToDefaults,
    hideAutoBalanceCapacityAlert, showMtpWarning, hideMtpWarning,
} from './gpu.js';
import {
    stopDashboardPolling, startDashboardPolling, renewToken, updateMetrics, updateStatus,
    updateDownloads, clearCompletedDownloads,
} from './metrics.js';
import {
    initDashboard, selectModel, applyModelConfig, setDefaultModel,
    startModel, stopModel, renameModel, deleteModel, downloadModel, updateModels,
    saveModelsDir, openVisionImportModal, closeVisionImportModal, submitVisionImport,
    onMmprojChange, closeTab,
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
win.getContextSize = getContextSize;
win.setContextSize = setContextSize;
win.stopModel = stopModel;
win.startModel = startModel;
win.renameModel = renameModel;
win.deleteModel = deleteModel;
win.setDefaultModel = setDefaultModel;
win.selectModel = selectModel;
win.closeTab = closeTab;
win.applyModelConfig = applyModelConfig;
win.renewToken = renewToken;
win.updateTotal = updateTotal;
win.balanceWeights = balanceWeights;
win.resetToDefaults = resetToDefaults;
win.hideAutoBalanceCapacityAlert = hideAutoBalanceCapacityAlert;
win.showMtpWarning = showMtpWarning;
win.hideMtpWarning = hideMtpWarning;
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

win.toggleSidebar = (force) => {
    const sidebar = document.getElementById('sidebar');
    const main = document.getElementById('main-content');
    if (!sidebar) return;
    const isMobile = window.matchMedia('(max-width: 768px)').matches;
    if (isMobile) {
        // Mobile: sidebar is hidden by default; '.open' slides it in.
        const willOpen = force === undefined ? !sidebar.classList.contains('open') : !!force;
        sidebar.classList.toggle('open', willOpen);
    } else {
        // Desktop: '.collapsed' hides the sidebar; '.full' removes the left margin.
        const isOpen = !sidebar.classList.contains('collapsed');
        const willOpen = force === undefined ? !isOpen : !!force;
        sidebar.classList.toggle('collapsed', !willOpen);
        if (main) main.classList.toggle('full', !willOpen);
    }
};

win.getInstanceChatUrl = (port) => `${window.location.origin}/ui/${port}/`;

win.openNativeChat = (port) => {
    window.open(win.getInstanceChatUrl(port), '_blank', 'noopener,noreferrer');
};

const apiLinkEl = document.getElementById('api-link');
if (apiLinkEl) apiLinkEl.innerText = `http://${window.fixedIp}:[PORTA]/v1`;

const dashboardEl = document.getElementById('dashboard');
if (dashboardEl && dashboardEl.style.display !== 'none') {
    initDashboard();
    startDashboardPolling();
}
