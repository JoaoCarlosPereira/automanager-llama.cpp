import { state } from './state.js?v=4.1.0';
import { handleLogin, handleLogout, changePassword, apiFetch, handleShutdown, handleUpdate } from './auth.js?v=4.1.0';
import { cancelAutoBalance } from './gpu.js?v=4.1.0';
import {
    syncContextSizeCustomVisibility,
    getContextSize, setContextSize, balanceWeights, updateTotal, resetToDefaults,
    hideAutoBalanceCapacityAlert, showMtpWarning, hideMtpWarning,
} from './gpu.js?v=4.1.0';
import {
    stopDashboardPolling, startDashboardPolling, renewToken, copyApiToken, updateMetrics, updateStatus,
    updateDownloads, clearCompletedDownloads, cancelDownload,
} from './metrics.js?v=4.1.0';
import {
    initDashboard, selectModel, selectModelFromEvent, applyModelConfig, setDefaultModel,
    startModel, stopModel, renameModel, deleteModel, downloadModel, updateModels,
    saveModelsDir, openVisionImportModal, closeVisionImportModal, submitVisionImport,
    onMmprojChange, closeTab, startSmartCalibration,
} from './models.js?v=4.1.0';
import { checkForUpdates, dismissVersionModal } from './version.js?v=4.1.0';

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
win.selectModelFromEvent = selectModelFromEvent;
win.closeTab = closeTab;
win.applyModelConfig = applyModelConfig;
win.renewToken = renewToken;
win.copyApiToken = copyApiToken;
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
win.cancelDownload = cancelDownload;
win.checkForUpdates = checkForUpdates;
win.dismissVersionModal = dismissVersionModal;
win.startSmartCalibration = startSmartCalibration;
win.runSmartCalibration = (btn) => {
    const tab = btn?.closest('.tab-content');
    if (!tab?.dataset?.path || !tab.id) return;
    startSmartCalibration(tab.dataset.path, tab.id);
};

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
