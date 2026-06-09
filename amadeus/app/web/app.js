import {
    cancelChatJob,
    deleteAttachment,
    requestChatJob,
    requestChatJobTrace,
    requestChatStream,
    requestJson,
    responseToError,
    uploadChatFile,
    uploadChatImage,
} from "./modules/api.js";
import { wireAppEvents } from "./bootstrap/wire-events.js";
import { createUiStatusController } from "./bootstrap/ui-status.js";
import { appState } from "./core/store.js";
import { createAsrModule } from "./features/asr.js";
import { createChatModule } from "./features/chat/index.js";
import { createLayoutModule } from "./features/layout/index.js";
import { createLive2DModule } from "./features/live2d/index.js";
import { createRolesModule } from "./features/roles.js";
import { createSessionsModule } from "./features/sessions.js";
import { createTtsModule } from "./features/tts.js";
import {
    addMessage,
    addSystemMessage,
    clearMessages,
    initializeMessageInteractions,
    removeMessage,
    updateMessage,
} from "./modules/messages.js";
import { createTraceModule } from "./modules/traces.js";
import {
    clamp,
    delay,
    formatTimestamp,
    normalizeSessionName,
    roundTo,
    smoothValue,
} from "./modules/utils.js";

const status = createUiStatusController();
const layout = createLayoutModule({
    addMessage,
    formatTimestamp,
    requestJson,
    setRunStatus: status.setRunStatus,
});
const live2d = createLive2DModule({
    clamp,
    requestJson,
    roundTo,
    responseToError,
    setRunStatus: status.setRunStatus,
});
const tts = createTtsModule({
    addMessage,
    applyMouthValue: live2d.applyMouthValue,
    clamp,
    requestJson,
    responseToError,
    setConnectionState: status.setConnectionState,
    setRunStatus: status.setRunStatus,
    smoothValue,
});
const asr = createAsrModule({
    addSystemMessage,
    clamp,
    delay,
    ensureAudioContextReady: tts.ensureAudioContextReady,
    requestJson,
    responseToError,
    setConnectionState: status.setConnectionState,
    setRunStatus: status.setRunStatus,
    stopSpeechPlayback: tts.stopSpeechPlayback,
});

tts.bindHooks({
    updateVoiceInputControls: asr.updateVoiceInputControls,
});

const sessions = createSessionsModule({
    addMessage,
    addSystemMessage,
    clearMessages,
    formatTimestamp,
    normalizeSessionName,
    requestJson,
    speakText: tts.speakText,
    setRunStatus: status.setRunStatus,
    stopSpeechPlayback: tts.stopSpeechPlayback,
});
const roles = createRolesModule({
    addMessage,
    normalizeSessionName,
    requestJson,
    setRunStatus: status.setRunStatus,
});
const traces = createTraceModule();
const chat = createChatModule({
    addMessage,
    applySessionSummaries: sessions.applySessionSummaries,
    cancelChatJob,
    createSpeechSession: tts.createSpeechSession,
    drainVoicePromptQueue: asr.drainVoicePromptQueue,
    deleteAttachment,
    ensureAudioContextReady: tts.ensureAudioContextReady,
    finalizeSpeechSession: tts.finalizeSpeechSession,
    normalizeSessionName,
    queueSpeechSessionText: tts.queueSpeechSessionText,
    removeMessage,
    requestChatJob,
    requestChatJobTrace,
    requestChatStream,
    requestSessionSummaries: sessions.requestSessionSummaries,
    resetTracePanel: traces.resetTracePanel,
    setActiveBackgroundJob: status.setActiveBackgroundJob,
    setChatBusy: status.setChatBusy,
    setRunStatus: status.setRunStatus,
    speakText: tts.speakText,
    startTracePanel: traces.startTracePanel,
    stopSpeechPlayback: tts.stopSpeechPlayback,
    syncCurrentSessionFromServer: sessions.syncCurrentSessionFromServer,
    uploadChatFile,
    uploadChatImage,
    applyTracePayload: traces.applyTracePayload,
    updateMessage,
});

status.bindFeatures({
    asr,
    chat,
    roles,
    sessions,
});

sessions.bindRoleHooks({
    closeRoleEditor: roles.closeRoleEditor,
    syncRolePanelForCurrentSession: roles.syncRolePanelForCurrentSession,
});
roles.bindSessionHooks({
    applySessionDetail: sessions.applySessionDetail,
});

document.addEventListener("DOMContentLoaded", initializePage);

async function initializePage() {
    layout.ensureSidebarToggleButtons();
    layout.initializeLive2DDrawer();
    layout.initializePageSplit();
    initializeMessageInteractions();
    wireAppEvents({
        asr,
        chat,
        layout,
        live2d,
        roles,
        sessions,
        status,
        tts,
    });

    layout.restoreSettingsPanelState();
    layout.restoreCronPanelState();
    layout.restoreHeartbeatPanelState();
    layout.restoreLive2DPanelState();
    layout.restoreRuntimePanelState();
    layout.restoreStageBackgroundPanelState();
    layout.restoreStageEffectsPanelState();
    layout.handleSettingsPanelToggle();
    live2d.setStageMessage("正在加载 Live2D 模型…");
    addSystemMessage("正在连接 Amadeus…");

    let config;
    try {
        config = await requestJson("/api/web/config");
        appState.config = config;
        layout.applyRuntimeConfig(config.runtime);
    } catch (error) {
        console.error(error);
        status.setConnectionState("error", "配置加载失败");
        status.setRunStatus(error.message || "配置加载失败");
        live2d.setStageMessage(error.message || "配置加载失败");
        addSystemMessage(`配置加载失败：${error.message || error}`);
        return;
    }

    const failures = [];

    try {
        const activeLive2DConfig = live2d.applyConfigToUI(config);
        live2d.initializePixiApplication();
        const live2dLoadPromise = live2d.loadLive2DModel(activeLive2DConfig);
        live2d.renderLive2DControls(activeLive2DConfig);
        await live2dLoadPromise;
        live2d.renderLive2DControls(appState.config.live2d);
    } catch (error) {
        console.error("Live2D init failed:", error);
        failures.push(`Live2D: ${error.message || error}`);
        live2d.setStageMessage(`Live2D 加载失败：${error.message || error}`);
    }

    layout.restoreSessionSidebarState();
    layout.restoreRoleSidebarState();

    try {
        await sessions.initializeSessionPanel(config.session_name);
    } catch (error) {
        console.error("Sessions init failed:", error);
        failures.push(`会话: ${error.message || error}`);
    }

    try {
        await roles.initializeRolePanel();
    } catch (error) {
        console.error("Roles init failed:", error);
        failures.push(`角色: ${error.message || error}`);
    }

    try {
        await tts.loadTtsOptions(config.tts);
    } catch (error) {
        console.error("TTS init failed:", error);
        failures.push(`TTS: ${error.message || error}`);
    }

    try {
        asr.applyAsrStatus(config.asr);
        asr.startAsrStatusPolling();
    } catch (error) {
        console.error("ASR init failed:", error);
        failures.push(`ASR: ${error.message || error}`);
    }

    traces.resetTracePanel();

    if (failures.length === 0) {
        status.setConnectionState("ready", "已连接");
        status.setRunStatus("准备就绪");
    } else {
        status.setConnectionState("ready", "部分功能不可用");
        status.setRunStatus(`${failures.length} 个组件加载失败`);
        addSystemMessage(`部分组件加载失败：${failures.join("；")}`);
    }
    status.setActiveBackgroundJob("");
}
