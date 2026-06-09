import { DOM } from "../core/dom.js";

export async function requestJson(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
        throw await responseToError(response);
    }
    return await response.json();
}

export async function uploadChatImage(file) {
    return await uploadFileWithProgress("/api/attachments/images", file);
}

export async function uploadChatFile(file) {
    return await uploadFileWithProgress("/api/attachments/files", file);
}

export async function deleteAttachment(attachmentId) {
    const response = await fetch(`/api/attachments/${encodeURIComponent(attachmentId)}`, {
        method: "DELETE",
    });
    if (!response.ok) {
        throw await responseToError(response);
    }
}

export async function requestChatStream(payload, handlers) {
    const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        throw await responseToError(response);
    }

    if (!response.body) {
        return await requestJson("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalPayload = null;

    while (true) {
        const { done, value } = await reader.read();
        if (done) {
            break;
        }

        buffer += decoder.decode(value, { stream: true });
        const parsed = await consumeChatStreamBuffer(buffer, handlers);
        buffer = parsed.buffer;
        if (parsed.finalPayload) {
            finalPayload = parsed.finalPayload;
        }
    }

    buffer += decoder.decode();
    const parsed = await consumeChatStreamBuffer(buffer, handlers);
    if (parsed.finalPayload) {
        finalPayload = parsed.finalPayload;
    }

    if (!finalPayload) {
        throw new Error("Chat stream ended without a final response.");
    }

    return finalPayload;
}

export async function requestChatJob(jobId) {
    return await requestJson(`/api/chat/jobs/${encodeURIComponent(jobId)}`);
}

export async function requestChatJobTrace(jobId) {
    return await requestJson(`/api/chat/jobs/${encodeURIComponent(jobId)}/trace`);
}

export async function cancelChatJob(jobId) {
    return await requestJson(
        `/api/chat/jobs/${encodeURIComponent(jobId)}/cancel`,
        {
            method: "POST",
        },
    );
}

export async function consumeChatStreamBuffer(buffer, handlers) {
    let remaining = buffer;
    let finalPayload = null;

    while (true) {
        const newlineIndex = remaining.indexOf("\n");
        if (newlineIndex === -1) {
            break;
        }

        const line = remaining.slice(0, newlineIndex).trim();
        remaining = remaining.slice(newlineIndex + 1);
        if (!line) {
            continue;
        }

        let event;
        try {
            event = JSON.parse(line);
        } catch (_error) {
            throw new Error(`Invalid chat stream event: ${line}`);
        }

        if (event.type === "chunk") {
            if (handlers && typeof handlers.onChunk === "function") {
                await handlers.onChunk(event.delta || "");
            }
            continue;
        }
        if (event.type === "done") {
            finalPayload = event;
            continue;
        }
        if (event.type === "error") {
            throw new Error(event.message || "Chat stream failed.");
        }
    }

    return {
        buffer: remaining,
        finalPayload: finalPayload,
    };
}

export async function responseToError(response) {
    let detail = `${response.status} ${response.statusText}`;
    try {
        const payload = await response.json();
        if (payload && typeof payload.detail === "string") {
            detail = payload.detail;
        }
    } catch (error) {
        console.warn("Non-JSON error response", error);
    }
    return new Error(detail);
}

async function uploadFileWithProgress(url, file) {
    return await new Promise((resolve, reject) => {
        const formData = new FormData();
        formData.append("file", file);

        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener("progress", (event) => {
            if (!event.lengthComputable) {
                return;
            }
            const percent = Math.round((event.loaded / event.total) * 100);
            if (DOM.composerStatusBanner) {
                DOM.composerStatusBanner.hidden = false;
                DOM.composerStatusBanner.textContent = `アップロード中... ${percent}%`;
            }
        });

        xhr.addEventListener("load", () => {
            if (DOM.composerStatusBanner) {
                DOM.composerStatusBanner.hidden = true;
            }
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    resolve(JSON.parse(xhr.responseText));
                } catch (error) {
                    reject(new Error("Invalid server response"));
                }
            } else {
                reject(buildXhrError(xhr));
            }
        });

        xhr.addEventListener("error", () => {
            if (DOM.composerStatusBanner) {
                DOM.composerStatusBanner.hidden = true;
            }
            reject(new Error("Network error"));
        });

        xhr.addEventListener("abort", () => {
            if (DOM.composerStatusBanner) {
                DOM.composerStatusBanner.hidden = true;
            }
            reject(new Error("Upload aborted"));
        });

        xhr.open("POST", url);
        xhr.send(formData);
    });
}

function buildXhrError(xhr) {
    let detail = `${xhr.status} ${xhr.statusText}`;
    try {
        const payload = JSON.parse(xhr.responseText);
        if (payload && typeof payload.detail === "string") {
            detail = payload.detail;
        }
    } catch (_error) {
        // Non-JSON error response — use the status text.
    }
    return new Error(detail);
}
