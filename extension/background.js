importScripts('job_scheduler.js');

const DEFAULT_SETTINGS = {
  apiUrl: 'http://127.0.0.1:8000',
  targetLang: 'id',
  readingDirection: 'rtl',
  translationMode: 'inpaint',
  fontScale: 1.0,
  allCaps: true,
  autoTranslate: true,
  translator: 'google',
  llmMergeOcr: true,
  loadingStyle: 'default',
  enabledDomains: []
};

async function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(DEFAULT_SETTINGS, (items) => {
      resolve(items);
    });
  });
}

function formatApiError(detail, status) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map(item => `${(item.loc || []).join('.')}: ${item.msg || 'Invalid value'}`).join('; ');
  }
  return `HTTP error ${status}`;
}

async function apiRequest(endpoint, options = {}) {
  const { apiUrl, expectBlob, ...fetchOptions } = options;
  const baseUrl = apiUrl || (await getSettings()).apiUrl;
  const url = `${baseUrl.replace(/\/$/, '')}${endpoint}`;
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), fetchOptions.body instanceof FormData ? 360000 : 30000);
  
  try {
    const res = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal
    });
    
    if (!res.ok) {
      let errorMsg = `HTTP error ${res.status}`;
      try {
        const errJson = await res.json();
        if (errJson.detail) errorMsg = formatApiError(errJson.detail, res.status);
      } catch {}
      return { success: false, error: errorMsg };
    }
    
    if (expectBlob) {
      const blob = await res.blob();
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          resolve({ success: true, data: reader.result });
        };
        reader.onerror = () => {
          resolve({ success: false, error: "Failed to read image data." });
        };
        reader.readAsDataURL(blob);
      });
    }
    
    const data = await res.json();
    return { success: true, data: data };
  } catch (error) {
    if (error.name === 'AbortError') {
      return { success: false, error: "Request timed out (server might be down or busy)." };
    }
    return { success: false, error: `Network error: ${error.message}` };
  } finally {
    clearTimeout(timeoutId);
  }
}

function base64ToBlob(base64, mimeType = 'image/jpeg') {
  const raw = base64.includes(',') ? base64.split(',')[1] : base64;
  const byteCharacters = atob(raw);
  const byteArrays = [];
  
  for (let offset = 0; offset < byteCharacters.length; offset += 512) {
    const slice = byteCharacters.slice(offset, offset + 512);
    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }
  
  return new Blob(byteArrays, { type: mimeType });
}

async function inpaintStream(data, sender) {
  const request = { data };
  const apiUrl = request.data.apiUrl || (await getSettings()).apiUrl;
  const url = `${apiUrl.replace(/\/$/, '')}/api/v1/translate/inpaint-stream`;

  const formData = new FormData();
  if (request.data.jobId) formData.append('job_id', request.data.jobId);
  if (request.data.fileBlob) {
    formData.append('file', request.data.fileBlob, 'image.jpg');
  } else if (request.data.fileData) {
    const blob = base64ToBlob(request.data.fileData, request.data.mimeType || 'image/jpeg');
    formData.append('file', blob, 'image.jpg');
  }
  for (const [key, value] of Object.entries(request.data)) {
    if (!['fileBlob', 'fileData', 'mimeType', 'imageSrc', 'jobId', 'apiUrl', 'priority', 'pageTitle'].includes(key)) {
      formData.append(key, value);
    }
  }

  const controller = new AbortController();
  let timeoutId;
  const resetStreamTimeout = () => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => controller.abort(), 60000);
  };
  // Allow server queueing before headers, then use the normal idle timer.
  timeoutId = setTimeout(() => controller.abort(), 360000);

  try {
    const res = await fetch(url, {
      method: 'POST',
      body: formData,
      signal: controller.signal
    });
    resetStreamTimeout();

    if (!res.ok) {
      let errorMsg = `HTTP error ${res.status}`;
      try {
        const err = await res.json();
        if (err.detail) errorMsg = formatApiError(err.detail, res.status);
      } catch {}
      return { success: false, error: errorMsg };
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalResult = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) buffer += decoder.decode() + '\n';
      if (value && value.length) resetStreamTimeout();

      if (value) buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const data = JSON.parse(trimmed);
          if (data.stage === 'heartbeat') continue;
          if (data.stage === 'done') {
            finalResult = {
              success: true,
              data: data.image_base64,
              totalDetected: data.total_detected,
              durationMs: data.duration_ms
            };
          } else if (data.stage === 'cancelled') {
            finalResult = { success: false, cancelled: true, error: data.message || 'Translation cancelled.' };
          } else if (data.stage === 'error') {
            finalResult = { success: false, error: data.message };
          } else {
            if (sender.tab?.id != null) {
              await chrome.tabs.sendMessage(sender.tab.id, {
                action: 'pipelineProgress',
                stage: data.stage,
                message: data.message,
                jobId: request.data.jobId,
                imageSrc: request.data.imageSrc,
                totalBubbles: data.total_bubbles
              }, sender.documentId ? { documentId: sender.documentId } :
                { frameId: sender.frameId ?? 0 }).catch(() => {});
            }
          }
        } catch (parseErr) {
          console.warn('JSON parse error in stream chunk:', parseErr);
        }
      }
      if (done) break;
    }

    if (finalResult) {
      return finalResult;
    } else {
      return { success: false, error: 'The stream ended without data.' };
    }
  } catch (fetchErr) {
    if (fetchErr.name === 'AbortError') {
      return { success: false, error: 'Request timed out (server may be busy).' };
    } else {
      return { success: false, error: fetchErr.message };
    }
  } finally {
    clearTimeout(timeoutId);
  }
}

const translationScheduler = new TranslationScheduler(inpaintStream,
  job => apiRequest(`/api/v1/translate/jobs/${encodeURIComponent(job.data.jobId)}/cancel`, {
    apiUrl: job.data.apiUrl, method: 'POST'
  }),
  (job, stage) => chrome.tabs.sendMessage(job.sender.tab.id, {
    action: 'pipelineProgress', jobId: job.data.jobId, stage,
    activeTab: translationScheduler.activeInfo()
  }, job.sender.documentId ? { documentId: job.sender.documentId } :
    { frameId: job.sender.frameId ?? 0 }).catch(() => {}));

chrome.tabs.onRemoved.addListener(tabId => translationScheduler.closeTab(tabId));
let focusRevision = 0;
async function refreshTranslationFocus() {
  const revision = ++focusRevision;
  try {
    const window = await chrome.windows.getLastFocused();
    const tabs = await chrome.tabs.query({ active: true, windowId: window.id });
    if (revision === focusRevision && tabs[0]?.id != null) {
      translationScheduler.setFocusedTab(tabs[0].id);
    }
  } catch {}
}
chrome.tabs.onActivated.addListener(() => refreshTranslationFocus());
chrome.windows.onFocusChanged.addListener(windowId => {
  // Leaving Chrome must preserve the last selected tab.
  if (windowId !== chrome.windows.WINDOW_ID_NONE) refreshTranslationFocus();
});
refreshTranslationFocus();
chrome.tabs.onUpdated.addListener((tabId, change) => {
  if (change.status === 'loading') translationScheduler.closeTab(tabId);
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  (async () => {
    try {
      if (request.action === 'translationTabWork') {
        if (sender.tab?.id != null) translationScheduler.setTabWork(sender.tab.id, request.pending);
        sendResponse({ success: true });
      }
      else if (request.action === 'checkHealth') {
        const result = await apiRequest('/api/v1/health', { method: 'GET' });
        sendResponse(result);
      } 
      else if (request.action === 'translatePage' || request.action === 'inpaintPage' || request.action === 'ocrRecognizeCrop') {
        const formData = new FormData();
        
        if (request.data.fileData) {
            const blob = base64ToBlob(request.data.fileData, request.data.mimeType || 'image/jpeg');
            formData.append('file', blob, 'image.jpg');
        }
        
        for (const [key, value] of Object.entries(request.data)) {
            if (key !== 'fileData' && key !== 'mimeType') {
                formData.append(key, value);
            }
        }
        
        let endpoint = '';
        let expectBlob = false;
        
        if (request.action === 'translatePage') endpoint = '/api/v1/translate/page';
        if (request.action === 'inpaintPage') {
            endpoint = '/api/v1/translate/inpaint-page';
            if (request.data.return_format === 'image') {
                expectBlob = true;
            }
        }
        if (request.action === 'ocrRecognizeCrop') endpoint = '/api/v1/ocr/recognize-crop';
        
        const result = await apiRequest(endpoint, {
          method: 'POST',
          body: formData,
          expectBlob: expectBlob
        });
        
        sendResponse(result);
      }
      else if (request.action === 'inpaintPageStream') {
        sendResponse(await translationScheduler.enqueue(request.data, sender));
      }
      else if (request.action === 'cancelTranslationJob') {
        sendResponse(await (translationScheduler.cancelOwned(request.jobId, sender) || apiRequest(`/api/v1/translate/jobs/${encodeURIComponent(request.jobId)}/cancel`, {
          apiUrl: request.apiUrl, method: 'POST'
        })));
      }
      else if (request.action === 'translateDialogues') {
        const result = await apiRequest('/api/v1/translate/dialogues', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(request.data)
        });
        sendResponse(result);
      }
      else if (request.action === 'fetchImage') {
        try {
          const fetchHeaders = { 'Accept': 'image/*' };
          if (sender.tab && sender.tab.url) {
            fetchHeaders['Referer'] = sender.tab.url;
          }
          const imgRes = await fetch(request.url, { headers: fetchHeaders });
          if (!imgRes.ok) {
            throw new Error(`Image request failed with HTTP ${imgRes.status}`);
          }
          const blob = await imgRes.blob();
          const reader = new FileReader();
          const dataUrl = await new Promise((resolve, reject) => {
            reader.onloadend = () => resolve(reader.result);
            reader.onerror = () => reject(new Error('Failed to read blob'));
            reader.readAsDataURL(blob);
          });
          sendResponse({ success: true, data: dataUrl });
        } catch (err) {
          sendResponse({ success: false, error: `Failed to fetch image: ${err.message}` });
        }
      }
      else if (request.action === 'getConfig') {
        const settings = await getSettings();
        sendResponse({ success: true, mode: settings.translationMode, ...settings });
      }
      else {
        sendResponse({ success: false, error: 'Unknown action' });
      }
    } catch (err) {
      sendResponse({ success: false, error: err.toString() });
    }
  })();
  
  return true;
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "translateMangaPage",
    title: "Translate this Manga Page",
    contexts: ["image"]
  });
});

let contextSelectionCounter = 0;
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "translateMangaPage" && info.srcUrl) {
    const selectedAt = Date.now() * 1000 + (++contextSelectionCounter % 1000);
    const jobId = crypto.randomUUID();
    const target = { frameId: info.frameId ?? 0 };
    try {
      const settings = await getSettings();
      const started = await chrome.tabs.sendMessage(tab.id, {
          action: 'contextMenuTranslateStart',
          srcUrl: info.srcUrl, jobId, settings, selectedAt
      }, target);
      if (!started?.success) throw new Error(started?.error || 'The image is no longer available.');
      
      // The content script owns scheduling, progress and result application.
      // Context-menu requests use the same single slot as automatic translation.
      
    } catch (error) {
      console.error("Context menu translation error:", error);
      await chrome.tabs.sendMessage(tab.id, {
          action: 'contextMenuTranslateError',
          srcUrl: info.srcUrl,
          jobId,
          error: error.toString()
      }, target).catch(() => {});
    }
  }
});
