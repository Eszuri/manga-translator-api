const DEFAULT_SETTINGS = {
  apiUrl: 'http://127.0.0.1:8000',
  targetLang: 'id',
  detectorType: 'hybrid',
  readingDirection: 'rtl',
  device: 'auto',
  translationMode: 'inpaint',
  fontScale: 1.0,
  allCaps: true,
  autoTranslate: true,
  translator: 'google',
  enabledDomains: []
};

async function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(DEFAULT_SETTINGS, (items) => {
      resolve(items);
    });
  });
}

async function apiRequest(endpoint, options = {}) {
  const settings = await getSettings();
  const url = `${settings.apiUrl.replace(/\/$/, '')}${endpoint}`;
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000);
  
  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    
    if (!res.ok) {
      let errorMsg = `HTTP error ${res.status}`;
      try {
        const errJson = await res.json();
        if (errJson.detail) errorMsg = errJson.detail;
      } catch (e) {}
      return { success: false, error: errorMsg };
    }
    
    if (options.expectBlob) {
      const blob = await res.blob();
      return new Promise((resolve, reject) => {
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
  const byteCharacters = atob(base64.split(',')[1]);
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

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  (async () => {
    try {
      if (request.action === 'checkHealth') {
        const result = await apiRequest('/api/v1/health', { method: 'GET' });
        sendResponse(result);
      } 
      else if (request.action === 'translatePage' || request.action === 'inpaintPage' || request.action === 'ocrRecognizeCrop') {
        const formData = new FormData();
        
        if (request.data.fileData) {
            const blob = base64ToBlob(request.data.fileData, request.data.mimeType || 'image/jpeg');
            formData.append('file', blob, 'image.jpg');
        }
        
        // Append other fields
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
        const settings = await getSettings();
        const url = `${settings.apiUrl.replace(/\/$/, '')}/api/v1/translate/inpaint-stream`;
        
        const formData = new FormData();
        if (request.data.fileData) {
          const blob = base64ToBlob(request.data.fileData, request.data.mimeType || 'image/jpeg');
          formData.append('file', blob, 'image.jpg');
        }
        for (const [key, value] of Object.entries(request.data)) {
          if (key !== 'fileData' && key !== 'mimeType' && key !== 'imageSrc') {
            formData.append(key, value);
          }
        }
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000);
        
        try {
          const res = await fetch(url, {
            method: 'POST',
            body: formData,
            signal: controller.signal
          });
          
          if (!res.ok) {
            let errorMsg = `HTTP error ${res.status}`;
            try {
              const err = await res.json();
              if (err.detail) errorMsg = err.detail;
            } catch (e) {}
            sendResponse({ success: false, error: errorMsg });
            return;
          }
          
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';
          let finalResult = null;
          
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            
            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed) continue;
              try {
                const data = JSON.parse(trimmed);
                if (data.stage === 'done') {
                  finalResult = {
                    success: true,
                    data: data.image_base64,
                    totalDetected: data.total_detected,
                    durationMs: data.duration_ms
                  };
                } else if (data.stage === 'error') {
                  finalResult = { success: false, error: data.message };
                } else {
                  if (sender.tab && sender.tab.id) {
                    chrome.tabs.sendMessage(sender.tab.id, {
                      action: 'pipelineProgress',
                      stage: data.stage,
                      message: data.message,
                      imageSrc: request.data.imageSrc,
                      totalBubbles: data.total_bubbles
                    }).catch(() => {});
                  }
                }
              } catch (parseErr) {
                console.warn('JSON parse error in stream chunk:', parseErr);
              }
            }
          }
          
          if (finalResult) {
            sendResponse(finalResult);
          } else {
            sendResponse({ success: false, error: 'Koneksi stream berakhir tanpa data.' });
          }
        } catch (fetchErr) {
          if (fetchErr.name === 'AbortError') {
            sendResponse({ success: false, error: 'Request timeout (server sibuk).' });
          } else {
            sendResponse({ success: false, error: fetchErr.message });
          }
        } finally {
          clearTimeout(timeoutId);
        }
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
          const imgRes = await fetch(request.url);
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

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "translateMangaPage" && info.srcUrl) {
    try {
      chrome.tabs.sendMessage(tab.id, { 
          action: 'contextMenuTranslateStart', 
          srcUrl: info.srcUrl 
      }).catch(err => {
          console.warn("Content script not ready or error:", err);
      });
      
      const settings = await getSettings();
      
      const imgRes = await fetch(info.srcUrl);
      const imgBlob = await imgRes.blob();
      
      const formData = new FormData();
      formData.append('file', imgBlob, 'image.jpg');
      formData.append('target_lang', settings.targetLang);
      formData.append('detector_type', settings.detectorType);
      formData.append('reading_direction', settings.readingDirection);
      formData.append('device', settings.device);
      formData.append('typeset', 'true');
      formData.append('font_scale', settings.fontScale.toString());
      formData.append('all_caps', settings.allCaps.toString());
      formData.append('return_format', settings.translationMode === 'inpaint' ? 'image' : 'json');
      
      const endpoint = settings.translationMode === 'inpaint' ? '/api/v1/translate/inpaint-page' : '/api/v1/translate/page';
      
      const result = await apiRequest(endpoint, {
          method: 'POST',
          body: formData,
          expectBlob: settings.translationMode === 'inpaint'
      });
      
      chrome.tabs.sendMessage(tab.id, {
          action: 'contextMenuTranslateResult',
          srcUrl: info.srcUrl,
          result: result,
          mode: settings.translationMode
      });
      
    } catch (error) {
      console.error("Context menu translation error:", error);
      chrome.tabs.sendMessage(tab.id, {
          action: 'contextMenuTranslateError',
          srcUrl: info.srcUrl,
          error: error.toString()
      });
    }
  }
});
