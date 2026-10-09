document.addEventListener('DOMContentLoaded', () => {
  const serverStatusDot = document.getElementById('serverStatusDot');
  const serverStatusText = document.getElementById('serverStatusText');
  const statStatus = document.getElementById('statStatus');

  const siteHostname = document.getElementById('siteHostname');
  const siteToggle = document.getElementById('siteToggle');

  const translator = document.getElementById('translator');
  const llmOcrMode = document.getElementById('llmOcrMode');
  const llmOcrModeRow = document.getElementById('llmOcrModeRow');
  const targetLang = document.getElementById('targetLang');
  const readingDirection = document.getElementById('readingDirection');
  const loadingStyle = document.getElementById('loadingStyle');
  const apiUrl = document.getElementById('apiUrl');
  const btnTranslate = document.getElementById('btnTranslate');

  let currentDomain = '';
  let healthCheckInterval = null;

  chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
    if (tab && tab.url) {
      try {
        const url = new URL(tab.url);
        if (url.protocol.startsWith('http')) {
          currentDomain = url.hostname;
          siteHostname.textContent = currentDomain;
          loadSiteStatus(currentDomain);
          return;
        }
      } catch {}
    }
    siteHostname.textContent = 'This page';
    siteToggle.disabled = true;
  });

  loadSettings();
  loadStats();
  checkApiHealth();
  healthCheckInterval = setInterval(checkApiHealth, 8000);

  siteToggle.addEventListener('change', () => {
    if (!currentDomain) return;
    const isEnabled = siteToggle.checked;

    chrome.storage.sync.get({ enabledDomains: [] }, (items) => {
      let enabledList = items.enabledDomains || [];
      if (isEnabled) {
        if (!enabledList.includes(currentDomain)) {
          enabledList.push(currentDomain);
        }
        statStatus.textContent = 'Ready';
        statStatus.classList.remove('disabled');
        btnTranslate.disabled = false;
        btnTranslate.style.opacity = '1';
      } else {
        enabledList = enabledList.filter(d => d !== currentDomain);
        statStatus.textContent = 'Disabled';
        statStatus.classList.add('disabled');
        btnTranslate.disabled = true;
        btnTranslate.style.opacity = '0.5';
      }

      chrome.storage.sync.set({ enabledDomains: enabledList }, () => {
        chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
          if (tab && tab.id) {
            chrome.tabs.sendMessage(tab.id, {
              action: 'setSiteEnabled',
              enabled: isEnabled
            }).catch(() => {});
          }
        });
      });
    });
  });

  [translator, llmOcrMode, targetLang, readingDirection, loadingStyle, apiUrl].forEach(el => {
    el.addEventListener('change', saveSettings);
  });

  btnTranslate.addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) return;

    btnTranslate.disabled = true;
    btnTranslate.style.opacity = '0.6';
    statStatus.textContent = 'Processing...';

    chrome.tabs.sendMessage(tab.id, { action: 'translateAllImages' }, () => {
      setTimeout(() => {
        btnTranslate.disabled = false;
        btnTranslate.style.opacity = '1';
      }, 1000);

      if (chrome.runtime.lastError) {
        statStatus.textContent = 'Refresh the page';
      } else {
        statStatus.textContent = 'Running...';
      }
    });
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'local' && changes.lastTranslationStats) {
      const stats = changes.lastTranslationStats.newValue;
      if (stats && stats.status && siteToggle.checked) {
        statStatus.textContent = stats.status;
      }
    }
  });

  function loadSiteStatus(domain) {
    chrome.storage.sync.get({ enabledDomains: [] }, (items) => {
      const enabledList = items.enabledDomains || [];
      const isEnabled = enabledList.includes(domain);
      siteToggle.checked = isEnabled;
      if (isEnabled) {
        statStatus.textContent = 'Ready';
        statStatus.classList.remove('disabled');
        btnTranslate.disabled = false;
        btnTranslate.style.opacity = '1';
      } else {
        statStatus.textContent = 'Disabled';
        statStatus.classList.add('disabled');
        btnTranslate.disabled = true;
        btnTranslate.style.opacity = '0.5';
      }
    });
  }

  function loadSettings() {
    chrome.storage.sync.get({
      apiUrl: 'http://127.0.0.1:8000',
      targetLang: 'id',
      readingDirection: 'rtl',
      translationMode: 'inpaint',
      fontScale: 1.0,
      allCaps: true,
      autoTranslate: true,
      translator: 'google',
      llmMergeOcr: true,
      loadingStyle: 'default'
    }, (items) => {
      apiUrl.value = items.apiUrl;
      targetLang.value = items.targetLang;
      readingDirection.value = items.readingDirection;
      translator.value = items.translator || 'google';
      llmOcrMode.value = items.llmMergeOcr === false ? 'individual' : 'combined';
      updateLlmOptions();
      loadingStyle.value = items.loadingStyle === 'minimal' ? 'minimal' : 'default';
    });
  }

  function saveSettings() {
    updateLlmOptions();
    const newSettings = {
      apiUrl: apiUrl.value.trim() || 'http://127.0.0.1:8000',
      targetLang: targetLang.value,
      readingDirection: readingDirection.value,
      translator: translator.value,
      llmMergeOcr: llmOcrMode.value === 'combined',
      loadingStyle: loadingStyle.value,
      translationMode: 'inpaint',
      autoTranslate: true
    };

    chrome.storage.sync.set(newSettings, () => {
      checkApiHealth();
    });
  }

  function checkApiHealth() {
    chrome.runtime.sendMessage({ action: 'checkHealth' }, (response) => {
      if (chrome.runtime.lastError || !response || !response.success) {
        serverStatusDot.classList.remove('online');
        serverStatusText.textContent = 'Offline';
      } else {
        serverStatusDot.classList.add('online');
        serverStatusText.textContent = 'Online';
      }
    });
  }

  function updateLlmOptions() {
    llmOcrModeRow.hidden = translator.value !== 'llm';
    llmOcrMode.disabled = translator.value !== 'llm';
    document.getElementById('llmOcrModeHint').textContent = llmOcrMode.value === 'combined'
      ? 'Understands all dialogue in this image together, then translates each bubble.'
      : 'Translates each bubble independently, without context from other bubbles.';
  }

  function loadStats() {
    chrome.storage.local.get(['lastTranslationStats'], (result) => {
      if (result.lastTranslationStats && result.lastTranslationStats.status && siteToggle.checked) {
        statStatus.textContent = result.lastTranslationStats.status;
      }
    });
  }

  window.addEventListener('unload', () => {
    if (healthCheckInterval) clearInterval(healthCheckInterval);
  });
});
