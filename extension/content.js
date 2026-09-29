// content.js — Manga Translator Content Script
// Auto-process: detects manga images and translates them one-by-one using inpainting
// Each image is replaced immediately as its translation completes

class MangaTranslator {
  constructor() {
    this.processedImages = new Set();   // Track already-processed <img> elements
    this.processingQueue = [];          // Images waiting to be processed
    this.isProcessing = false;          // Lock: only one image at a time
    this.settings = null;
    this.totalProcessed = 0;
    this.totalImages = 0;
    this.isScanning = false;
    this.isEnabled = false;

    // Progress bar element
    this.progressBar = null;

    this.init();
  }

  async init() {
    await this.loadSettings();
    this.setupMessageListener();
    this.setupKeyboardShortcuts();

    // Default is NONAKTIF: only process if site is in enabledDomains
    const hostname = window.location.hostname;
    const enabledList = (this.settings && this.settings.enabledDomains) || [];
    if (!enabledList.includes(hostname)) {
      this.isEnabled = false;
      return;
    }

    this.isEnabled = true;

    // Auto-start: find all manga images and process them
    this.scanAndProcess();

    // Watch for dynamically loaded images (infinite scroll readers)
    this.setupMutationObserver();
  }

  async loadSettings() {
    return new Promise((resolve) => {
      chrome.storage.sync.get({
        apiUrl: 'http://127.0.0.1:8000',
        targetLang: 'id',
        translator: 'google',
        detectorType: 'hybrid',
        readingDirection: 'rtl',
        device: 'auto',
        fontScale: 1.0,
        allCaps: true,
        enabledDomains: []
      }, (items) => {
        this.settings = items;
        resolve(items);
      });
    });
  }

  // ─── Message Listener (from popup / background) ────────────────────
  setupMessageListener() {
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      switch (request.action) {
        case 'translateAllImages':
          this.scanAndProcess();
          sendResponse({ success: true, status: 'started' });
          break;

        case 'setSiteEnabled':
          this.isEnabled = request.enabled;
          if (this.isEnabled) {
            this.scanAndProcess();
          } else {
            this.hideProgressBar();
            this.isProcessing = false;
            this.processingQueue = [];
          }
          sendResponse({ success: true });
          break;

        case 'pipelineProgress':
          this.handlePipelineProgress(request);
          sendResponse({ success: true });
          break;

        case 'getPageStats':
          sendResponse({
            totalImages: this.totalImages,
            translatedCount: this.totalProcessed,
            mode: 'inpaint'
          });
          break;

        case 'toggleOverlays':
        case 'clearTranslations':
          sendResponse({ success: true });
          break;

        case 'contextMenuTranslateStart':
          // Show loading on specific image
          this.showImageLoading(request.srcUrl);
          sendResponse({ success: true });
          break;

        case 'contextMenuTranslateResult':
          if (request.result && request.result.success) {
            this.applyInpaintBySrc(request.srcUrl, request.result.data);
          }
          sendResponse({ success: true });
          break;

        case 'contextMenuTranslateError':
          this.hideImageLoading(request.srcUrl);
          sendResponse({ success: true });
          break;

        default:
          sendResponse({ success: false });
      }
      return true;
    });

    // Reload settings when changed from popup
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === 'sync') this.loadSettings();
    });
  }

  // ─── Keyboard Shortcuts ────────────────────────────────────────────
  setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      if (e.altKey && e.key.toLowerCase() === 't') {
        e.preventDefault();
        this.scanAndProcess();
      } else if (e.altKey && e.key.toLowerCase() === 'c') {
        e.preventDefault();
        this.restoreAllOriginals();
      }
    });
  }

  // ─── Scan & Auto-Process (Waits for all images to load first) ───────
  async scanAndProcess() {
    if (!this.isEnabled) return;
    if (this.isScanning) return;
    this.isScanning = true;

    // Show initial waiting indicator in progress bar
    this.showProgressBar();
    if (this.progressBar) {
      const pText = this.progressBar.querySelector('.manga-translator-progress-text');
      if (pText) pText.textContent = '⏳ Menunggu semua gambar di halaman termuat...';
    }

    // Wait until all images currently on the page are completely loaded
    await this.waitForAllImagesToLoad();

    // Now all loaded images have natural dimensions available
    const allImgs = Array.from(document.querySelectorAll('img'));
    const mangaImgs = allImgs.filter(img => !this.processedImages.has(img) && this.isMangaImage(img));

    if (mangaImgs.length === 0 && this.processingQueue.length === 0 && !this.isProcessing) {
      this.hideProgressBar();
      this.isScanning = false;
      return;
    }

    // Wrap and queue all confirmed manga images
    mangaImgs.forEach(img => {
      this.processedImages.add(img);
      this.wrapImage(img);
      this.processingQueue.push(img);
    });

    this.totalImages = this.processedImages.size;

    if (this.progressBar) {
      const pText = this.progressBar.querySelector('.manga-translator-progress-text');
      if (pText) pText.textContent = `Semua gambar termuat (${this.totalImages} halaman) • Memulai...`;
    }

    this.isScanning = false;

    // Start sequential processing (one-by-one)
    this.processNext();
  }

  async waitForAllImagesToLoad() {
    const maxWaitMs = 15000; // max wait 15s fallback so broken ad images don't block
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitMs) {
      const allImgs = Array.from(document.querySelectorAll('img')).filter(
        img => !this.processedImages.has(img) && (img.src || img.dataset.src || img.dataset.lazySrc)
      );

      const pending = allImgs.filter(img => !img.complete || img.naturalWidth === 0);

      if (pending.length === 0 && allImgs.length > 0) {
        // Wait brief 600ms stabilization window
        await new Promise(r => setTimeout(r, 600));
        const checkAgain = Array.from(document.querySelectorAll('img')).filter(
          img => !this.processedImages.has(img) && (img.src || img.dataset.src || img.dataset.lazySrc) && (!img.complete || img.naturalWidth === 0)
        );
        if (checkAgain.length === 0) {
          break; // All images loaded!
        }
      }

      if (this.progressBar) {
        const pText = this.progressBar.querySelector('.manga-translator-progress-text');
        if (pText) {
          pText.textContent = `⏳ Menunggu gambar termuat (${pending.length} tersisa)...`;
        }
      }

      await Promise.race([
        Promise.all(pending.map(img => new Promise(resolve => {
          if (img.complete && img.naturalWidth > 0) return resolve();
          const onDone = () => {
            img.removeEventListener('load', onDone);
            img.removeEventListener('error', onDone);
            resolve();
          };
          img.addEventListener('load', onDone);
          img.addEventListener('error', onDone);
        }))),
        new Promise(r => setTimeout(r, 800))
      ]);
    }
  }

  isMangaImage(img) {
    // Must be large enough to be a manga page (not thumbnail/icon/avatar)
    return img.naturalWidth > 300 && img.naturalHeight > 400;
  }

  // ─── Sequential Processing (one-by-one) ────────────────────────────
  async processNext() {
    if (this.isProcessing) return;         // Already processing one
    if (this.processingQueue.length === 0) {
      this.hideProgressBar();
      return;
    }

    this.isProcessing = true;
    const img = this.processingQueue.shift();

    this.showProgressBar();
    this.updateProgressBar();
    this.showImageLoading(null, img);

    try {
      await this.translateImage(img);
      this.totalProcessed++;
    } catch (e) {
      console.error('[MangaTranslator] Failed to translate image:', e);
    }

    this.updateProgressBar();
    this.hideImageLoading(null, img);
    this.isProcessing = false;

    // Process next image in queue
    this.processNext();
  }

  // ─── Translate Single Image (Inpaint Stream) ───────────────────────
  async translateImage(img) {
    const settings = this.settings || await this.loadSettings();
    const fileData = await this.getImageDataUrl(img);
    this.currentProcessingImg = img;

    const result = await this.sendMessage({
      action: 'inpaintPageStream',
      data: {
        fileData: fileData,
        imageSrc: img.src,
        mimeType: 'image/jpeg',
        target_lang: settings.targetLang,
        translator: settings.translator || 'llm',
        detector_type: settings.detectorType,
        reading_direction: settings.readingDirection,
        device: settings.device,
        typeset: 'true',
        font_scale: String(settings.fontScale),
        all_caps: String(settings.allCaps)
      }
    });

    if (result.success && result.data) {
      // Replace image immediately
      img.src = result.data;

      // Save stats
      chrome.storage.local.set({
        lastTranslationStats: {
          bubblesDetected: result.totalDetected || this.totalProcessed,
          processingTimeMs: result.durationMs || 0,
          status: `${this.totalProcessed + 1}/${this.totalImages}`
        }
      });
    } else {
      console.warn('[MangaTranslator] Inpaint failed:', result.error);
    }
    this.currentProcessingImg = null;
  }

  // ─── Real-Time Pipeline Progress Handler ───────────────────────────
  handlePipelineProgress(request) {
    const img = this.currentProcessingImg || this.findImageBySrc(request.imageSrc);
    if (!img) return;

    const wrapper = img.closest('.manga-translator-wrapper');
    if (!wrapper) return;

    const loading = wrapper.querySelector('.manga-translator-loading');
    if (!loading) return;

    const text = loading.querySelector('.manga-translator-loading-text');

    const stageNames = {
      detect: 'Deteksi bubble...',
      ocr: 'OCR...',
      translate: 'Translate...',
      inpaint: 'Inpainting...',
      render: 'Render...'
    };

    const label = stageNames[request.stage] || 'Translate...';
    if (text) text.textContent = label;

    // Update top progress bar
    if (this.progressBar) {
      const pText = this.progressBar.querySelector('.manga-translator-progress-text');
      if (pText) {
        const currentIdx = this.totalProcessed + 1;
        const total = this.totalImages || currentIdx;
        pText.textContent = `Halaman ${currentIdx}/${total} • ${label}`;
      }
    }
  }

  // ─── Image Wrapper ─────────────────────────────────────────────────
  wrapImage(img) {
    if (img.closest('.manga-translator-wrapper')) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'manga-translator-wrapper';
    img.parentNode.insertBefore(wrapper, img);
    wrapper.appendChild(img);

    const loading = document.createElement('div');
    loading.className = 'manga-translator-loading';
    loading.innerHTML = `
      <div class="manga-translator-loading-spinner"></div>
      <div class="manga-translator-loading-text">Deteksi bubble...</div>
    `;
    loading.style.display = 'none';
    wrapper.appendChild(loading);
  }

  // ─── Loading State Per-Image ───────────────────────────────────────
  showImageLoading(srcUrl, imgEl) {
    const img = imgEl || this.findImageBySrc(srcUrl);
    if (!img) return;
    const wrapper = img.closest('.manga-translator-wrapper');
    if (wrapper) {
      const loading = wrapper.querySelector('.manga-translator-loading');
      if (loading) {
        const text = loading.querySelector('.manga-translator-loading-text');
        if (text) text.textContent = 'Deteksi bubble...';
        loading.style.display = 'flex';
      }
    }
  }

  hideImageLoading(srcUrl, imgEl) {
    const img = imgEl || this.findImageBySrc(srcUrl);
    if (!img) return;
    const wrapper = img.closest('.manga-translator-wrapper');
    if (wrapper) {
      const loading = wrapper.querySelector('.manga-translator-loading');
      if (loading) loading.style.display = 'none';
    }
  }

  findImageBySrc(srcUrl) {
    if (!srcUrl) return null;
    for (const img of this.processedImages) {
      if (img.src === srcUrl || img.dataset.mtOriginalSrc === srcUrl) return img;
    }
    return null;
  }

  // ─── Restore Originals ─────────────────────────────────────────────
  applyInpaintBySrc(srcUrl, dataUrl) {
    const img = this.findImageBySrc(srcUrl);
    if (img) {
      img.src = dataUrl;
      this.hideImageLoading(srcUrl);
    }
  }

  // ─── Progress Bar ──────────────────────────────────────────────────
  showProgressBar() {
    if (!this.progressBar) {
      this.progressBar = document.createElement('div');
      this.progressBar.className = 'manga-translator-progress-bar';
      this.progressBar.innerHTML = `
        <div class="manga-translator-progress-fill"></div>
        <div class="manga-translator-progress-text">Translating...</div>
      `;
      document.body.appendChild(this.progressBar);
    }
    this.progressBar.style.display = 'flex';
  }

  updateProgressBar() {
    if (!this.progressBar) return;
    const text = this.progressBar.querySelector('.manga-translator-progress-text');
    const fill = this.progressBar.querySelector('.manga-translator-progress-fill');

    const remaining = this.processingQueue.length;
    const total = this.totalProcessed + remaining + (this.isProcessing ? 1 : 0);
    const pct = total === 0 ? 0 : (this.totalProcessed / total) * 100;

    text.textContent = `Translated ${this.totalProcessed}/${total} pages`;
    fill.style.width = `${pct}%`;
  }

  hideProgressBar() {
    if (this.progressBar) {
      // Show completion briefly
      const text = this.progressBar.querySelector('.manga-translator-progress-text');
      const fill = this.progressBar.querySelector('.manga-translator-progress-fill');
      text.textContent = `✅ Done! ${this.totalProcessed} pages translated`;
      fill.style.width = '100%';

      setTimeout(() => {
        if (this.progressBar) this.progressBar.style.display = 'none';
      }, 3000);
    }
  }

  // ─── Mutation Observer (infinite scroll) ───────────────────────────
  setupMutationObserver() {
    let debounceTimer = null;
    const observer = new MutationObserver((mutations) => {
      if (!this.isEnabled) return;
      let hasNewImages = false;
      for (const mutation of mutations) {
        for (const node of mutation.addedNodes) {
          if (node.nodeName === 'IMG' || (node.querySelectorAll && node.querySelectorAll('img').length > 0)) {
            hasNewImages = true;
            break;
          }
        }
        if (hasNewImages) break;
      }
      if (hasNewImages) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => this.scanAndProcess(), 1000);
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  // ─── Helpers ───────────────────────────────────────────────────────
  async getImageDataUrl(img) {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      return canvas.toDataURL('image/jpeg', 0.9);
    } catch (e) {
      // CORS tainted — fetch via background service worker
      const result = await this.sendMessage({ action: 'fetchImage', url: img.src });
      if (result.success) return result.data;
      throw new Error('Cannot access image: ' + e.message);
    }
  }

  sendMessage(msg) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage(msg, (response) => {
        if (chrome.runtime.lastError) {
          resolve({ success: false, error: chrome.runtime.lastError.message });
        } else {
          resolve(response || { success: false, error: 'No response' });
        }
      });
    });
  }
}

// ─── Initialize on page ready ──────────────────────────────────────
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => new MangaTranslator());
} else {
  new MangaTranslator();
}
