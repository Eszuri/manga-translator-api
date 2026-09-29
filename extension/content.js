class MangaTranslator {
  constructor() {
    this.processedImages = new Set();
    this.processingQueue = [];
    this.maxConcurrentJobs = 2;
    this.activeJobs = new Map();
    this.settings = null;
    this.totalProcessed = 0;
    this.totalImages = 0;
    this.isScanning = false;
    this.isEnabled = false;
    this.progressBar = null;
    this.progressBarHideTimer = null;
    this.contextMenuLoadingSources = new Map();
    this.imageLoadingCounts = new WeakMap();

    this.init();
  }

  async init() {
    await this.loadSettings();
    this.setupMessageListener();
    this.setupKeyboardShortcuts();

    const hostname = window.location.hostname;
    const enabledList = (this.settings && this.settings.enabledDomains) || [];
    if (!enabledList.includes(hostname)) {
      this.isEnabled = false;
      return;
    }

    this.isEnabled = true;
    this.scanAndProcess();
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
        loadingStyle: 'default',
        enabledDomains: []
      }, (items) => {
        this.settings = items;
        resolve(items);
      });
    });
  }

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
            this.processingQueue.forEach((img) => this.processedImages.delete(img));
            this.processingQueue = [];
            this.activeJobs.forEach(({ img }) => {
              this.finishImageLoading(null, img);
              delete img.dataset.mtJobId;
              this.processedImages.delete(img);
            });
            this.activeJobs.clear();
            this.hideProgressBar();
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
          this.startContextMenuLoading(request.srcUrl);
          sendResponse({ success: true });
          break;

        case 'contextMenuTranslateResult':
          if (request.result && request.result.success) {
            this.applyInpaintBySrc(request.srcUrl, request.result.data);
          }
          this.finishContextMenuLoading(request.srcUrl);
          sendResponse({ success: true });
          break;

        case 'contextMenuTranslateError':
          this.finishContextMenuLoading(request.srcUrl);
          sendResponse({ success: true });
          break;

        default:
          sendResponse({ success: false });
      }
      return true;
    });

    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === 'sync') {
        this.loadSettings().then(() => {
          if (changes.loadingStyle) this.syncLoadingStyle();
        });
      }
    });
  }

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

  async scanAndProcess() {
    if (!this.isEnabled) return;
    if (this.isScanning) return;
    this.isScanning = true;

    this.showProgressBar();
    if (this.progressBar) {
      const pText = this.progressBar.querySelector('.manga-translator-progress-text');
      if (pText) pText.textContent = '⏳ Menunggu semua gambar di halaman termuat...';
    }

    await this.waitForAllImagesToLoad();
    if (!this.isEnabled) {
      this.isScanning = false;
      return;
    }

    const allImgs = Array.from(document.querySelectorAll('img'));
    const mangaImgs = allImgs.filter(img => !this.processedImages.has(img) && this.isMangaImage(img));

    if (mangaImgs.length === 0 && this.processingQueue.length === 0 && this.activeJobs.size === 0) {
      this.isScanning = false;
      this.hideProgressBar();
      return;
    }

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
    this.pumpQueue();
  }

  async waitForAllImagesToLoad() {
    const maxWaitMs = 15000;
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitMs) {
      const allImgs = Array.from(document.querySelectorAll('img')).filter(
        img => !this.processedImages.has(img) && (img.src || img.dataset.src || img.dataset.lazySrc)
      );

      const pending = allImgs.filter(img => !img.complete || img.naturalWidth === 0);

      if (pending.length === 0 && allImgs.length > 0) {
        await new Promise(r => setTimeout(r, 600));
        const checkAgain = Array.from(document.querySelectorAll('img')).filter(
          img => !this.processedImages.has(img) && (img.src || img.dataset.src || img.dataset.lazySrc) && (!img.complete || img.naturalWidth === 0)
        );
        if (checkAgain.length === 0) {
          break;
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
    return img.naturalWidth > 300 && img.naturalHeight > 400;
  }

  pumpQueue() {
    if (!this.isEnabled) return;

    while (this.activeJobs.size < this.maxConcurrentJobs && this.processingQueue.length > 0) {
      this.startTranslationJob(this.processingQueue.shift());
    }

    if (this.processingQueue.length === 0 && this.activeJobs.size === 0) {
      this.hideProgressBar();
    }
  }

  createJobId() {
    if (crypto.randomUUID) return crypto.randomUUID();
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  async startTranslationJob(img) {
    const jobId = this.createJobId();
    const job = { id: jobId, img, originalSrc: img.currentSrc || img.src };
    this.activeJobs.set(jobId, job);
    img.dataset.mtJobId = jobId;

    this.showProgressBar();
    this.updateProgressBar();
    this.beginImageLoading(null, img);

    try {
      const result = await this.translateImage(job);
      const isCurrentJob = this.activeJobs.get(jobId) === job && img.dataset.mtJobId === jobId;
      if (!isCurrentJob) return;

      if (result.success && result.data) {
        img.src = result.data;
        this.totalProcessed++;
        chrome.storage.local.set({
          lastTranslationStats: {
            bubblesDetected: result.totalDetected || 0,
            processingTimeMs: result.durationMs || 0,
            status: `${this.totalProcessed}/${this.totalImages}`
          }
        });
      } else {
        console.warn('[MangaTranslator] Inpaint failed:', result.error);
      }
    } catch (e) {
      console.error('[MangaTranslator] Failed to translate image:', e);
    } finally {
      const ownsJob = this.activeJobs.get(jobId) === job;
      if (ownsJob) {
        this.activeJobs.delete(jobId);
        if (img.dataset.mtJobId === jobId) delete img.dataset.mtJobId;
        this.finishImageLoading(null, img);
      }
      this.updateProgressBar();
      this.pumpQueue();
    }
  }

  async translateImage(job) {
    const settings = this.settings || await this.loadSettings();
    const fileData = await this.getImageDataUrl(job.img);

    return this.sendMessage({
      action: 'inpaintPageStream',
      data: {
        fileData: fileData,
        imageSrc: job.originalSrc,
        jobId: job.id,
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
  }

  handlePipelineProgress(request) {
    if (this.isMinimalLoadingStyle()) return;

    const job = this.activeJobs.get(request.jobId);
    if (!job || job.img.dataset.mtJobId !== request.jobId) return;
    const { img } = job;

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

    if (this.progressBar) {
      const pText = this.progressBar.querySelector('.manga-translator-progress-text');
      if (pText) {
        const currentIdx = this.totalProcessed + 1;
        const total = this.totalImages || currentIdx;
        pText.textContent = `Halaman ${currentIdx}/${total} • ${label}`;
      }
    }
  }

  wrapImage(img) {
    if (!img.dataset.mtOriginalSrc) img.dataset.mtOriginalSrc = img.src;
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

    const minimalLoading = document.createElement('div');
    minimalLoading.className = 'manga-translator-minimal-loader';
    minimalLoading.setAttribute('role', 'status');
    minimalLoading.setAttribute('aria-label', 'Menerjemahkan manga');
    minimalLoading.innerHTML = `<img src="${chrome.runtime.getURL('icons/icon.svg')}" alt="">`;
    minimalLoading.style.display = 'none';
    wrapper.appendChild(minimalLoading);
  }

  showImageLoading(srcUrl, imgEl) {
    const img = imgEl || this.findImageBySrc(srcUrl);
    if (!img) return;
    let wrapper = img.closest('.manga-translator-wrapper');
    if (!wrapper) {
      this.wrapImage(img);
      wrapper = img.closest('.manga-translator-wrapper');
    }
    if (!wrapper) return;

    if (this.isMinimalLoadingStyle()) {
      const minimalLoading = wrapper.querySelector('.manga-translator-minimal-loader');
      if (minimalLoading) minimalLoading.style.display = 'flex';
      return;
    }

    const loading = wrapper.querySelector('.manga-translator-loading');
    if (loading) {
      const text = loading.querySelector('.manga-translator-loading-text');
      if (text) text.textContent = 'Deteksi bubble...';
      loading.style.display = 'flex';
    }
  }

  hideImageLoading(srcUrl, imgEl) {
    const img = imgEl || this.findImageBySrc(srcUrl);
    if (!img) return;
    const wrapper = img.closest('.manga-translator-wrapper');
    if (!wrapper) return;

    if (this.isMinimalLoadingStyle()) {
      const minimalLoading = wrapper.querySelector('.manga-translator-minimal-loader');
      if (minimalLoading) minimalLoading.style.display = 'none';
      return;
    }

    const loading = wrapper.querySelector('.manga-translator-loading');
    if (loading) loading.style.display = 'none';
  }

  findImageBySrc(srcUrl) {
    if (!srcUrl) return null;
    for (const img of this.processedImages) {
      if (img.src === srcUrl || img.dataset.mtOriginalSrc === srcUrl) return img;
    }
    return Array.from(document.images).find(
      img => img.src === srcUrl || img.dataset.mtOriginalSrc === srcUrl
    ) || null;
  }

  applyInpaintBySrc(srcUrl, dataUrl) {
    const img = this.findImageBySrc(srcUrl);
    if (img) {
      img.src = dataUrl;
    }
  }

  showProgressBar() {
    if (this.isMinimalLoadingStyle()) {
      this.hideDefaultProgressBar();
      return;
    }

    this.hideAllMinimalLoaders();
    if (this.progressBarHideTimer) {
      clearTimeout(this.progressBarHideTimer);
      this.progressBarHideTimer = null;
    }

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

  isMinimalLoadingStyle() {
    return this.settings && this.settings.loadingStyle === 'minimal';
  }

  hasActiveBatchLoading() {
    return this.isEnabled && (this.isScanning || this.activeJobs.size > 0 || this.processingQueue.length > 0);
  }

  hasActiveLoading() {
    return this.hasActiveBatchLoading() || this.contextMenuLoadingSources.size > 0;
  }

  beginImageLoading(srcUrl, imgEl) {
    const img = imgEl || this.findImageBySrc(srcUrl);
    if (!img) return null;

    const activeCount = this.imageLoadingCounts.get(img) || 0;
    this.imageLoadingCounts.set(img, activeCount + 1);
    this.showImageLoading(null, img);
    return img;
  }

  finishImageLoading(srcUrl, imgEl) {
    const img = imgEl || this.findImageBySrc(srcUrl);
    if (!img) return null;

    const activeCount = this.imageLoadingCounts.get(img) || 0;
    if (activeCount <= 1) {
      this.imageLoadingCounts.delete(img);
      this.hideImageLoading(null, img);
    } else {
      this.imageLoadingCounts.set(img, activeCount - 1);
    }
    return img;
  }

  startContextMenuLoading(srcUrl) {
    const activeLoading = this.contextMenuLoadingSources.get(srcUrl);
    const img = activeLoading?.img || this.findImageBySrc(srcUrl);
    if (activeLoading) {
      activeLoading.count += 1;
    } else {
      this.contextMenuLoadingSources.set(srcUrl, { count: 1, img });
    }
    this.beginImageLoading(null, img);
  }

  finishContextMenuLoading(srcUrl) {
    const activeLoading = this.contextMenuLoadingSources.get(srcUrl);
    if (!activeLoading) return;

    if (activeLoading.count > 1) {
      activeLoading.count -= 1;
    } else {
      this.contextMenuLoadingSources.delete(srcUrl);
    }
    this.finishImageLoading(null, activeLoading.img);
  }

  hideAllMinimalLoaders() {
    document.querySelectorAll('.manga-translator-minimal-loader').forEach((loading) => {
      loading.style.display = 'none';
    });
  }

  hideDefaultProgressBar() {
    if (this.progressBarHideTimer) {
      clearTimeout(this.progressBarHideTimer);
      this.progressBarHideTimer = null;
    }
    if (this.progressBar) this.progressBar.style.display = 'none';
  }

  syncLoadingStyle() {
    if (this.isMinimalLoadingStyle()) {
      this.hideDefaultProgressBar();
      document.querySelectorAll('.manga-translator-loading').forEach((loading) => {
        loading.style.display = 'none';
      });
      this.hideAllMinimalLoaders();
      this.activeJobs.forEach(({ img }) => this.showImageLoading(null, img));
      this.contextMenuLoadingSources.forEach(({ img }, srcUrl) => this.showImageLoading(srcUrl, img));
      return;
    }

    this.hideAllMinimalLoaders();
    if (!this.hasActiveLoading()) return;

    if (this.hasActiveBatchLoading()) {
      this.showProgressBar();
      if (this.isScanning && this.activeJobs.size === 0 && this.processingQueue.length === 0) {
        const text = this.progressBar.querySelector('.manga-translator-progress-text');
        if (text) text.textContent = '⏳ Menunggu semua gambar di halaman termuat...';
      } else {
        this.updateProgressBar();
      }
    }

    this.activeJobs.forEach(({ img }) => this.showImageLoading(null, img));
    this.contextMenuLoadingSources.forEach(({ img }, srcUrl) => this.showImageLoading(srcUrl, img));
  }

  updateProgressBar() {
    if (!this.progressBar) return;
    const text = this.progressBar.querySelector('.manga-translator-progress-text');
    const fill = this.progressBar.querySelector('.manga-translator-progress-fill');

    const remaining = this.processingQueue.length;
    const total = this.totalImages || this.totalProcessed + remaining + this.activeJobs.size;
    const pct = total === 0 ? 0 : (this.totalProcessed / total) * 100;

    text.textContent = `Translated ${this.totalProcessed}/${total} pages`;
    fill.style.width = `${pct}%`;
  }

  hideProgressBar() {
    if (this.isMinimalLoadingStyle()) {
      this.hideDefaultProgressBar();
      return;
    }

    this.hideAllMinimalLoaders();
    if (!this.progressBar) return;

    const text = this.progressBar.querySelector('.manga-translator-progress-text');
    const fill = this.progressBar.querySelector('.manga-translator-progress-fill');
    text.textContent = `✅ Done! ${this.totalProcessed} pages translated`;
    fill.style.width = '100%';

    if (this.progressBarHideTimer) clearTimeout(this.progressBarHideTimer);
    this.progressBarHideTimer = setTimeout(() => {
      if (this.progressBar) this.progressBar.style.display = 'none';
      this.progressBarHideTimer = null;
    }, 3000);
  }

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

  async getImageDataUrl(img) {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      return canvas.toDataURL('image/jpeg', 0.9);
    } catch (e) {
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

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => new MangaTranslator());
} else {
  new MangaTranslator();
}
