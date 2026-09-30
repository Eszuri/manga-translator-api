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
    this.scanDebounceTimer = null;
    this.viewportPriorityTimer = null;

    this.init();
  }

  async init() {
    await this.loadSettings();
    if (window.MangaTranslationCache) {
      window.MangaTranslationCache.pruneOldCache(7).catch(() => {});
    }
    this.setupMessageListener();
    this.setupKeyboardShortcuts();
    this.setupViewportPriority();

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
        device: 'gpu',
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
          (async () => {
            if (window.MangaTranslationCache) {
              await window.MangaTranslationCache.clearPageCache(window.location.href);
            }
            this.restoreAllOriginals();
            this.scanAndProcess({ bypassCache: true });
          })();
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

  async scanAndProcess(options = {}) {
    if (!this.isEnabled) return;
    if (this.isScanning) return;
    this.isScanning = true;

    this.showProgressBar();
    const settings = this.settings || await this.loadSettings();
    let restoredFromCache = 0;

    if (this.progressBar) {
      const pText = this.progressBar.querySelector('.manga-translator-progress-text');
      if (pText) pText.textContent = '⚡ Checking image cache...';
    }

    if (!options.bypassCache && window.MangaTranslationCache) {
      restoredFromCache += await this.restoreCachedImages(settings, this.getPageImages());
    }

    if (!this.isEnabled) {
      this.isScanning = false;
      return;
    }

    if (this.progressBar) {
      const pText = this.progressBar.querySelector('.manga-translator-progress-text');
      if (pText) pText.textContent = '⏳ Waiting for images that are not cached...';
    }

    await this.waitForAllImagesToLoad();
    if (!this.isEnabled) {
      this.isScanning = false;
      return;
    }

    let mangaImgs = this.getPageImages().filter(
      img => !this.processedImages.has(img) && this.isMangaImage(img)
    );

    if (!options.bypassCache && window.MangaTranslationCache && mangaImgs.length > 0) {
      restoredFromCache += await this.restoreCachedImages(settings, mangaImgs);
      mangaImgs = mangaImgs.filter(img => !this.processedImages.has(img));
    }

    if (mangaImgs.length === 0 && this.processingQueue.length === 0 && this.activeJobs.size === 0) {
      this.totalImages = this.processedImages.size;
      this.isScanning = false;
      this.updateProgressBar();
      this.hideProgressBar();
      return;
    }

    const prioritizedImages = this.prioritizeImagesByViewport(mangaImgs);
    for (const img of prioritizedImages) {
      const descriptor = this.getImageCacheDescriptor(img, settings);
      const originalSrc = descriptor.originalSrc;
      if (!img.dataset.mtOriginalSrc) {
        img.dataset.mtOriginalSrc = originalSrc;
      }

      img.dataset.mtCacheKey = descriptor.cacheKey;

      this.processedImages.add(img);
      this.wrapImage(img);
      this.processingQueue.push(img);
    }

    this.totalImages = this.processedImages.size;

    if (this.progressBar) {
      const pText = this.progressBar.querySelector('.manga-translator-progress-text');
      if (pText) {
        if (this.processingQueue.length === 0 && this.activeJobs.size === 0) {
          pText.textContent = `All images restored from cache (${this.totalProcessed} pages) ✅`;
        } else if (restoredFromCache > 0) {
          pText.textContent = `${restoredFromCache} restored from cache • Processing remaining (${this.processingQueue.length} pages)...`;
        } else {
          pText.textContent = `All images loaded (${this.totalImages} pages) • Starting...`;
        }
      }
    }

    this.updateProgressBar();
    this.isScanning = false;
    this.pumpQueue();
  }

  async waitForAllImagesToLoad() {
    const maxWaitMs = 15000;
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitMs) {
      const allImgs = this.getPageImages().filter(
        img => !this.processedImages.has(img) && (img.src || img.dataset.src || img.dataset.lazySrc)
      );
      if (allImgs.length === 0) break;

      const pending = allImgs.filter(img => !img.complete);

      if (pending.length === 0) {
        await new Promise(r => setTimeout(r, 600));
        const checkAgain = this.getPageImages().filter(
          img => !this.processedImages.has(img) && (img.src || img.dataset.src || img.dataset.lazySrc) && !img.complete
        );
        if (checkAgain.length === 0) {
          break;
        }
      }

      if (this.progressBar) {
        const pText = this.progressBar.querySelector('.manga-translator-progress-text');
        if (pText) {
          pText.textContent = `⏳ Waiting for images to load (${pending.length} remaining)...`;
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
    return img.complete && this.isMangaDimensions(img.naturalWidth, img.naturalHeight);
  }

  getMangaDimensionThreshold() {
    return { minWidth: 500, minHeight: 700 };
  }

  isMangaDimensions(width, height) {
    const { minWidth, minHeight } = this.getMangaDimensionThreshold();
    return Number(width) >= minWidth && Number(height) >= minHeight;
  }

  getPageImages() {
    return Array.from(document.querySelectorAll('img')).filter(
      img => !img.closest('.manga-translator-minimal-loader')
    );
  }

  getCacheSource(img) {
    return img.dataset.mtOriginalSrc ||
      img.dataset.src ||
      img.dataset.lazySrc ||
      img.dataset.original ||
      img.currentSrc ||
      img.src ||
      '';
  }

  getImageCacheDescriptor(img, settings, knownIndex = null) {
    const originalSrc = this.getCacheSource(img);
    const legacySource = this.getCurrentImageSource(img) || originalSrc;
    const imageIndex = knownIndex === null ? this.getPageImages().indexOf(img) : knownIndex;
    const cache = window.MangaTranslationCache;

    return {
      img,
      originalSrc,
      imageIndex,
      cacheKey: cache ? cache.buildCacheKey(
        settings,
        originalSrc,
        window.location.href,
        imageIndex
      ) : '',
      legacyCacheKey: cache && cache.buildLegacyCacheKey
        ? cache.buildLegacyCacheKey(settings, legacySource)
        : ''
    };
  }

  async restoreCachedImages(settings, candidateImages) {
    const cache = window.MangaTranslationCache;
    if (!cache) return 0;

    const candidates = new Set(candidateImages || []);
    const descriptors = this.getPageImages()
      .map((img, imageIndex) => this.getImageCacheDescriptor(img, settings, imageIndex))
      .filter(({ img, originalSrc }) =>
        candidates.has(img) && !this.processedImages.has(img) && Boolean(originalSrc)
      );

    if (descriptors.length === 0) return 0;

    const keys = descriptors.flatMap(({ cacheKey, legacyCacheKey }) =>
      legacyCacheKey ? [cacheKey, legacyCacheKey] : [cacheKey]
    );
    const cachedEntries = cache.getCachedTranslations
      ? await cache.getCachedTranslations(keys)
      : new Map(await Promise.all(keys.map(async key => [key, await cache.getCachedTranslation(key)])));

    let restored = 0;
    for (const descriptor of descriptors) {
      const currentEntry = cachedEntries.get(descriptor.cacheKey);
      const legacyEntry = cachedEntries.get(descriptor.legacyCacheKey);
      const cached = currentEntry || legacyEntry;
      const originalWidth = cached && cached.originalWidth || descriptor.img.naturalWidth;
      const originalHeight = cached && cached.originalHeight || descriptor.img.naturalHeight;
      if (!this.isMangaDimensions(originalWidth, originalHeight)) continue;

      const imageUrl = cache.createImageUrl(cached);
      if (!imageUrl) continue;

      if (!descriptor.img.dataset.mtOriginalSrc) {
        descriptor.img.dataset.mtOriginalSrc = descriptor.originalSrc;
      }
      if (descriptor.img.dataset.mtCacheObjectUrl) {
        cache.revokeImageUrl(descriptor.img.dataset.mtCacheObjectUrl);
      }

      descriptor.img.dataset.mtCacheKey = descriptor.cacheKey;
      descriptor.img.src = imageUrl;
      if (imageUrl.startsWith('blob:')) {
        descriptor.img.dataset.mtCacheObjectUrl = imageUrl;
      } else {
        delete descriptor.img.dataset.mtCacheObjectUrl;
      }

      this.processedImages.add(descriptor.img);
      this.wrapImage(descriptor.img);
      this.totalProcessed++;
      restored++;

      if (!currentEntry && legacyEntry) {
        cache.saveCachedTranslation({
          cacheKey: descriptor.cacheKey,
          pageUrl: window.location.href,
          originalSrc: descriptor.originalSrc,
          translatedBlob: legacyEntry.translatedBlob,
          translatedData: legacyEntry.translatedData,
          originalWidth,
          originalHeight,
          targetLang: settings.targetLang,
          translator: settings.translator,
          readingDirection: settings.readingDirection,
          timestamp: Date.now()
        }).catch(() => {});
      }
    }

    return restored;
  }

  getCurrentImageSource(img) {
    return img.currentSrc || img.src || img.dataset.src || img.dataset.lazySrc || '';
  }

  scheduleScan(delayMs = 300) {
    clearTimeout(this.scanDebounceTimer);
    this.scanDebounceTimer = setTimeout(() => {
      this.scanDebounceTimer = null;
      if (this.isEnabled) this.scanAndProcess();
    }, delayMs);
  }

  isImageInViewport(img) {
    const rect = img.getBoundingClientRect();
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth;
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight;

    return rect.width > 0 &&
      rect.height > 0 &&
      rect.bottom > 0 &&
      rect.right > 0 &&
      rect.top < viewportHeight &&
      rect.left < viewportWidth;
  }

  prioritizeImagesByViewport(images) {
    const visible = [];
    const outsideViewport = [];

    for (const img of images) {
      if (this.isImageInViewport(img)) {
        visible.push(img);
      } else {
        outsideViewport.push(img);
      }
    }

    return [...visible, ...outsideViewport];
  }

  prioritizeProcessingQueue() {
    if (this.processingQueue.length < 2) return;
    this.processingQueue = this.prioritizeImagesByViewport(this.processingQueue);
  }

  hasVisibleActiveJob() {
    return Array.from(this.activeJobs.values()).some(({ img }) => this.isImageInViewport(img));
  }

  setupViewportPriority() {
    const refreshPriority = () => {
      if (!this.isEnabled || this.processingQueue.length < 2 || this.viewportPriorityTimer) return;

      this.viewportPriorityTimer = setTimeout(() => {
        this.viewportPriorityTimer = null;
        this.prioritizeProcessingQueue();
        this.pumpQueue();
      }, 100);
    };

    window.addEventListener('scroll', refreshPriority, { passive: true });
    window.addEventListener('resize', refreshPriority, { passive: true });
  }

  pumpQueue() {
    if (!this.isEnabled) return;
    this.prioritizeProcessingQueue();

    while (this.activeJobs.size < this.maxConcurrentJobs && this.processingQueue.length > 0) {
      this.prioritizeProcessingQueue();
      const nextImage = this.processingQueue[0];
      if (!this.isImageInViewport(nextImage) && this.hasVisibleActiveJob()) break;

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
    const settings = this.settings || await this.loadSettings();
    const descriptor = this.getImageCacheDescriptor(img, settings);
    const job = {
      id: jobId,
      img,
      originalSrc: this.getCurrentImageSource(img),
      originalWidth: img.naturalWidth,
      originalHeight: img.naturalHeight,
      cacheKey: img.dataset.mtCacheKey || descriptor.cacheKey
    };
    this.activeJobs.set(jobId, job);
    img.dataset.mtJobId = jobId;

    this.showProgressBar();
    this.updateProgressBar();
    this.beginImageLoading(null, img);

    try {
      const result = await this.translateImage(job);
      const isCurrentJob = this.activeJobs.get(jobId) === job && img.dataset.mtJobId === jobId;
      if (!isCurrentJob) return;

      if (this.getCurrentImageSource(img) !== job.originalSrc) {
        this.processedImages.delete(img);
        delete img.dataset.mtCacheKey;
        this.scheduleScan();
        console.warn('[MangaTranslator] Image source changed while translating; stale result ignored.');
        return;
      }

      if (result.success && result.data) {
        if (img.dataset.mtCacheObjectUrl && window.MangaTranslationCache) {
          window.MangaTranslationCache.revokeImageUrl(img.dataset.mtCacheObjectUrl);
          delete img.dataset.mtCacheObjectUrl;
        }
        img.src = result.data;
        this.totalProcessed++;

        if (window.MangaTranslationCache) {
          window.MangaTranslationCache.saveCachedTranslation({
            cacheKey: job.cacheKey,
            pageUrl: window.location.href,
            originalSrc: job.originalSrc,
            translatedData: result.data,
            originalWidth: job.originalWidth,
            originalHeight: job.originalHeight,
            targetLang: settings.targetLang,
            translator: settings.translator,
            readingDirection: settings.readingDirection,
            timestamp: Date.now()
          }).catch(err => console.warn('[MangaTranslator] Failed to cache translation:', err));
        }

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
      if (e && (e.code === 'IMAGE_NOT_READY' || e.code === 'IMAGE_SOURCE_CHANGED')) {
        this.processedImages.delete(img);
        delete img.dataset.mtCacheKey;
        this.scheduleScan();
      }
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
    const fileData = await this.getImageDataUrl(job.img, job.originalSrc);

    return this.sendMessage({
      action: 'inpaintPageStream',
      data: {
        fileData: fileData,
        imageSrc: job.originalSrc,
        jobId: job.id,
        mimeType: 'image/jpeg',
        target_lang: settings.targetLang,
        translator: settings.translator || 'llm',
        detector_type: settings.detectorType === 'comic_text_detector'
          ? 'comic_text_detector'
          : 'hybrid',
        reading_direction: settings.readingDirection,
        device: 'gpu',
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
      detect: 'Detecting bubbles...',
      ocr: 'OCR...',
      translate: 'Translating...',
      inpaint: 'Inpainting...',
      render: 'Rendering...'
    };

    const label = stageNames[request.stage] || 'Translate...';
    if (text) text.textContent = label;

    if (this.progressBar) {
      const pText = this.progressBar.querySelector('.manga-translator-progress-text');
      if (pText) {
        const currentIdx = this.totalProcessed + 1;
        const total = this.totalImages || currentIdx;
        pText.textContent = `Page ${currentIdx}/${total} • ${label}`;
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
      <div class="manga-translator-loading-text">Detecting bubbles...</div>
    `;
    loading.style.display = 'none';
    wrapper.appendChild(loading);

    const minimalLoading = document.createElement('div');
    minimalLoading.className = 'manga-translator-minimal-loader';
    minimalLoading.setAttribute('role', 'status');
    minimalLoading.setAttribute('aria-label', 'Translating manga');
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
      if (text) text.textContent = 'Detecting bubbles...';
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
      const originalWidth = img.naturalWidth;
      const originalHeight = img.naturalHeight;
      if (!img.dataset.mtOriginalSrc) img.dataset.mtOriginalSrc = srcUrl;
      if (img.dataset.mtCacheObjectUrl && window.MangaTranslationCache) {
        window.MangaTranslationCache.revokeImageUrl(img.dataset.mtCacheObjectUrl);
        delete img.dataset.mtCacheObjectUrl;
      }
      img.src = dataUrl;
      if (window.MangaTranslationCache) {
        const descriptor = this.getImageCacheDescriptor(img, this.settings);
        window.MangaTranslationCache.saveCachedTranslation({
          cacheKey: descriptor.cacheKey,
          pageUrl: window.location.href,
          originalSrc: srcUrl,
          translatedData: dataUrl,
          originalWidth,
          originalHeight,
          targetLang: (this.settings && this.settings.targetLang) || 'id',
          translator: (this.settings && this.settings.translator) || 'google',
          readingDirection: (this.settings && this.settings.readingDirection) || 'rtl',
          timestamp: Date.now()
        }).catch(() => {});
      }
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
        if (text) text.textContent = '⏳ Waiting for all page images to load...';
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
    const observer = new MutationObserver((mutations) => {
      if (!this.isEnabled) return;
      let hasNewImages = false;
      for (const mutation of mutations) {
        if (mutation.type === 'attributes' && mutation.target.nodeName === 'IMG') {
          hasNewImages = true;
          break;
        }
        for (const node of mutation.addedNodes) {
          if (node.nodeName === 'IMG' || (node.querySelectorAll && node.querySelectorAll('img').length > 0)) {
            hasNewImages = true;
            break;
          }
        }
        if (hasNewImages) break;
      }
      if (hasNewImages) {
        this.scheduleScan(1000);
      }
    });
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['src', 'srcset']
    });

    document.addEventListener('load', (event) => {
      const img = event.target;
      if (
        this.isEnabled &&
        img instanceof HTMLImageElement &&
        !img.closest('.manga-translator-minimal-loader') &&
        !this.processedImages.has(img)
      ) {
        this.scheduleScan();
      }
    }, true);
  }

  createImageReadinessError(message, code) {
    const error = new Error(message);
    error.code = code;
    return error;
  }

  async waitForImageReady(img, expectedSrc, timeoutMs = 20000) {
    const initialSrc = this.getCurrentImageSource(img);
    if (!initialSrc || (expectedSrc && initialSrc !== expectedSrc)) {
      throw this.createImageReadinessError(
        'Image source changed before it was ready.',
        'IMAGE_SOURCE_CHANGED'
      );
    }

    if (!img.complete || img.naturalWidth === 0 || img.naturalHeight === 0) {
      await new Promise((resolve, reject) => {
        let timeoutId = null;
        const cleanup = () => {
          img.removeEventListener('load', onLoad);
          img.removeEventListener('error', onError);
          if (timeoutId) clearTimeout(timeoutId);
        };
        const onLoad = () => {
          cleanup();
          resolve();
        };
        const onError = () => {
          cleanup();
          reject(this.createImageReadinessError('Image failed to load.', 'IMAGE_NOT_READY'));
        };

        img.addEventListener('load', onLoad, { once: true });
        img.addEventListener('error', onError, { once: true });
        timeoutId = setTimeout(() => {
          cleanup();
          reject(this.createImageReadinessError('Timed out waiting for image load.', 'IMAGE_NOT_READY'));
        }, timeoutMs);
      });
    }

    try {
      await img.decode();
    } catch (error) {
      throw this.createImageReadinessError(
        `Image could not be decoded: ${error.message}`,
        'IMAGE_NOT_READY'
      );
    }

    const decodedSrc = this.getCurrentImageSource(img);
    if (expectedSrc && decodedSrc !== expectedSrc) {
      throw this.createImageReadinessError(
        'Image source changed while it was decoding.',
        'IMAGE_SOURCE_CHANGED'
      );
    }
    if (!img.complete || img.naturalWidth === 0 || img.naturalHeight === 0) {
      throw this.createImageReadinessError('Image is not fully decoded.', 'IMAGE_NOT_READY');
    }
    return decodedSrc;
  }

  async getImageDataUrl(img, expectedSrc) {
    const decodedSrc = await this.waitForImageReady(img, expectedSrc);
    try {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Canvas 2D context is unavailable.');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.9);
      if (this.getCurrentImageSource(img) !== decodedSrc) {
        throw this.createImageReadinessError(
          'Image source changed while canvas was being created.',
          'IMAGE_SOURCE_CHANGED'
        );
      }
      return imageDataUrl;
    } catch (e) {
      if (e && (e.code === 'IMAGE_NOT_READY' || e.code === 'IMAGE_SOURCE_CHANGED')) throw e;
      const result = await this.sendMessage({ action: 'fetchImage', url: decodedSrc });
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

  restoreAllOriginals() {
    this.processingQueue = [];
    this.activeJobs.forEach(({ img }) => {
      this.finishImageLoading(null, img);
      delete img.dataset.mtJobId;
    });
    this.activeJobs.clear();
    this.hideProgressBar();

    document.querySelectorAll('img[data-mt-original-src]').forEach(img => {
      if (img.dataset.mtCacheObjectUrl && window.MangaTranslationCache) {
        window.MangaTranslationCache.revokeImageUrl(img.dataset.mtCacheObjectUrl);
        delete img.dataset.mtCacheObjectUrl;
      }
      img.src = img.dataset.mtOriginalSrc;
      delete img.dataset.mtCacheKey;
    });
    this.processedImages.clear();
    this.totalProcessed = 0;
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => new MangaTranslator());
} else {
  new MangaTranslator();
}
