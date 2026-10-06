class MangaTranslator {
  constructor() {
    this.processedImages = new Set();
    this.pendingImages = new Set();
    this.failedImages = new Map();
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
    this.operationGeneration = 0;
    this.scanToken = null;
    this.responsiveImageStates = new WeakMap();
    this.latestContextJobs = new WeakMap();
    this.imageObserver = null;
    this.cacheClearGeneration = null;
    this.restoredImageSources = new WeakMap();
    this.settingsLoadId = 0;

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
    this.setupMutationObserver();

    const hostname = window.location.hostname;
    const enabledList = (this.settings && this.settings.enabledDomains) || [];
    if (!enabledList.includes(hostname)) {
      this.isEnabled = false;
      return;
    }

    this.isEnabled = true;
    this.scanAndProcess();
  }

  async loadSettings() {
    const loadId = ++this.settingsLoadId;
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
        if (loadId === this.settingsLoadId) this.settings = items;
        resolve(items);
      });
    });
  }

  setupMessageListener() {
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      switch (request.action) {
        case 'translateAllImages':
          (async () => {
            this.restoreAllOriginals({ keepOriginals: false });
            const generation = this.operationGeneration;
            this.cacheClearGeneration = generation;
            try {
              if (window.MangaTranslationCache) {
                await window.MangaTranslationCache.clearPageCache(window.location.href);
              }
            } finally {
              if (this.cacheClearGeneration === generation) this.cacheClearGeneration = null;
            }
            if (this.isEnabled && generation === this.operationGeneration) {
              this.scanAndProcess({ bypassCache: true });
            }
          })().catch(error => console.warn('[MangaTranslator] Rescan failed:', error));
          sendResponse({ success: true, status: 'started' });
          break;

        case 'setSiteEnabled':
          this.setSiteEnabled(request.enabled);
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
          sendResponse(this.startContextMenuLoading(request.srcUrl, request.jobId, request.settings));
          break;

        case 'contextMenuTranslateResult':
          if (request.result && request.result.success) {
            this.applyInpaintBySrc(request.jobId, request.result.data);
          }
          this.finishContextMenuLoading(request.jobId);
          sendResponse({ success: true });
          break;

        case 'contextMenuTranslateError':
          this.finishContextMenuLoading(request.jobId);
          sendResponse({ success: true });
          break;

        default:
          sendResponse({ success: false });
      }
      return true;
    });

    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === 'sync') {
        if (changes.enabledDomains &&
            !(changes.enabledDomains.newValue || []).includes(window.location.hostname)) {
          this.setSiteEnabled(false);
        }
        this.loadSettings().then(() => {
          if (changes.enabledDomains) {
            this.setSiteEnabled((this.settings.enabledDomains || []).includes(window.location.hostname));
          }
          if (changes.loadingStyle) this.syncLoadingStyle();
        });
      }
    });
  }

  setSiteEnabled(enabled) {
    enabled = Boolean(enabled);
    if (this.isEnabled === enabled) return;
    this.cancelPendingWork();
    this.isEnabled = enabled;
    if (enabled) {
      this.restoredImageSources = new WeakMap();
      this.scanAndProcess();
    } else {
      this.hideProgressBar();
    }
  }

  setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      if (e.altKey && e.key.toLowerCase() === 't') {
        e.preventDefault();
        this.scanAndProcess({ retryFailed: true });
      } else if (e.altKey && e.key.toLowerCase() === 'c') {
        e.preventDefault();
        this.restoreAllOriginals();
      }
    });
  }

  async scanAndProcess(options = {}) {
    if (!this.isEnabled) return;
    if (this.cacheClearGeneration === this.operationGeneration) return;
    if (this.isScanning) return;
    if (options.retryFailed) {
      this.failedImages.clear();
      this.restoredImageSources = new WeakMap();
    }
    this.isScanning = true;
    const token = {};
    const generation = this.operationGeneration;
    this.scanToken = token;
    try {
      this.showProgressBar();
      const settings = this.settings || await this.loadSettings();
      if (!this.isEnabled || generation !== this.operationGeneration) return;
      let restoredFromCache = 0;

      if (this.progressBar) {
        const pText = this.progressBar.querySelector('.manga-translator-progress-text');
        if (pText) pText.textContent = '⚡ Checking image cache...';
      }

      if (!options.bypassCache && window.MangaTranslationCache) {
        restoredFromCache += await this.restoreCachedImages(settings, this.getPageImages(), generation);
      }

      if (!this.isEnabled || generation !== this.operationGeneration) {
        return;
      }

      if (this.progressBar) {
        const pText = this.progressBar.querySelector('.manga-translator-progress-text');
        if (pText) pText.textContent = '⏳ Waiting for images that are not cached...';
      }

      await this.waitForAllImagesToLoad(generation);
      if (!this.isEnabled || generation !== this.operationGeneration) {
        return;
      }

      let mangaImgs = this.getPageImages().filter(
        img => this.canProcessImage(img) && this.isMangaImage(img)
      );

      if (!options.bypassCache && window.MangaTranslationCache && mangaImgs.length > 0) {
        restoredFromCache += await this.restoreCachedImages(settings, mangaImgs, generation);
        if (!this.isEnabled || generation !== this.operationGeneration) return;
        mangaImgs = mangaImgs.filter(img => this.canProcessImage(img));
      }

      if (mangaImgs.length === 0 && this.processingQueue.length === 0 && this.activeJobs.size === 0) {
        this.totalImages = this.processedImages.size + this.pendingImages.size + this.failedImages.size;
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

        this.failedImages.delete(img);
        this.pendingImages.add(img);
        this.wrapImage(img);
        this.processingQueue.push(img);
      }

      this.totalImages = this.processedImages.size + this.pendingImages.size + this.failedImages.size;

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
      this.pumpQueue();
    } catch (error) {
      console.warn('[MangaTranslator] Image scan failed:', error);
      if (this.scanToken === token) this.hideProgressBar();
    } finally {
      if (this.scanToken === token) {
        this.isScanning = false;
        this.scanToken = null;
      }
    }
  }

  async waitForAllImagesToLoad(generation = this.operationGeneration) {
    const maxWaitMs = 15000;
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitMs) {
      if (!this.isEnabled || generation !== this.operationGeneration) return;
      const allImgs = this.getPageImages().filter(
        img => this.canProcessImage(img) && (img.src || img.dataset.src || img.dataset.lazySrc)
      );
      if (allImgs.length === 0) break;

      const pending = allImgs.filter(img => !img.complete);

      if (pending.length === 0) {
        await new Promise(r => setTimeout(r, 600));
        const checkAgain = this.getPageImages().filter(
          img => this.canProcessImage(img) && (img.src || img.dataset.src || img.dataset.lazySrc) && !img.complete
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
    const state = this.responsiveImageStates.get(img);
    if (this.hasTranslatedImageSource(img, state)) return state.originalSrc;
    return (img.complete && img.naturalWidth > 0 ? this.getCurrentImageSource(img) : '') ||
      img.dataset.src ||
      img.dataset.lazySrc ||
      img.dataset.original ||
      img.currentSrc ||
      img.src ||
      '';
  }

  hasTranslatedImageSource(img, state = this.responsiveImageStates.get(img)) {
    return Boolean(state?.appliedSrc && img.getAttribute('src') === state.appliedSrc &&
      !img.hasAttribute('srcset') && state.sources.every(({ source }) => !source.hasAttribute('srcset')));
  }

  getImageSourceSignature(img, includeCurrentSrc = true) {
    const picture = img.parentElement?.tagName === 'PICTURE' ? img.parentElement : null;
    return JSON.stringify([
      includeCurrentSrc ? img.currentSrc : null,
      img.getAttribute('src'), img.getAttribute('srcset'), img.getAttribute('sizes'),
      picture ? Array.from(picture.querySelectorAll('source')).map(source => [
        source.getAttribute('srcset'), source.getAttribute('sizes'),
        source.getAttribute('media'), source.getAttribute('type')
      ]) : []
    ]);
  }

  captureImageSource(img) {
    const previous = this.responsiveImageStates.get(img);
    if (this.hasTranslatedImageSource(img, previous)) return previous;
    const picture = img.parentElement?.tagName === 'PICTURE' ? img.parentElement : null;
    const state = {
      originalSrc: this.getCacheSource(img),
      attributes: ['src', 'srcset', 'sizes'].map(name => [name, img.getAttribute(name)]),
      sources: picture ? Array.from(picture.querySelectorAll('source')).map(source => ({
        source, attributes: ['srcset', 'sizes'].map(name => [name, source.getAttribute(name)])
      })) : [],
      appliedSrc: null
    };
    this.responsiveImageStates.set(img, state);
    return state;
  }

  setTranslatedImageSource(img, imageUrl) {
    const state = this.captureImageSource(img);
    img.removeAttribute('srcset');
    img.removeAttribute('sizes');
    state.sources.forEach(({ source }) => source.removeAttribute('srcset'));
    img.dataset.mtOriginalSrc = state.originalSrc;
    img.src = imageUrl;
    state.appliedSrc = imageUrl;
  }

  restoreImageSource(img) {
    const state = this.responsiveImageStates.get(img);
    if (this.hasTranslatedImageSource(img, state)) {
      const restoreAttributes = (element, attributes) => attributes.forEach(([name, value]) => {
        if (value === null) element.removeAttribute(name);
        else element.setAttribute(name, value);
      });
      restoreAttributes(img, state.attributes);
      state.sources.forEach(({ source, attributes }) => restoreAttributes(source, attributes));
    }
    this.responsiveImageStates.delete(img);
    delete img.dataset.mtOriginalSrc;
  }

  getImageCacheDescriptor(img, settings, knownIndex = null) {
    const originalSrc = this.getCacheSource(img);
    const imageIndex = knownIndex === null ? this.getPageImages().indexOf(img) : knownIndex;
    const cache = window.MangaTranslationCache;

    return {
      img,
      originalSrc,
      sourceSignature: this.getImageSourceSignature(img),
      imageIndex,
      cacheKey: cache ? cache.buildCacheKey(
        settings,
        originalSrc,
        window.location.href,
        imageIndex
      ) : ''
    };
  }

  async restoreCachedImages(settings, candidateImages, generation = this.operationGeneration) {
    const cache = window.MangaTranslationCache;
    if (!cache || !this.isEnabled || generation !== this.operationGeneration) return 0;

    const candidates = new Set(candidateImages || []);
    const descriptors = this.getPageImages()
      .map((img, imageIndex) => this.getImageCacheDescriptor(img, settings, imageIndex))
      .filter(({ img, originalSrc }) =>
        candidates.has(img) && this.canProcessImage(img) && Boolean(originalSrc)
      );

    if (descriptors.length === 0) return 0;

    const keys = descriptors.map(({ cacheKey }) => cacheKey);
    const cachedEntries = cache.getCachedTranslations
      ? await cache.getCachedTranslations(keys)
      : new Map(await Promise.all(keys.map(async key => [key, await cache.getCachedTranslation(key)])));

    let restored = 0;
    for (const descriptor of descriptors) {
      if (!this.isEnabled || generation !== this.operationGeneration) break;
      if (!descriptor.img.isConnected || !this.canProcessImage(descriptor.img) ||
          descriptor.sourceSignature !== this.getImageSourceSignature(descriptor.img)) continue;
      const cached = cachedEntries.get(descriptor.cacheKey);
      if (!cached || cache.normalizeUrlForKey(cached.originalSrc, true) !==
          cache.normalizeUrlForKey(descriptor.originalSrc, true)) continue;
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
      this.setTranslatedImageSource(descriptor.img, imageUrl);
      if (imageUrl.startsWith('blob:')) {
        descriptor.img.dataset.mtCacheObjectUrl = imageUrl;
      } else {
        delete descriptor.img.dataset.mtCacheObjectUrl;
      }

      this.processedImages.add(descriptor.img);
      this.wrapImage(descriptor.img);
      this.totalProcessed++;
      restored++;
    }

    return restored;
  }

  getCurrentImageSource(img) {
    return img.currentSrc || img.src || img.dataset.src || img.dataset.lazySrc || '';
  }

  canProcessImage(img) {
    if (this.processedImages.has(img) || this.pendingImages.has(img)) return false;
    const restoredSource = this.restoredImageSources.get(img);
    if (restoredSource) {
      if (restoredSource === this.getImageSourceSignature(img, false)) return false;
      this.restoredImageSources.delete(img);
    }
    const failure = this.failedImages.get(img);
    return !failure || failure.source !== this.getCurrentImageSource(img) ||
      Date.now() - failure.timestamp >= 30000;
  }

  recordImageFailure(img, error) {
    this.failedImages.set(img, {
      source: this.getCurrentImageSource(img),
      timestamp: Date.now(),
      error: error || 'Image translation failed.'
    });
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
    const generation = this.operationGeneration;
    const jobId = this.createJobId();
    const settings = { ...(this.settings || await this.loadSettings()) };
    if (!this.isEnabled || generation !== this.operationGeneration || !img.isConnected) return;
    const descriptor = this.getImageCacheDescriptor(img, settings);
    const originalSrc = this.getCurrentImageSource(img);
    const job = {
      id: jobId,
      img,
      originalSrc,
      sourceSignature: this.getImageSourceSignature(img),
      generation,
      settings,
      originalWidth: img.naturalWidth,
      originalHeight: img.naturalHeight,
      cacheKey: window.MangaTranslationCache
        ? window.MangaTranslationCache.buildCacheKey(settings, originalSrc, window.location.href, descriptor.imageIndex)
        : descriptor.cacheKey
    };
    this.failedImages.delete(img);
    this.pendingImages.add(img);
    img.dataset.mtOriginalSrc = originalSrc;
    img.dataset.mtCacheKey = job.cacheKey;
    this.activeJobs.set(jobId, job);
    img.dataset.mtJobId = jobId;

    this.showProgressBar();
    this.updateProgressBar();
    this.beginImageLoading(null, img);

    try {
      const result = await this.translateImage(job);
      const isCurrentJob = this.activeJobs.get(jobId) === job && img.dataset.mtJobId === jobId;
      if (!isCurrentJob || generation !== this.operationGeneration || !img.isConnected) return;

      if (this.getImageSourceSignature(img) !== job.sourceSignature) {
        delete img.dataset.mtOriginalSrc;
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
        this.setTranslatedImageSource(img, result.data);
        this.processedImages.add(img);
        this.failedImages.delete(img);
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
        this.recordImageFailure(img, result.error);
        console.warn('[MangaTranslator] Inpaint failed:', result.error);
      }
    } catch (e) {
      if (this.activeJobs.get(jobId) !== job || img.dataset.mtJobId !== jobId) return;
      if (e && (e.code === 'IMAGE_NOT_READY' || e.code === 'IMAGE_SOURCE_CHANGED')) {
        delete img.dataset.mtOriginalSrc;
        delete img.dataset.mtCacheKey;
        this.scheduleScan();
      } else {
        this.recordImageFailure(img, e && e.message || String(e));
      }
      console.error('[MangaTranslator] Failed to translate image:', e);
    } finally {
      const ownsJob = this.activeJobs.get(jobId) === job;
      if (ownsJob) {
        this.activeJobs.delete(jobId);
        this.pendingImages.delete(img);
        if (img.dataset.mtJobId === jobId) delete img.dataset.mtJobId;
        this.finishImageLoading(null, img);
      }
      this.updateProgressBar();
      this.pumpQueue();
    }
  }

  async translateImage(job) {
    const settings = job.settings;
    const fileData = await this.getImageDataUrl(job.img, job.originalSrc);

    return this.sendMessage({
      action: 'inpaintPageStream',
      data: {
        fileData: fileData,
        imageSrc: job.originalSrc,
        jobId: job.id,
        apiUrl: settings.apiUrl,
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
    this.captureImageSource(img);
    if (!img.dataset.mtOriginalSrc) img.dataset.mtOriginalSrc = this.getCacheSource(img);
    if (img.closest('.manga-translator-wrapper')) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'manga-translator-wrapper';
    const target = img.parentElement?.tagName === 'PICTURE' ? img.parentElement : img;
    target.parentNode.insertBefore(wrapper, target);
    wrapper.appendChild(target);

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
      if (img.currentSrc === srcUrl || img.src === srcUrl || img.dataset.mtOriginalSrc === srcUrl) return img;
    }
    return Array.from(document.images).find(
      img => img.currentSrc === srcUrl || img.src === srcUrl || img.dataset.mtOriginalSrc === srcUrl
    ) || null;
  }

  applyInpaintBySrc(jobId, dataUrl) {
    const job = this.contextMenuLoadingSources.get(jobId);
    if (!job || typeof dataUrl !== 'string' || !dataUrl) return;
    const { img, settings, originalSrc, originalWidth, originalHeight, cacheKey } = job;
    if (job.generation !== this.operationGeneration || !img.isConnected ||
        this.latestContextJobs.get(img) !== jobId ||
        job.sourceSignature !== this.getImageSourceSignature(img)) return;
    if (img.dataset.mtCacheObjectUrl && window.MangaTranslationCache) {
      window.MangaTranslationCache.revokeImageUrl(img.dataset.mtCacheObjectUrl);
      delete img.dataset.mtCacheObjectUrl;
    }
    this.setTranslatedImageSource(img, dataUrl);
    img.dataset.mtCacheKey = cacheKey;
    if (!this.processedImages.has(img)) this.totalProcessed++;
    this.processedImages.add(img);
    this.failedImages.delete(img);
    if (window.MangaTranslationCache) {
      window.MangaTranslationCache.saveCachedTranslation({
        cacheKey,
        pageUrl: window.location.href,
        originalSrc,
        translatedData: dataUrl,
        originalWidth,
        originalHeight,
        targetLang: settings.targetLang,
        translator: settings.translator,
        readingDirection: settings.readingDirection,
        timestamp: Date.now()
      }).catch(() => {});
    }
    this.totalImages = this.processedImages.size + this.pendingImages.size + this.failedImages.size;
    this.updateProgressBar();
    this.pumpQueue();
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

  startContextMenuLoading(srcUrl, jobId, requestSettings) {
    const img = this.findImageBySrc(srcUrl);
    if (!img || !jobId || !requestSettings || this.contextMenuLoadingSources.has(jobId)) {
      return { success: false, error: 'The image or translation request is no longer available.' };
    }
    const settings = { ...requestSettings };
    const descriptor = this.getImageCacheDescriptor(img, settings);
    this.contextMenuLoadingSources.set(jobId, {
      ...descriptor, settings, srcUrl,
      generation: this.operationGeneration,
      originalWidth: img.naturalWidth,
      originalHeight: img.naturalHeight
    });
    this.latestContextJobs.set(img, jobId);
    this.processingQueue = this.processingQueue.filter(queued => queued !== img);
    this.activeJobs.forEach((job, activeId) => {
      if (job.img !== img) return;
      this.finishImageLoading(null, img);
      this.activeJobs.delete(activeId);
    });
    delete img.dataset.mtJobId;
    this.pendingImages.add(img);
    this.beginImageLoading(null, img);
    return { success: true, sourceUrl: descriptor.originalSrc };
  }

  finishContextMenuLoading(jobId) {
    const activeLoading = this.contextMenuLoadingSources.get(jobId);
    if (!activeLoading) return;
    this.contextMenuLoadingSources.delete(jobId);
    if (this.latestContextJobs.get(activeLoading.img) === jobId) {
      this.latestContextJobs.delete(activeLoading.img);
      this.pendingImages.delete(activeLoading.img);
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
      this.contextMenuLoadingSources.forEach(({ img }) => this.showImageLoading(null, img));
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
    this.contextMenuLoadingSources.forEach(({ img }) => this.showImageLoading(null, img));
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
    const failed = this.failedImages.size;
    text.textContent = failed
      ? `${this.totalProcessed} pages translated; ${failed} failed. Press Alt+T to retry.`
      : `✅ Done! ${this.totalProcessed} pages translated`;
    fill.style.width = failed
      ? `${this.totalImages ? (this.totalProcessed / this.totalImages) * 100 : 0}%`
      : '100%';

    if (this.progressBarHideTimer) clearTimeout(this.progressBarHideTimer);
    this.progressBarHideTimer = setTimeout(() => {
      if (this.progressBar) this.progressBar.style.display = 'none';
      this.progressBarHideTimer = null;
    }, 3000);
  }

  setupMutationObserver() {
    if (this.imageObserver) return;
    const observer = new MutationObserver((mutations) => {
      if (!this.isEnabled) return;
      let hasNewImages = false;
      for (const mutation of mutations) {
        if (mutation.type === 'attributes' && ['IMG', 'SOURCE'].includes(mutation.target.nodeName)) {
          const images = mutation.target.nodeName === 'IMG'
            ? [mutation.target] : Array.from(mutation.target.parentElement?.querySelectorAll('img') || []);
          images.forEach(img => {
            if (this.processedImages.has(img) && !this.hasTranslatedImageSource(img)) {
              this.processedImages.delete(img);
              this.totalProcessed = Math.max(0, this.totalProcessed - 1);
            }
          });
          hasNewImages = true;
          continue;
        }
        for (const node of mutation.addedNodes) {
          if (node.nodeName === 'IMG' || (node.querySelectorAll && node.querySelectorAll('img').length > 0)) {
            hasNewImages = true;
            break;
          }
        }
      }
      if (hasNewImages) {
        this.scheduleScan(1000);
      }
    });
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['src', 'srcset', 'sizes', 'media', 'type']
    });
    this.imageObserver = observer;

    document.addEventListener('load', (event) => {
      const img = event.target;
      if (
        this.isEnabled &&
        img instanceof HTMLImageElement &&
        !img.closest('.manga-translator-minimal-loader') &&
        this.canProcessImage(img)
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

  cancelPendingWork() {
    this.operationGeneration++;
    this.scanToken = null;
    this.cacheClearGeneration = null;
    this.isScanning = false;
    if (this.scanDebounceTimer) clearTimeout(this.scanDebounceTimer);
    this.scanDebounceTimer = null;
    this.processingQueue = [];
    this.activeJobs.forEach(({ img }) => {
      this.finishImageLoading(null, img);
      delete img.dataset.mtJobId;
    });
    this.activeJobs.clear();
    this.contextMenuLoadingSources.forEach((job, jobId) => this.finishContextMenuLoading(jobId));
    this.pendingImages.clear();
  }

  restoreAllOriginals({ keepOriginals = true } = {}) {
    this.cancelPendingWork();
    if (!keepOriginals) this.restoredImageSources = new WeakMap();
    this.hideProgressBar();

    document.querySelectorAll('img[data-mt-original-src]').forEach(img => {
      if (img.dataset.mtCacheObjectUrl && window.MangaTranslationCache) {
        window.MangaTranslationCache.revokeImageUrl(img.dataset.mtCacheObjectUrl);
        delete img.dataset.mtCacheObjectUrl;
      }
      this.restoreImageSource(img);
      if (keepOriginals) {
        this.restoredImageSources.set(img, this.getImageSourceSignature(img, false));
      }
      delete img.dataset.mtCacheKey;
    });
    this.processedImages.clear();
    this.pendingImages.clear();
    this.failedImages.clear();
    this.totalProcessed = 0;
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => new MangaTranslator());
} else {
  new MangaTranslator();
}
