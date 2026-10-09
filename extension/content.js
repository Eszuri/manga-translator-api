class MangaTranslator {
  constructor() {
    this.processedImages = new Set();
    this.pendingImages = new Set();
    this.failedImages = new Map();
    this.processingQueue = [];
    this.runningJob = null;
    this.runningImage = null;
    this.manualRequests = new Map();
    this.activeJobs = new Map();
    this.completedJobs = new Map();
    this.sourceOrder = new Map();
    this.prioritySource = null;
    this.latestContextSelection = 0;
    this.batchReady = false;
    this.imageLoadFailures = new Map();
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
    this.setupImageViewSync();
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
        llmMergeOcr: true,
        readingDirection: 'rtl',
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
          sendResponse(this.startContextMenuLoading(request.srcUrl, request.jobId, request.settings, request.selectedAt));
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
    this.syncActiveJobImages();
    if (!this.isEnabled) return;
    if (this.cacheClearGeneration === this.operationGeneration) return;
    if (this.isScanning) return;
    if (options.retryFailed) {
      this.failedImages.clear();
      this.imageLoadFailures.clear();
      this.restoredImageSources = new WeakMap();
    }
    this.isScanning = true;
    const token = {};
    let loadComplete = false;
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
      loadComplete = true;
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

      if (!this.batchReady) this.sourceOrder.clear();
      this.orderImagesByPage(this.getPageImages().filter(img => this.isMangaImage(img)))
        .forEach(img => this.rememberImageOrder(img));
      const prioritizedImages = this.orderImagesByPage(mangaImgs);
      for (const img of prioritizedImages) this.rememberImageOrder(img);
      for (const img of prioritizedImages) {
        try {
          await this.waitForImageReady(img);
        } catch (error) {
          this.recordImageFailure(img, error.message);
          continue;
        }
        if (!this.isEnabled || generation !== this.operationGeneration) return;
        if (!img.isConnected || !this.canProcessImage(img) || !this.isMangaImage(img)) continue;
        const descriptor = this.getImageCacheDescriptor(img, settings);
        const originalSrc = descriptor.originalSrc;
        if (!img.dataset.mtOriginalSrc) {
          img.dataset.mtOriginalSrc = originalSrc;
        }

        img.dataset.mtCacheKey = descriptor.cacheKey;

        this.failedImages.delete(img);
        this.pendingImages.add(img);
        this.wrapImage(img);
        this.setQueuedState(img, true);
        if (!this.processingQueue.includes(img)) this.processingQueue.push(img);
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
        if (loadComplete) this.batchReady = true;
        this.pumpQueue();
      }
    }
  }

  async waitForAllImagesToLoad(generation = this.operationGeneration) {
    let settledAt = null;
    let previousSources = '';
    const started = new Map();
    const deadline = Date.now() + 60000;
    while (this.isEnabled && generation === this.operationGeneration) {
      const allImgs = this.getPageImages().filter(
        img => this.canProcessImage(img) && !this.getImageExclusionReason(img, false) &&
          (img.src || img.dataset.src || img.dataset.lazySrc || img.dataset.original ||
           img.dataset.url || img.dataset.highres || img.getAttribute('data-srcset') ||
           this.getPictureSources(img).some(source => source.getAttribute('data-srcset')))
      );
      const pending = [];
      const sources = [];
      for (const img of allImgs) {
        this.loadLazyImage(img);
        const source = this.getLoadingIdentity(img);
        sources.push(source);
        if (this.imageLoadFailures.has(source)) continue;
        if (!started.has(img)) started.set(img, Date.now());
        if (img.complete && img.naturalWidth > 0) continue;
        if ((img.complete && img.naturalWidth === 0) || Date.now() - started.get(img) >= 30000 || Date.now() >= deadline) {
          const reason = img.complete ? 'Image failed to load.' : 'Image loading exceeded 30 seconds.';
          this.imageLoadFailures.set(source, reason);
          this.recordImageFailure(img, reason);
          console.warn('[MangaTranslator] Skipping image:', source, reason);
        } else pending.push(img);
      }
      const sourceSnapshot = JSON.stringify(sources.sort());
      if (sourceSnapshot !== previousSources) settledAt = null;
      previousSources = sourceSnapshot;

      if (pending.length === 0) {
        if (settledAt === null) settledAt = Date.now();
        if (Date.now() - settledAt >= 600) break;
      } else settledAt = null;

      if (this.progressBar && this.activeJobs.size === 0) {
        const pText = this.progressBar.querySelector('.manga-translator-progress-text');
        if (pText) {
          pText.textContent = `⏳ Waiting for images to load (${pending.length} remaining)...`;
        }
      }

      await new Promise(resolve => setTimeout(resolve, 250));
    }
  }

  getLoadingIdentity(img) {
    if (this.hasResponsiveImageSource(img) && !img.complete) {
      return `responsive:${this.getImageSourceSignature(img, false)}`;
    }
    return this.getCanonicalSource(img) || img.getAttribute('src') ||
      `responsive:${this.getImageSourceSignature(img, false)}`;
  }

  loadLazyImage(img) {
    // Promote native and common data-attribute lazy sources without scrolling
    // the page or requesting a different chapter from an infinite reader.
    if (img.loading === 'lazy') img.loading = 'eager';
    const lazySrcset = img.getAttribute('data-srcset');
    if (lazySrcset && !img.getAttribute('srcset')) img.setAttribute('srcset', lazySrcset);
    const lazySrc = img.dataset.src || img.dataset.lazySrc || img.dataset.original || img.dataset.url || img.dataset.highres;
    if (lazySrc && (!img.getAttribute('src') || img.src.startsWith('data:') ||
        /placeholder|blank/i.test(img.src) || (img.complete && !this.isMangaDimensions(img.naturalWidth, img.naturalHeight)))) {
      try {
        const source = new URL(lazySrc, window.location.href).href;
        if (img.src !== source) img.src = source;
      } catch { /* An invalid lazy URL will be reported as a load failure. */ }
    }
    this.getPictureSources(img).forEach(source => {
      const srcset = source.getAttribute('data-srcset');
      if (srcset && !source.getAttribute('srcset')) source.setAttribute('srcset', srcset);
    });
  }

  isMangaImage(img) {
    return img.complete && this.isMangaDimensions(img.naturalWidth, img.naturalHeight) &&
      !this.getImageExclusionReason(img);
  }

  getImageExclusionReason(img, checkIntrinsic = true) {
    const deferredSource = !checkIntrinsic && Boolean(img.dataset.src || img.dataset.lazySrc ||
      img.dataset.original || img.dataset.url || img.dataset.highres || img.getAttribute('data-srcset') ||
      this.getPictureSources(img).some(source => source.getAttribute('data-srcset')));
    if (checkIntrinsic && img.complete && !this.isMangaDimensions(img.naturalWidth, img.naturalHeight)) {
      return 'The original image must be at least 500 × 700 pixels.';
    }

    // Use layout size, not viewport intersection: full-size pages below the
    // viewport still belong in the queue. Adapt to narrow/mobile readers.
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 1024;
    const minDisplayWidth = Math.min(280, viewportWidth * 0.6);
    const width = img.clientWidth;
    const height = img.clientHeight;
    if (!deferredSource && width > 0 && height > 0 && (width < minDisplayWidth || height < 200)) {
      return 'The image is displayed too small and looks like a thumbnail.';
    }
    if (!deferredSource && img.complete && (width <= 0 || height <= 0)) {
      return 'The image is hidden or has no display area.';
    }

    // Inspect nearby image/card elements only; a page-wide sidebar/related
    // container must not classify unrelated reader pages as thumbnails.
    const marker = /(?:^|[^a-z0-9])(?:thumb(?:nail)?s?|covers?|posters?|avatars?|logos?|icons?|banners?|recommendations?|related|(?:manga|comic|series|book)[-_ ](?:card|cover|item))(?:$|[^a-z0-9])/i;
    for (let element = img, depth = 0; element && depth < 4; element = element.parentElement, depth++) {
      const identity = `${element.id || ''} ${element.getAttribute('class') || ''}`
        .replace(/([a-z])([A-Z])/g, '$1-$2');
      if (marker.test(identity)) return 'The image is marked as a thumbnail, cover, or recommendation.';
    }

    for (const source of [img.currentSrc, img.getAttribute('src'), img.dataset.src,
      img.dataset.lazySrc, img.dataset.original, img.dataset.url, img.dataset.highres]) {
      if (!source || /^(?:data|blob):/i.test(source)) continue;
      try {
        const path = decodeURIComponent(new URL(source, window.location.href).pathname);
        if (/(?:^|\/)(?:thumb(?:nail)?s?|covers?|posters?|avatars?|logos?|icons?|banners?)(?:\/|[_.-]|$)/i.test(path)) {
          return 'The image URL identifies a thumbnail, cover, or decorative image.';
        }
      } catch {}
    }
    return '';
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

  getCanonicalSource(img) {
    this.refreshImageSource(img);
    const state = this.responsiveImageStates.get(img);
    if (this.hasTranslatedImageSource(img, state)) return state.originalSrc;

    const lazySrc = img.dataset.src || img.dataset.lazySrc || img.dataset.original || img.dataset.url || img.dataset.highres;
    const responsive = this.hasResponsiveImageSource(img);
    if (responsive && !img.complete) return '';
    const current = (responsive ? img.currentSrc || img.src : img.src || img.currentSrc) || '';

    if (lazySrc && (!current || current.startsWith('data:') || current.startsWith('blob:') || current.includes('placeholder') || current.includes('blank'))) {
      try {
        return new URL(lazySrc, window.location.href).href;
      } catch {
        return lazySrc;
      }
    }

    if (current && !current.startsWith('data:') && !current.startsWith('blob:')) {
      try {
        return new URL(current, window.location.href).href;
      } catch {
        return current;
      }
    }

    if (lazySrc) {
      try {
        return new URL(lazySrc, window.location.href).href;
      } catch {
        return lazySrc;
      }
    }

    return current;
  }

  getCacheSource(img) {
    return this.getCanonicalSource(img);
  }

  getPictureSources(img) {
    return img.parentElement?.tagName === 'PICTURE'
      ? Array.from(img.parentElement.querySelectorAll('source')) : [];
  }

  hasResponsiveImageSource(img) {
    return Boolean(img.getAttribute('srcset') ||
      this.getPictureSources(img).some(source => source.getAttribute('srcset')));
  }

  hasSamePictureSources(img, sources) {
    const current = this.getPictureSources(img);
    return current.length === sources.length && current.every((source, index) => source === sources[index]);
  }

  hasTranslatedImageSource(img, state = this.responsiveImageStates.get(img)) {
    return Boolean(state?.appliedSrc &&
      (img.src === state.appliedSrc || img.getAttribute('src') === state.appliedSrc) &&
      this.hasSamePictureSources(img, state.sources.map(({ source }) => source)) &&
      state.appliedSignature === this.getImageSourceSignature(img, false));
  }

  refreshImageSource(img) {
    const state = this.responsiveImageStates.get(img);
    if (!state || this.hasTranslatedImageSource(img, state)) return;
    if (!state.appliedSrc && state.sourceSignature === this.getImageSourceSignature(img, false) &&
        this.hasSamePictureSources(img, state.sources.map(({ source }) => source))) return;

    // Restore only attributes still owned by us; preserve the page's new source.
    const restoreOwned = (element, original, applied) => {
      original.forEach(([name, value]) => {
        if (element.getAttribute(name) !== applied.get(name)) return;
        if (value === null) element.removeAttribute(name);
        else element.setAttribute(name, value);
      });
    };
    if (state.appliedSrc) {
      restoreOwned(img, state.attributes, state.appliedAttributes);
      restoreOwned(img, Object.entries(state.dataAttributes), state.appliedAttributes);
      state.sources.forEach(({ source, attributes, appliedAttributes }) => {
        restoreOwned(source, attributes, appliedAttributes);
      });
    }
    this.responsiveImageStates.delete(img);
    if (img.dataset.mtCacheObjectUrl && window.MangaTranslationCache) {
      window.MangaTranslationCache.revokeImageUrl(img.dataset.mtCacheObjectUrl);
    }
    ['mtOriginalSrc', 'mtTranslated', 'mtCacheKey', 'mtCacheObjectUrl'].forEach(name => delete img.dataset[name]);
    if (this.processedImages.delete(img)) this.totalProcessed = Math.max(0, this.totalProcessed - 1);
    this.failedImages.delete(img);
    this.restoredImageSources.delete(img);
    this.processingQueue = this.processingQueue.filter(queued => queued !== img);
    this.manualRequests.delete(img);
    this.setQueuedState(img, false);
    this.activeJobs.forEach((job, jobId) => {
      if (job.img !== img) return;
      // DOM/lazy-loading changes do not cancel an uploaded server job.
      this.finishImageLoading(null, img);
    });
    this.contextMenuLoadingSources.forEach((job, jobId) => {
      if (job.img === img) this.finishContextMenuLoading(jobId);
    });
    this.pendingImages.delete(img);
    delete img.dataset.mtJobId;
    if (this.isEnabled) this.scheduleScan();
  }

  getImageSourceSignature(img, includeCurrentSrc = true) {
    const picture = img.parentElement?.tagName === 'PICTURE' ? img.parentElement : null;
    return JSON.stringify([
      includeCurrentSrc ? img.currentSrc : null,
      img.getAttribute('src'), img.getAttribute('srcset'), img.getAttribute('sizes'),
      ['data-src', 'data-lazy-src', 'data-original', 'data-url', 'data-highres']
        .map(name => img.getAttribute(name)),
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
      originalSrc: this.getCanonicalSource(img),
      sourceSignature: this.getImageSourceSignature(img, false),
      attributes: ['src', 'srcset', 'sizes'].map(name => [name, img.getAttribute(name)]),
      dataAttributes: {},
      sources: picture ? Array.from(picture.querySelectorAll('source')).map(source => ({
        source, attributes: ['srcset', 'sizes'].map(name => [name, source.getAttribute(name)])
      })) : [],
      appliedSrc: null
    };

    ['data-src', 'data-lazy-src', 'data-original', 'data-url', 'data-highres'].forEach(attr => {
      if (img.hasAttribute(attr)) {
        state.dataAttributes[attr] = img.getAttribute(attr);
      }
    });

    this.responsiveImageStates.set(img, state);
    return state;
  }

  setTranslatedImageSource(img, imageUrl) {
    const state = this.captureImageSource(img);
    img.removeAttribute('srcset');
    img.removeAttribute('sizes');
    state.sources.forEach(({ source }) => {
      source.removeAttribute('srcset');
      source.srcset = imageUrl;
    });
    img.dataset.mtOriginalSrc = state.originalSrc;
    img.dataset.mtTranslated = '1';
    img.src = imageUrl;
    state.appliedSrc = imageUrl;

    ['data-src', 'data-lazy-src', 'data-original', 'data-url', 'data-highres'].forEach(attr => {
      if (img.hasAttribute(attr)) {
        img.setAttribute(attr, imageUrl);
      }
    });
    state.appliedAttributes = new Map(
      [...state.attributes.map(([name]) => name), ...Object.keys(state.dataAttributes)]
        .map(name => [name, img.getAttribute(name)])
    );
    state.sources.forEach(sourceState => {
      sourceState.appliedAttributes = new Map(sourceState.attributes.map(([name]) =>
        [name, sourceState.source.getAttribute(name)]));
    });
    state.appliedSignature = this.getImageSourceSignature(img, false);
  }

  restoreImageSource(img) {
    this.refreshImageSource(img);
    const state = this.responsiveImageStates.get(img);
    if (state) {
      const restoreAttributes = (element, attributes) => attributes.forEach(([name, value]) => {
        if (value === null) element.removeAttribute(name);
        else element.setAttribute(name, value);
      });
      restoreAttributes(img, state.attributes);
      state.sources.forEach(({ source, attributes }) => restoreAttributes(source, attributes));
      if (state.dataAttributes) {
        Object.entries(state.dataAttributes).forEach(([attr, val]) => {
          if (val === null) img.removeAttribute(attr);
          else img.setAttribute(attr, val);
        });
      }
    }
    this.responsiveImageStates.delete(img);
    delete img.dataset.mtOriginalSrc;
    delete img.dataset.mtTranslated;
  }

  getImageCacheDescriptor(img, settings) {
    const canonicalSrc = this.getCanonicalSource(img);
    const cache = window.MangaTranslationCache;

    return {
      img,
      originalSrc: canonicalSrc,
      sourceSignature: this.getImageSourceSignature(img, false),
      sourceElements: this.getPictureSources(img),
      cacheKey: cache ? cache.buildCacheKey(
        settings,
        canonicalSrc,
        window.location.href
      ) : ''
    };
  }

  isImageSourceCurrent({ img, originalSrc, sourceSignature, sourceElements }) {
    return img.isConnected && this.getCanonicalSource(img) === originalSrc &&
      this.hasSamePictureSources(img, sourceElements) &&
      this.getImageSourceSignature(img, false) === sourceSignature;
  }

  assertImageSourceCurrent(descriptor) {
    if (this.isImageSourceCurrent(descriptor)) return;
    const error = new Error('The image source changed while translation was pending.');
    error.code = 'IMAGE_SOURCE_CHANGED';
    throw error;
  }

  async restoreCachedImages(settings, candidateImages, generation = this.operationGeneration) {
    const cache = window.MangaTranslationCache;
    if (!cache || !this.isEnabled || generation !== this.operationGeneration) return 0;

    const candidates = new Set(candidateImages || []);
    const descriptors = this.getPageImages()
      .map(img => this.getImageCacheDescriptor(img, settings))
      .filter(({ img, originalSrc }) =>
        candidates.has(img) && this.canProcessImage(img) && !this.getImageExclusionReason(img) && Boolean(originalSrc)
      );

    if (descriptors.length === 0) return 0;

    const keys = descriptors.map(({ cacheKey }) => cacheKey);
    const cachedEntries = cache.getCachedTranslations
      ? await cache.getCachedTranslations(keys)
      : new Map(await Promise.all(keys.map(async key => [key, await cache.getCachedTranslation(key)])));

    let restored = 0;
    for (const descriptor of descriptors) {
      if (!this.isEnabled || generation !== this.operationGeneration) break;
      if (!this.isImageSourceCurrent(descriptor) || !this.canProcessImage(descriptor.img)) continue;
      const cached = cachedEntries.get(descriptor.cacheKey);
      if (!cached) continue;
      if (this.getImageExclusionReason(descriptor.img) ||
          !this.isMangaDimensions(cached.originalWidth, cached.originalHeight)) continue;
      if (this.hasResponsiveImageSource(descriptor.img)) {
        try {
          await this.waitForImageReady(descriptor.img);
        } catch {
          continue;
        }
        if (!this.isEnabled || generation !== this.operationGeneration) break;
        if (!this.isImageSourceCurrent(descriptor) || !this.canProcessImage(descriptor.img)) continue;
      }

      if (this.getImageExclusionReason(descriptor.img)) continue;
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
    this.refreshImageSource(img);
    if (this.processedImages.has(img) || this.pendingImages.has(img) || this.runningImage === img) return false;
    const source = this.getCanonicalSource(img);
    if (this.imageLoadFailures.has(source) || this.imageLoadFailures.has(this.getLoadingIdentity(img))) return false;
    if (Array.from(this.activeJobs.values()).some(job => job.originalSrc === source)) return false;
    const completed = this.completedJobs.get(this.getImageCacheDescriptor(img, this.settings || {}).cacheKey || source);
    if (completed && completed.originalSrc === source && img.isConnected) {
      this.setTranslatedImageSource(img, completed.data);
      this.processedImages.add(img);
      this.wrapImage(img);
      return false;
    }
    if (source && [...Array.from(this.processedImages).filter(other => other.isConnected), ...this.pendingImages, this.runningImage].some(
      other => other && other !== img && this.getCanonicalSource(other) === source
    )) return false;
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

  orderImagesByPage(images) {
    const entries = images.map((img, index) => ({ img, index, rect: img.getBoundingClientRect() }))
      .sort((a, b) => a.rect.top - b.rect.top || a.index - b.index);
    const rows = [];
    for (const entry of entries) {
      const row = rows[rows.length - 1];
      if (!row || entry.rect.top - row[0].rect.top > 4) rows.push([entry]);
      else row.push(entry);
    }
    const direction = this.settings?.readingDirection === 'ltr' ? 1 : -1;
    return rows.flatMap(row => row.sort((a, b) =>
      direction * (a.rect.left - b.rect.left) || a.index - b.index).map(entry => entry.img));
  }

  rememberImageOrder(img) {
    const source = this.getCanonicalSource(img);
    if (!this.sourceOrder.has(source)) this.sourceOrder.set(source, this.sourceOrder.size);
    return this.sourceOrder.get(source);
  }

  prioritizeProcessingQueue() {
    // Readers may recreate <img> nodes while scrolling. A source is one page,
    // even when several DOM nodes reference it.
    const sources = new Set();
    const images = new Set();
    this.processingQueue = this.processingQueue.map(img => {
      if (img.isConnected) return img;
      const source = this.getCanonicalSource(img);
      const replacement = this.getPageImages().find(candidate => candidate.isConnected &&
        this.getCanonicalSource(candidate) === source && this.isMangaImage(candidate));
      if (!replacement) return img;
      const manual = this.manualRequests.get(img);
      this.manualRequests.delete(img);
      this.pendingImages.delete(img);
      this.setQueuedState(img, false);
      if (manual) this.manualRequests.set(replacement, manual);
      this.pendingImages.add(replacement);
      this.wrapImage(replacement);
      this.setQueuedState(replacement, true);
      return replacement;
    });
    this.processingQueue = this.processingQueue.filter(img => {
      if (images.has(img)) return false;
      images.add(img);
      const source = this.getCanonicalSource(img);
      if (!img.isConnected || (source && sources.has(source))) {
        this.pendingImages.delete(img);
        this.manualRequests.delete(img);
        this.setQueuedState(img, false);
        return false;
      }
      if (source) sources.add(source);
      return true;
    });
    if (this.prioritySource && !sources.has(this.prioritySource) &&
        !Array.from(this.activeJobs.values()).some(job => job.originalSrc === this.prioritySource)) {
      this.prioritySource = null;
    }
    this.processingQueue.forEach(img => this.rememberImageOrder(img));
    this.processingQueue.sort((a, b) => {
      const aSource = this.getCanonicalSource(a);
      const bSource = this.getCanonicalSource(b);
      if ((aSource === this.prioritySource) !== (bSource === this.prioritySource)) return aSource === this.prioritySource ? -1 : 1;
      return this.sourceOrder.get(aSource) - this.sourceOrder.get(bSource);
    });
  }

  setupImageViewSync() {
    const sync = () => this.syncActiveJobImages();
    window.addEventListener('scroll', sync, { passive: true });
    window.addEventListener('resize', sync, { passive: true });
  }

  pumpQueue() {
    if ((!this.isEnabled && this.manualRequests.size === 0) || this.runningJob) return;
    this.prioritizeProcessingQueue();

    while (this.processingQueue.length > 0) {
      const first = this.processingQueue[0];
      if ((this.isScanning || !this.batchReady) && !this.manualRequests.has(first)) return;
      const img = this.processingQueue.shift();
      this.setQueuedState(img, false);
      if (!img.isConnected || !this.isMangaImage(img)) {
        if (this.getCanonicalSource(img) === this.prioritySource) this.prioritySource = null;
        this.pendingImages.delete(img);
        this.manualRequests.delete(img);
        continue;
      }
      // Reserve the slot before any async settings/decode/network operation.
      // Keep it until settlement, even when cancelPendingWork clears activeJobs.
      this.runningImage = img;
      const generation = this.operationGeneration;
      this.runningJob = Promise.resolve().then(() => this.startTranslationJob(img, generation))
        .catch(error => console.warn('[MangaTranslator] Job failed:', error))
        .finally(() => {
          if (!this.processingQueue.includes(img) &&
              !Array.from(this.activeJobs.values()).some(job => job.img === img)) {
            this.pendingImages.delete(img);
          }
          this.runningJob = null;
          this.runningImage = null;
          this.pumpQueue();
        });
      return;
    }

    if (!this.isScanning && this.processingQueue.length === 0 && this.activeJobs.size === 0) {
      this.hideProgressBar();
    }
  }

  createJobId() {
    if (crypto.randomUUID) return crypto.randomUUID();
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  requestJobCancellation(job, requeue = true) {
    if (job.cancelRequested) {
      if (!requeue) job.requeueAfterCancel = false;
      return;
    }
    job.cancelRequested = true;
    job.requeueAfterCancel = requeue;
    this.handlePipelineProgress({ jobId: job.id, stage: 'cancelling' });
    if (job.submitted) {
      job.cancelReply = this.sendMessage({ action: 'cancelTranslationJob', jobId: job.id, apiUrl: job.settings.apiUrl });
    }
  }

  async waitForCancellation(job) {
    if (!job.submitted) return;
    let reply = job.cancelReply ? await job.cancelReply : null;
    // Keep the single slot occupied if the stream disappeared but the server
    // has not confirmed release. Never infer cancellation from a network error.
    while (!reply?.success || !reply.data?.settled) {
      this.handlePipelineProgress({ jobId: job.id, stage: 'cancelling' });
      await new Promise(resolve => setTimeout(resolve, 1000));
      reply = await this.sendMessage({ action: 'cancelTranslationJob', jobId: job.id, apiUrl: job.settings.apiUrl });
    }
  }

  requeueInterruptedJob(job) {
    const img = this.getPageImages().find(candidate => candidate.isConnected &&
      this.getCanonicalSource(candidate) === job.originalSrc && this.isMangaImage(candidate));
    if (!img) return; // A later DOM scan restores it using the saved source order.
    if (this.processingQueue.some(candidate => this.getCanonicalSource(candidate) === job.originalSrc)) return;
    this.pendingImages.add(img);
    if (job.manual) this.manualRequests.set(img, { settings: job.settings });
    this.processingQueue.push(img);
    this.setQueuedState(img, true);
  }

  bindJobImage(job) {
    const matches = candidate => candidate.isConnected && !this.getImageExclusionReason(candidate) &&
      this.getCanonicalSource(candidate) === job.originalSrc;
    const img = matches(job.img) ? job.img : this.getPageImages().find(matches);
    if (!img) return null;
    if (!job.submitted && job.img !== img) return null;
    if (job.img !== img) {
      this.finishImageLoading(null, job.img);
      this.pendingImages.delete(job.img);
      if (job.img.dataset.mtJobId === job.id) delete job.img.dataset.mtJobId;
      job.img = img;
    }
    this.processingQueue = this.processingQueue.filter(candidate => candidate !== img);
    this.pendingImages.add(img);
    img.dataset.mtJobId = job.id;
    this.wrapImage(img);
    this.setQueuedState(img, false);
    // Do not increment the loading reference count on each scroll event.
    if (!this.imageLoadingCounts.get(img)) this.beginImageLoading(null, img);
    else this.showImageLoading(null, img);
    return img;
  }

  syncActiveJobImages() {
    this.activeJobs.forEach(job => {
      if (job.generation !== this.operationGeneration) return;
      if (this.bindJobImage(job) && job.lastProgress) this.handlePipelineProgress(job.lastProgress);
    });
  }

  async startTranslationJob(img, generation = this.operationGeneration) {
    const jobId = this.createJobId();
    const manual = this.manualRequests.get(img);
    this.manualRequests.delete(img);
    const settings = { ...(manual?.settings || this.settings || await this.loadSettings()) };
    if ((!this.isEnabled && !manual) || generation !== this.operationGeneration || !img.isConnected || !this.isMangaImage(img)) return;
    const descriptor = this.getImageCacheDescriptor(img, settings);
    const originalSrc = descriptor.originalSrc;
    // A newer context selection can arrive during async settings loading.
    if (this.prioritySource && this.prioritySource !== originalSrc) {
      if (manual) this.manualRequests.set(img, manual);
      if (!this.processingQueue.includes(img)) this.processingQueue.push(img);
      return;
    }
    if (this.prioritySource === originalSrc) this.prioritySource = null;
    const job = {
      id: jobId,
      manual: Boolean(manual),
      img,
      originalSrc,
      generation,
      settings,
      originalWidth: img.naturalWidth,
      originalHeight: img.naturalHeight,
      cacheKey: descriptor.cacheKey,
      sourceSignature: descriptor.sourceSignature,
      sourceElements: descriptor.sourceElements
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
      if (this.activeJobs.get(jobId) !== job || generation !== this.operationGeneration) return;
      if (job.cancelRequested || result.cancelled) {
        if (!job.cancelRequested) this.requestJobCancellation(job);
        return;
      }

      if (result.success && result.data) {
        // Preserve a completed response even while its page is virtualized away.
        this.completedJobs.set(job.cacheKey || originalSrc, { originalSrc, data: result.data });
        img = this.bindJobImage(job);
        if (img) {
          if (img.dataset.mtCacheObjectUrl && window.MangaTranslationCache) {
            window.MangaTranslationCache.revokeImageUrl(img.dataset.mtCacheObjectUrl);
            delete img.dataset.mtCacheObjectUrl;
          }
          this.setTranslatedImageSource(img, result.data);
          if (!this.processedImages.has(img)) this.totalProcessed++;
          this.processedImages.add(img);
          this.failedImages.delete(img);
        }

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
            llmMergeOcr: settings.llmMergeOcr !== false,
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
        // A failed transport does not prove that the native worker has stopped.
        if (job.submitted) this.requestJobCancellation(job, false);
        this.recordImageFailure(job.img, result.error);
        console.warn('[MangaTranslator] Inpaint failed:', result.error);
      }
    } catch (e) {
      if (this.activeJobs.get(jobId) !== job) return;
      if (job.cancelRequested) return;
      if (e && (e.code === 'IMAGE_NOT_READY' || e.code === 'IMAGE_SOURCE_CHANGED')) {
        this.scheduleScan();
      } else {
        this.recordImageFailure(job.img, e && e.message || String(e));
      }
      console.error('[MangaTranslator] Failed to translate image:', e);
    } finally {
      if (job.cancelRequested) await this.waitForCancellation(job);
      const ownsJob = this.activeJobs.get(jobId) === job;
      if (ownsJob) {
        this.activeJobs.delete(jobId);
        this.pendingImages.delete(job.img);
        if (job.img.dataset.mtJobId === jobId) delete job.img.dataset.mtJobId;
        this.finishImageLoading(null, job.img);
        if (job.cancelRequested && job.requeueAfterCancel && generation === this.operationGeneration) {
          this.requeueInterruptedJob(job);
        }
      }
      this.updateProgressBar();
      this.pumpQueue();
    }
  }

  async translateImage(job) {
    const settings = job.settings;
    const fileData = await this.getImageDataUrl(job.img, job.originalSrc, job);
    if (job.cancelRequested) return { success: false, cancelled: true };
    this.assertImageSourceCurrent(job);
    if ((!this.isEnabled && !job.manual) || job.generation !== this.operationGeneration || this.activeJobs.get(job.id) !== job) {
      const error = new Error('The translation request was cancelled.');
      error.code = 'JOB_CANCELLED';
      throw error;
    }

    job.submitted = true;
    return this.sendMessage({
      action: 'inpaintPageStream',
      data: {
        fileData: fileData,
        imageSrc: job.originalSrc,
        source: /^(data|blob):/i.test(job.originalSrc) ? 'browser image' : job.originalSrc.slice(0, 8192),
        jobId: job.id,
        apiUrl: settings.apiUrl,
        mimeType: 'image/jpeg',
        target_lang: settings.targetLang,
        translator: settings.translator || 'llm',
        llm_merge_ocr: String(settings.llmMergeOcr !== false),
        reading_direction: settings.readingDirection,
        typeset: 'true',
        font_scale: String(settings.fontScale),
        all_caps: String(settings.allCaps)
      }
    });
  }

  handlePipelineProgress(request) {
    const contextJob = this.contextMenuLoadingSources.get(request.jobId);
    const job = contextJob || this.activeJobs.get(request.jobId);
    if (job?.cancelRequested && request.stage !== 'cancelling') return;
    if (job && !contextJob) {
      job.lastProgress = request;
      this.bindJobImage(job);
    }
    if (this.isMinimalLoadingStyle()) return;
    if (!job || !job.img.isConnected || job.generation !== this.operationGeneration) return;
    if (contextJob) {
      if (this.latestContextJobs.get(job.img) !== request.jobId || !this.isImageSourceCurrent(job)) return;
    } else if (job.img.dataset.mtJobId !== request.jobId) return;
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
      render: 'Rendering...',
      cancelling: 'Cancelling; waiting for server...'
    };

    const label = stageNames[request.stage] || 'Translate...';
    if (text && text.textContent !== label) text.textContent = label;

    if (this.progressBar && !contextJob) {
      const pText = this.progressBar.querySelector('.manga-translator-progress-text');
      if (pText) {
        const currentIdx = this.totalProcessed + 1;
        const total = this.totalImages || currentIdx;
        const progressLabel = `Page ${currentIdx}/${total} • ${label}`;
        if (pText.textContent !== progressLabel) pText.textContent = progressLabel;
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
    const queued = document.createElement('span');
    queued.className = 'manga-translator-queue-label';
    queued.textContent = 'Queued';
    queued.setAttribute('role', 'status');
    queued.style.cssText = 'display:none;position:absolute;left:8px;top:8px;z-index:10;padding:4px 8px;background:#172033;color:white;border-radius:4px;font:12px sans-serif;pointer-events:none';
    wrapper.appendChild(queued);
  }

  setQueuedState(img, queued) {
    const label = img.closest('.manga-translator-wrapper')?.querySelector('.manga-translator-queue-label');
    if (label) label.style.display = queued ? 'block' : 'none';
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
      const job = this.activeJobs.get(img.dataset.mtJobId);
      if (text && !job?.lastProgress && text.textContent !== 'Detecting bubbles...') text.textContent = 'Detecting bubbles...';
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
    const matches = img => img.isConnected &&
      (this.getCanonicalSource(img) === srcUrl || img.currentSrc === srcUrl || img.src === srcUrl);
    for (const img of this.processedImages) {
      if (matches(img)) return img;
    }
    return Array.from(document.images).find(matches) || null;
  }

  applyInpaintBySrc(jobId, dataUrl) {
    const job = this.contextMenuLoadingSources.get(jobId);
    if (!job || typeof dataUrl !== 'string' || !dataUrl) return;
    const { img, settings, originalSrc, originalWidth, originalHeight, cacheKey } = job;
    if (job.generation !== this.operationGeneration || !img.isConnected ||
        this.latestContextJobs.get(img) !== jobId) return;
    if (!this.isImageSourceCurrent(job)) {
      if (this.isEnabled) this.scheduleScan();
      return;
    }
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
        llmMergeOcr: settings.llmMergeOcr !== false,
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

  startContextMenuLoading(srcUrl, jobId, requestSettings, selectedAt = Date.now() * 1000) {
    if (selectedAt < this.latestContextSelection) return { success: true, status: 'superseded' };
    this.latestContextSelection = selectedAt;
    let img = this.findImageBySrc(srcUrl);
    if (!img || !jobId || !requestSettings || this.contextMenuLoadingSources.has(jobId)) {
      return { success: false, error: 'The image or translation request is no longer available.' };
    }
    const excluded = this.getImageExclusionReason(img);
    if (excluded) return { success: false, error: excluded };
    if (!img.complete) return { success: false, error: 'Wait for the original image to finish loading.' };
    const source = this.getCanonicalSource(img);
    const active = Array.from(this.activeJobs.values()).find(job => job.originalSrc === source);
    if (active && !active.cancelRequested) {
      this.prioritySource = null;
      return { success: true, status: 'already_processing' };
    }
    const cacheKey = this.getImageCacheDescriptor(img, requestSettings).cacheKey;
    const completed = this.completedJobs.get(cacheKey || source);
    if (completed || (this.processedImages.has(img) && img.dataset.mtCacheKey === cacheKey)) {
      this.prioritySource = null;
      if (completed) this.setTranslatedImageSource(img, completed.data);
      return { success: true, status: 'already_completed' };
    }
    this.prioritySource = source;
    if (active) {
      active.manual = true;
      active.requeueAfterCancel = true;
      return { success: true, status: 'queued' };
    }
    img = this.processingQueue.find(candidate => this.getCanonicalSource(candidate) === source && candidate.isConnected) || img;
    this.manualRequests.set(img, { settings: { ...requestSettings } });
    this.imageLoadFailures.delete(source);
    this.failedImages.delete(img);
    this.pendingImages.add(img);
    this.wrapImage(img);
    this.setQueuedState(img, true);
    if (!this.processingQueue.includes(img)) this.processingQueue.push(img);
    this.activeJobs.forEach(job => this.requestJobCancellation(job));
    this.pumpQueue();
    return { success: true, status: 'queued' };
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
    const uniqueSources = images => new Set([...images].map(img => this.getCanonicalSource(img) || img));
    const remaining = uniqueSources(this.processingQueue).size;
    const completedSources = new Set([...uniqueSources(this.processedImages),
      ...Array.from(this.completedJobs.values(), job => job.originalSrc)]);
    const total = new Set([...uniqueSources(new Set([
      ...this.processedImages, ...this.pendingImages, ...this.failedImages.keys(),
      ...this.processingQueue, ...Array.from(this.activeJobs.values(), job => job.img)
    ])), ...completedSources, ...Array.from(this.activeJobs.values(), job => job.originalSrc)]).size;
    this.totalImages = total;
    this.totalProcessed = completedSources.size;
    if (!this.progressBar) return;
    const text = this.progressBar.querySelector('.manga-translator-progress-text');
    const fill = this.progressBar.querySelector('.manga-translator-progress-fill');
    const pct = total === 0 ? 0 : (this.totalProcessed / total) * 100;

    text.textContent = `Translated ${this.totalProcessed}/${total} pages • Active: ${this.activeJobs.size} • Queued: ${remaining}`;
    fill.style.width = `${pct}%`;
    const active = Array.from(this.activeJobs.values())[0];
    if (active?.lastProgress) this.handlePipelineProgress(active.lastProgress);
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
      this.syncActiveJobImages();
      let hasNewImages = false;
      for (const mutation of mutations) {
        if (mutation.type === 'attributes' && ['IMG', 'SOURCE'].includes(mutation.target.nodeName)) {
          const images = mutation.target.nodeName === 'IMG'
            ? [mutation.target] : Array.from(mutation.target.parentElement?.querySelectorAll('img') || []);
          images.forEach(img => {
            if (this.canProcessImage(img)) hasNewImages = true;
          });
          continue;
        }
        for (const node of mutation.addedNodes) {
          if (node.nodeName === 'IMG' || (node.querySelectorAll && node.querySelectorAll('img').length > 0)) {
            hasNewImages = true;
            break;
          }
        }
      }
      if (hasNewImages && this.isEnabled) {
        this.scheduleScan(600);
      }
    });
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['src', 'srcset', 'sizes', 'media', 'type',
        'data-src', 'data-lazy-src', 'data-original', 'data-url', 'data-highres']
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
        this.scheduleScan(300);
      }
    }, true);
  }

  async waitForImageReady(img, timeoutMs = 15000) {
    if (img.complete && img.naturalWidth > 0 && img.naturalHeight > 0) {
      try {
        await img.decode();
      } catch (error) { throw new Error('Image decoding failed: ' + error.message); }
      return;
    }

    await new Promise((resolve, reject) => {
      let timeoutId = null;
      const cleanup = () => {
        img.removeEventListener('load', onLoad);
        img.removeEventListener('error', onError);
        if (timeoutId) clearTimeout(timeoutId);
      };
      const onLoad = async () => {
        cleanup();
        try {
          await img.decode();
        } catch (error) { reject(new Error('Image decoding failed: ' + error.message)); return; }
        resolve();
      };
      const onError = () => {
        cleanup();
        reject(new Error('Image failed to load on page.'));
      };

      img.addEventListener('load', onLoad, { once: true });
      img.addEventListener('error', onError, { once: true });
      timeoutId = setTimeout(() => {
        cleanup();
        reject(new Error('Timed out waiting for image load.'));
      }, timeoutMs);
      // Loading may have finished between the first check and listener setup.
      if (img.complete) {
        if (img.naturalWidth > 0 && img.naturalHeight > 0) onLoad();
        else onError();
      }
    });
  }

  async getImageDataUrl(img, canonicalSrc, descriptor = this.getImageCacheDescriptor(img, this.settings)) {
    this.assertImageSourceCurrent(descriptor);
    await this.waitForImageReady(img);
    this.assertImageSourceCurrent(descriptor);
    if (!img.complete || img.naturalWidth <= 0 || img.naturalHeight <= 0) {
      const error = new Error('The original image has not finished loading.');
      error.code = 'IMAGE_NOT_READY';
      throw error;
    }

    try {
      if (this.hasTranslatedImageSource(img) || this.getCurrentImageSource(img) !== canonicalSrc) {
        throw new Error('Fetch the original source instead of capturing different pixels.');
      }
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Canvas 2D context is unavailable.');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      this.assertImageSourceCurrent(descriptor);
      return canvas.toDataURL('image/jpeg', 0.95);
    } catch (error) {
      if (error.code === 'IMAGE_SOURCE_CHANGED') throw error;
      const fetchTarget = canonicalSrc || this.getCanonicalSource(img) || img.currentSrc || img.src;
      if (fetchTarget && fetchTarget.startsWith('blob:')) {
        const blob = await fetch(fetchTarget).then(r => r.blob());
        const data = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result);
          reader.onerror = () => reject(new Error('Failed to read blob'));
          reader.readAsDataURL(blob);
        });
        this.assertImageSourceCurrent(descriptor);
        return data;
      }
      const result = await this.sendMessage({ action: 'fetchImage', url: fetchTarget });
      this.assertImageSourceCurrent(descriptor);
      if (result.success && result.data) return result.data;
      throw new Error('Cannot access image: ' + (result.error || 'Failed to capture image data.'));
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
    this.activeJobs.forEach(job => this.requestJobCancellation(job, false));
    this.completedJobs.clear();
    this.sourceOrder.clear();
    this.prioritySource = null;
    this.batchReady = false;
    this.imageLoadFailures.clear();
    this.operationGeneration++;
    this.scanToken = null;
    this.cacheClearGeneration = null;
    this.isScanning = false;
    if (this.scanDebounceTimer) clearTimeout(this.scanDebounceTimer);
    this.scanDebounceTimer = null;
    this.processingQueue.forEach(img => this.setQueuedState(img, false));
    this.processingQueue = [];
    this.manualRequests.clear();
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
