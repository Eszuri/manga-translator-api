const MangaTranslationCache = {
  dbName: 'MangaTranslatorCache',
  dbVersion: 1,
  storeName: 'translations',
  _dbPromise: null,

  openDB() {
    if (this._dbPromise) return this._dbPromise;

    this._dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          const store = db.createObjectStore(this.storeName, { keyPath: 'cacheKey' });
          store.createIndex('pageUrl', 'pageUrl', { unique: false });
          store.createIndex('timestamp', 'timestamp', { unique: false });
        }
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => {
        this._dbPromise = null;
        reject(request.error);
      };
    });

    return this._dbPromise;
  },

  normalizeUrlForKey(value, keepSearch = false) {
    if (!value) return '';

    try {
      const url = new URL(value, window.location.href);
      return `${url.origin}${url.pathname}${keepSearch ? url.search : ''}`;
    } catch {
      const withoutHash = String(value).split('#')[0];
      return keepSearch ? withoutHash : withoutHash.split('?')[0];
    }
  },

  buildCacheKey(settings, originalSrc, pageUrl = window.location.href, imageIndex = -1) {
    const lang = (settings && settings.targetLang) || 'id';
    const translator = (settings && settings.translator) || 'google';
    const direction = (settings && settings.readingDirection) || 'rtl';
    const fontScale = (settings && settings.fontScale) ?? 1.0;
    const allCaps = (settings && settings.allCaps) ?? true;
    const apiUrl = (settings && settings.apiUrl) || 'http://127.0.0.1:8000';
    const source = this.normalizeUrlForKey(originalSrc, true);

    if (source && (source.startsWith('http://') || source.startsWith('https://'))) {
      return `v7_${JSON.stringify([lang, translator, direction, fontScale, allCaps, apiUrl, source])}`;
    }

    const page = this.normalizeUrlForKey(pageUrl, false);
    return `v7_fallback_${JSON.stringify([lang, translator, direction, fontScale, allCaps, apiUrl, page, imageIndex, source])}`;
  },

  async getCachedTranslation(cacheKey) {
    if (!cacheKey) return null;
    try {
      const db = await this.openDB();
      return new Promise((resolve) => {
        const tx = db.transaction(this.storeName, 'readonly');
        const store = tx.objectStore(this.storeName);
        const req = store.get(cacheKey);
        req.onsuccess = () => resolve(req.result || null);
        req.onerror = () => resolve(null);
      });
    } catch (e) {
      console.warn('[MangaTranslationCache] Read error:', e);
      return null;
    }
  },

  async getCachedTranslations(cacheKeys) {
    const uniqueKeys = [...new Set((cacheKeys || []).filter(Boolean))];
    if (uniqueKeys.length === 0) return new Map();

    try {
      const db = await this.openDB();
      return await new Promise((resolve) => {
        const tx = db.transaction(this.storeName, 'readonly');
        const store = tx.objectStore(this.storeName);
        const results = new Map();

        uniqueKeys.forEach((cacheKey) => {
          const request = store.get(cacheKey);
          request.onsuccess = () => {
            if (request.result) results.set(cacheKey, request.result);
          };
        });

        tx.oncomplete = () => resolve(results);
        tx.onerror = () => resolve(new Map());
        tx.onabort = () => resolve(new Map());
      });
    } catch (e) {
      console.warn('[MangaTranslationCache] Bulk read error:', e);
      return new Map();
    }
  },

  dataUrlToBlob(dataUrl) {
    if (!dataUrl || !dataUrl.startsWith('data:')) return null;

    try {
      const separatorIndex = dataUrl.indexOf(',');
      if (separatorIndex < 0) return null;

      const header = dataUrl.slice(0, separatorIndex);
      const encoded = dataUrl.slice(separatorIndex + 1);
      const mimeType = header.match(/^data:([^;,]+)/)?.[1] || 'image/png';
      const binary = header.includes(';base64') ? atob(encoded) : decodeURIComponent(encoded);
      const bytes = new Uint8Array(binary.length);

      for (let index = 0; index < binary.length; index++) {
        bytes[index] = binary.charCodeAt(index);
      }

      return new Blob([bytes], { type: mimeType });
    } catch (e) {
      console.warn('[MangaTranslationCache] Blob conversion error:', e);
      return null;
    }
  },

  createImageUrl(entry) {
    if (entry && entry.translatedBlob instanceof Blob) {
      return URL.createObjectURL(entry.translatedBlob);
    }
    return (entry && entry.translatedData) || null;
  },

  revokeImageUrl(imageUrl) {
    if (imageUrl && imageUrl.startsWith('blob:')) URL.revokeObjectURL(imageUrl);
  },

  async saveCachedTranslation(entry) {
    if (!entry || !entry.cacheKey || (!entry.translatedData && !entry.translatedBlob)) return;
    try {
      const translatedBlob = entry.translatedBlob || this.dataUrlToBlob(entry.translatedData);
      if (!translatedBlob) return;

      const db = await this.openDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(this.storeName, 'readwrite');
        const store = tx.objectStore(this.storeName);
        store.put({
          cacheKey: entry.cacheKey,
          pageUrl: entry.pageUrl || window.location.href,
          originalSrc: entry.originalSrc,
          translatedBlob,
          originalWidth: Number(entry.originalWidth) || 0,
          originalHeight: Number(entry.originalHeight) || 0,
          targetLang: entry.targetLang || 'id',
          translator: entry.translator || 'google',
          readingDirection: entry.readingDirection || 'rtl',
          timestamp: entry.timestamp || Date.now()
        });
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    } catch (e) {
      console.warn('[MangaTranslationCache] Save error:', e);
    }
  },

  async clearPageCache(pageUrl) {
    const targetUrl = pageUrl || window.location.href;
    try {
      const db = await this.openDB();
      return new Promise((resolve) => {
        const tx = db.transaction(this.storeName, 'readwrite');
        const store = tx.objectStore(this.storeName);
        const index = store.index('pageUrl');
        const req = index.openCursor(IDBKeyRange.only(targetUrl));

        req.onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            cursor.delete();
            cursor.continue();
          } else {
            resolve();
          }
        };
        req.onerror = () => resolve();
      });
    } catch (e) {
      console.warn('[MangaTranslationCache] Clear page error:', e);
    }
  },

  async pruneOldCache(maxAgeDays = 7) {
    const maxAgeMs = maxAgeDays * 24 * 60 * 60 * 1000;
    const cutoff = Date.now() - maxAgeMs;

    try {
      const db = await this.openDB();
      return new Promise((resolve) => {
        const tx = db.transaction(this.storeName, 'readwrite');
        const store = tx.objectStore(this.storeName);
        const index = store.index('timestamp');
        const req = index.openCursor(IDBKeyRange.upperBound(cutoff));

        req.onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            cursor.delete();
            cursor.continue();
          } else {
            resolve();
          }
        };
        req.onerror = () => resolve();
      });
    } catch (e) {
      console.warn('[MangaTranslationCache] Prune error:', e);
    }
  }
};

if (typeof window !== 'undefined') {
  window.MangaTranslationCache = MangaTranslationCache;
}
