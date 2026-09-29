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

  buildCacheKey(settings, originalSrc) {
    const lang = (settings && settings.targetLang) || 'id';
    const translator = (settings && settings.translator) || 'google';
    const direction = (settings && settings.readingDirection) || 'rtl';
    const source = originalSrc || '';
    return `v2_${lang}_${translator}_${direction}_${source}`;
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

  async saveCachedTranslation(entry) {
    if (!entry || !entry.cacheKey || !entry.translatedData) return;
    try {
      const db = await this.openDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(this.storeName, 'readwrite');
        const store = tx.objectStore(this.storeName);
        store.put({
          cacheKey: entry.cacheKey,
          pageUrl: entry.pageUrl || window.location.href,
          originalSrc: entry.originalSrc,
          translatedData: entry.translatedData,
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
