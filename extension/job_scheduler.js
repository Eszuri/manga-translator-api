// One browser-wide stream slot. The backend also serializes native work.
class TranslationScheduler {
  constructor(run, cancel, notify) {
    this.run = run;
    this.cancel = cancel;
    this.notify = notify;
    this.queue = [];
    this.active = null;
    this.lastTab = null;
    this.focusedTab = null;
    this.handoffTimer = null;
    this.tabWork = new Map();
  }

  setTabWork(tabId, pending) {
    this.tabWork.set(tabId, Boolean(pending));
    this.pump();
  }

  setFocusedTab(tabId) {
    this.focusedTab = tabId;
    // Focus changes reorder pending work, never interrupt an uploaded image.
    this.notifyWaiting();
    this.pump();
  }

  enqueue(data, sender) {
    if (!data.jobId || sender.tab?.id == null) {
      return Promise.resolve({ success: false, error: 'A tab and unique job ID are required.' });
    }
    if ([this.active, ...this.queue].some(job => job?.data.jobId === data.jobId)) {
      return Promise.resolve({ success: false, error: 'Duplicate translation job ID.' });
    }
    return new Promise(resolve => {
      const job = { data, sender, resolve, cancelled: false };
      this.queue.push(job);
      this.notify(job, 'waiting_turn');
      // Notify through Chrome periodically while queued; no server upload yet.
      job.timer = setInterval(() => this.notify(job, 'waiting_turn'), 20000);
      if (data.priority && this.active) this.stop(this.active);
      this.pump();
    });
  }

  activeInfo() {
    const job = this.active;
    if (!job) return null;
    return {
      tabId: job.sender.tab.id,
      title: job.data.pageTitle || job.sender.tab.title || 'Manga page',
      pageUrl: job.sender.url || job.sender.tab.url || '',
      imageSrc: job.data.imageSrc || '',
      cancelling: job.cancelled
    };
  }

  notifyWaiting() {
    for (const job of this.queue) this.notify(job, 'waiting_turn');
  }

  stop(job) {
    if (job.cancelled) return job.stopping;
    job.cancelled = true;
    this.notifyWaiting();
    if (job !== this.active) {
      this.queue = this.queue.filter(candidate => candidate !== job);
      clearInterval(job.timer);
      job.resolve({ success: false, cancelled: true });
      return Promise.resolve({ success: true, data: { settled: true } });
    }
    job.stopping = (async () => {
      // A lost stream is not evidence that GPU work has stopped.
      for (;;) {
        let reply;
        try { reply = await this.cancel(job); } catch {}
        if (reply?.success && reply.data?.settled) return reply;
        this.notify(job, 'cancelling');
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    })();
    return job.stopping;
  }

  cancelOwned(id, sender) {
    const job = [this.active, ...this.queue].find(candidate => candidate &&
      candidate.data.jobId === id && candidate.sender.tab.id === sender.tab?.id &&
      candidate.sender.frameId === sender.frameId &&
      candidate.sender.documentId === sender.documentId);
    return job ? this.stop(job) : null;
  }

  closeTab(tabId) {
    this.tabWork.delete(tabId);
    if (this.focusedTab === tabId) this.focusedTab = null;
    for (const job of [this.active, ...this.queue]) {
      if (job?.sender.tab.id === tabId) this.stop(job);
    }
  }

  async pump() {
    if (this.active || this.handoffTimer || !this.queue.length) return;
    // Explicit selections win, followed by the last focused browser tab.
    let index = this.queue.findLastIndex(job => job.data.priority);
    if (index < 0) index = this.queue.findIndex(job => job.sender.tab.id === this.focusedTab);
    // The focused tab may be decoding/uploading its next image. An empty
    // background queue is not evidence that its page batch has finished.
    if (index < 0 && this.tabWork.get(this.focusedTab)) return;
    if (index < 0) index = this.queue.findIndex(job => job.sender.tab.id !== this.lastTab);
    if (index < 0) index = 0;
    const job = this.queue.splice(index, 1)[0];
    this.active = job;
    this.notifyWaiting();
    clearInterval(job.timer);
    try {
      const result = await this.run(job.data, job.sender);
      const interrupted = job.cancelled;
      if (!result.success && !job.cancelled) this.stop(job);
      if (job.stopping) await job.stopping;
      job.resolve(interrupted ? { success: false, cancelled: true } : result);
    } catch (error) {
      await this.stop(job);
      job.resolve({ success: false, error: error.message });
    } finally {
      this.lastTab = job.sender.tab.id;
      this.active = null;
      // Content scripts submit one image at a time. Give the finishing tab a
      // short handoff window to submit its next image before selecting a tab.
      this.handoffTimer = setTimeout(() => {
        this.handoffTimer = null;
        this.pump();
      }, 150);
    }
  }
}
