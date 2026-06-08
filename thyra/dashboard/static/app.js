/* Thyra Memory Dashboard — Alpine.js component */
function dashboard() {
    return {
        /* ── namespace ───────────────────────────────────────────────────────── */
        namespaces: [],
        activeNs: { user_id: 'default', agent_id: 'claude-code-global' },

        /* ── header status ───────────────────────────────────────────────────── */
        status: {
            system_enabled: true,
            formation_enabled: true,
            max_memories: 0,
            active_memories: 0,
            archived_memories: 0,
            last_nightly: 0,
        },
        maxInput: 0,

        /* ── tabs ────────────────────────────────────────────────────────────── */
        tab: 'memories',

        /* ── memories ────────────────────────────────────────────────────────── */
        memories: [],
        memTotal: 0,
        memPage: 1,
        memPageSize: 50,
        memPages: 1,
        memSearch: '',
        memCat: '',
        memType: '',
        memSortBy: 'strength',
        memArchived: false,
        memProb: false,
        memLoading: false,
        memIsGlobal: false,
        _searchTimer: null,

        /* ── memory detail ───────────────────────────────────────────────────── */
        detail: null,

        /* ── stats ───────────────────────────────────────────────────────────── */
        stats: {},
        statsLoaded: false,

        /* ── activity ────────────────────────────────────────────────────────── */
        activity: [],
        activityLoaded: false,

        /* ── logs ────────────────────────────────────────────────────────────── */
        logEntries: [],
        logSinceTs: 0,
        logAutoScroll: true,
        _logTimer: null,
        _statusTimer: null,

        /* ══════════════════════════════════════════════════════════════════════ */

        async init() {
            await this.loadNamespaces();
            await this.loadStatus();
            await this.loadMemories();

            this._statusTimer = setInterval(() => this.loadStatus(), 10000);
            this._logTimer = setInterval(() => {
                if (this.tab === 'logs') this.pollLogs();
            }, 3000);
        },

        /* ── namespaces ──────────────────────────────────────────────────────── */
        async loadNamespaces() {
            try {
                const [nsResp, curResp] = await Promise.all([
                    fetch('/api/namespaces').then(r => r.json()),
                    fetch('/api/current').then(r => r.json()),
                ]);
                this.namespaces = nsResp.namespaces || [];
                if (this.namespaces.length) {
                    // Prefer the namespace that matches the active Claude session,
                    // then fall back to the one with the most memories.
                    const cur = curResp;
                    const preferred =
                        this.namespaces.find(
                            n => n.user_id === cur.user_id && n.agent_id === cur.agent_id
                        ) || this.namespaces[0];
                    this.activeNs = { user_id: preferred.user_id, agent_id: preferred.agent_id };
                    this.maxInput = 0;
                }
            } catch (e) { console.error('loadNamespaces', e); }
        },

        nsLabel(ns) {
            return `${ns.agent_id} (${ns.mem_count})`;
        },

        async onNsChange(event) {
            const idx = parseInt(event.target.value);
            this.activeNs = { user_id: this.namespaces[idx].user_id, agent_id: this.namespaces[idx].agent_id };
            this.statsLoaded = false;
            this.activityLoaded = false;
            this.logEntries = [];
            this.logSinceTs = 0;
            await this.loadStatus();
            await this.loadMemories();
        },

        nsIndex() {
            return this.namespaces.findIndex(
                n => n.user_id === this.activeNs.user_id && n.agent_id === this.activeNs.agent_id
            );
        },

        /* ── status ──────────────────────────────────────────────────────────── */
        async loadStatus() {
            try {
                const params = this._nsParams();
                const r = await fetch(`/api/status?${params}`).then(r => r.json());
                this.status = r;
                this.maxInput = r.max_memories;
            } catch (e) { console.error('loadStatus', e); }
        },

        async toggleSystem() {
            const enabled = !this.status.system_enabled;
            try {
                await this._post('/api/toggle/system', { ...this.activeNs, enabled });
                this.status.system_enabled = enabled;
            } catch (e) { console.error('toggleSystem', e); }
        },

        async toggleFormation() {
            const enabled = !this.status.formation_enabled;
            try {
                await this._post('/api/toggle/formation', { ...this.activeNs, enabled });
                this.status.formation_enabled = enabled;
            } catch (e) { console.error('toggleFormation', e); }
        },

        async applyMaxMemories() {
            const count = parseInt(this.maxInput) || 0;
            try {
                await this._post('/api/settings/max-memories', { ...this.activeNs, count });
                this.status.max_memories = count;
            } catch (e) { console.error('applyMaxMemories', e); }
        },

        /* ── tab switching ───────────────────────────────────────────────────── */
        async switchTab(t) {
            this.tab = t;
            if (t === 'stats' && !this.statsLoaded) await this.loadStats();
            if (t === 'activity' && !this.activityLoaded) await this.loadActivity();
        },

        /* ── memories ────────────────────────────────────────────────────────── */
        async loadMemories() {
            this.memPage = 1;
            await this._fetchMem();
        },

        onSearchInput() {
            clearTimeout(this._searchTimer);
            this._searchTimer = setTimeout(() => this.loadMemories(), 300);
        },

        async _fetchMem() {
            this.memLoading = true;
            try {
                const p = new URLSearchParams({
                    ...Object.fromEntries(
                        Object.entries(this.activeNs).map(([k, v]) => [k, v])
                    ),
                    page: this.memPage,
                    page_size: this.memPageSize,
                    sort_by: this.memSortBy,
                    archived: this.memArchived,
                });
                if (this.memSearch.trim()) {
                    p.set('search', this.memSearch.trim());
                    p.set('global_search', 'true');
                }
                if (this.memCat) p.set('category', this.memCat);
                if (this.memType) p.set('memory_type', this.memType);
                if (this.memProb) p.set('probationary', 'true');

                const r = await fetch('/api/memories?' + p).then(r => r.json());
                this.memories = r.memories || [];
                this.memTotal = r.total || 0;
                this.memPages = r.pages || 1;
                this.memIsGlobal = r.global_search || false;
            } catch (e) {
                console.error('_fetchMem', e);
            } finally {
                this.memLoading = false;
            }
        },

        async prevPage() {
            if (this.memPage > 1) { this.memPage--; await this._fetchMem(); }
        },

        async nextPage() {
            if (this.memPage < this.memPages) { this.memPage++; await this._fetchMem(); }
        },

        strPct(m) {
            return Math.min(100, m.current_level * 10).toFixed(1);
        },

        /* ── detail modal ────────────────────────────────────────────────────── */
        async openDetail(mem) {
            const id = (typeof mem === 'object') ? mem.id : mem;
            const ns = (typeof mem === 'object')
                ? { user_id: mem.user_id, agent_id: mem.agent_id }
                : this.activeNs;
            try {
                const params = new URLSearchParams(ns).toString();
                const resp = await fetch(`/api/memories/${id}?${params}`);
                if (!resp.ok) { console.error('openDetail: not found', id); return; }
                this.detail = await resp.json();
            } catch (e) { console.error('openDetail', e); }
        },

        closeDetail() { this.detail = null; },

        async deleteMemory(id) {
            if (!id) return;
            if (!confirm(`Delete memory ${id.slice(0,16)}…?\nThis cannot be undone.`)) return;
            const ns = { user_id: this.detail?.user_id || this.activeNs.user_id, agent_id: this.detail?.agent_id || this.activeNs.agent_id };
            try {
                await this._post(`/api/memories/${id}/delete`, ns);
                this.detail = null;
                await this.loadMemories();
                await this.loadStatus();
            } catch (e) { console.error('deleteMemory', e); alert('Failed to delete memory. Check the console for details.'); }
        },

        async archiveMemory(id) {
            if (!id) return;
            if (!confirm(`Archive memory ${id.slice(0,16)}…?`)) return;
            const ns = { user_id: this.detail?.user_id || this.activeNs.user_id, agent_id: this.detail?.agent_id || this.activeNs.agent_id };
            try {
                await this._post(`/api/memories/${id}/archive`, ns);
                this.detail = null;
                await this.loadMemories();
                await this.loadStatus();
            } catch (e) { console.error('archiveMemory', e); alert('Failed to archive memory. Check the console for details.'); }
        },

        /* ── stats ───────────────────────────────────────────────────────────── */
        async loadStats() {
            try {
                const r = await fetch(`/api/stats?${this._nsParams()}`).then(r => r.json());
                this.stats = r;
                this.statsLoaded = true;
            } catch (e) { console.error('loadStats', e); }
        },

        catMaxCount() {
            if (!this.stats.by_category?.length) return 1;
            return Math.max(1, ...this.stats.by_category.map(c => c.count));
        },

        histMaxCount() {
            if (!this.stats.strength_histogram?.buckets) return 1;
            return Math.max(1, ...this.stats.strength_histogram.buckets);
        },

        histBarPct(count) {
            return Math.round((count / this.histMaxCount()) * 85);
        },

        lastNightlyLabel() {
            const ts = this.status.last_nightly;
            if (!ts) return 'never';
            return new Date(ts).toLocaleString();
        },

        maxMemLabel() {
            return this.status.max_memories === 0 ? 'unlimited' : String(this.status.max_memories);
        },

        /* ── activity ────────────────────────────────────────────────────────── */
        async loadActivity() {
            try {
                const r = await fetch(`/api/activity?${this._nsParams()}&limit=50`).then(r => r.json());
                this.activity = r.turns || [];
                this.activityLoaded = true;
            } catch (e) { console.error('loadActivity', e); }
        },

        /* ── logs ────────────────────────────────────────────────────────────── */
        async pollLogs() {
            try {
                const r = await fetch(`/api/logs?since_ts=${this.logSinceTs}&limit=100`).then(r => r.json());
                const entries = r.entries || [];
                if (entries.length) {
                    this.logEntries.push(...entries);
                    if (this.logEntries.length > 500) this.logEntries = this.logEntries.slice(-500);
                    this.logSinceTs = entries[entries.length - 1].ts;
                    if (this.logAutoScroll) {
                        this.$nextTick(() => {
                            const el = this.$refs.logOutput;
                            if (el) el.scrollTop = el.scrollHeight;
                        });
                    }
                }
            } catch (e) { /* silent */ }
        },

        clearLogs() { this.logEntries = []; },

        logLevelClass(level) {
            return 'log-' + level.toLowerCase();
        },

        /* ── helpers ─────────────────────────────────────────────────────────── */
        _nsParams() {
            return new URLSearchParams(this.activeNs).toString();
        },

        async _post(url, body) {
            const r = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
            return r.json();
        },

        fmtTime(ms) {
            if (!ms) return '—';
            return new Date(ms).toLocaleString();
        },

        fmtLogTs(ts) {
            return new Date(ts * 1000).toLocaleTimeString();
        },

        fmtShort(id) {
            return id ? id.slice(0, 20) + '…' : '—';
        },

        categories: [
            'constraints', 'identity', 'preferences', 'relationships', 'tasks',
            'goals', 'context', 'skills', 'habits', 'communication',
            'knowledge', 'events', 'health', 'finance', 'routines',
        ],
    };
}
